"""M6D fixed 2000-step, 128-clip training of only the M6B-4 action-token adapter."""

from __future__ import annotations

import collections
import json
import math
import time
from pathlib import Path

import torch
import torch.nn.functional as F

from training.wan.m6d_select_128 import OUT, SEED, digest
from training.wan.vace_action_adapter import ACTION_TO_ID
from training.wan.vace_action_token_prototype import ActionTokenAdapter, BLOCKS, MODEL, temporal_mask
from training.wan.vace_action_token_smoke import save_json

STEPS = 2000
LR = 2e-4
CHECKPOINTS = (500, 1000, 1500, 2000)
PROMPT = "A first-person gameplay view in a retro 3D stone maze."


def load_cache() -> tuple[torch.Tensor, list[dict], list[dict]]:
    manifest_path = OUT / "subset_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    cache = OUT / "cache"
    index = json.loads((cache / "index.json").read_text())
    summary = json.loads((cache / "summary.json").read_text())
    rows = manifest["records"]
    if (manifest["clip_count"] != 128 or summary["status"] != "complete"
            or index["subset_manifest_sha256"] != digest(manifest_path)
            or index["sample_ids"] != manifest["clip_ids"]
            or index["prompt"] != PROMPT
            or not summary["native_vace_mask_unchanged"]):
        raise RuntimeError("128-sample subset or native VACE cache changed")
    prompt_path = Path(index["prompt_embedding_path"])
    if digest(prompt_path) != index["prompt_embedding_sha256"]:
        raise RuntimeError("Generic prompt embedding changed")
    prompt = torch.load(prompt_path, map_location="cpu", weights_only=True)
    if prompt.shape != (1, 512, 4096):
        raise RuntimeError("Prompt shape changed")
    samples = []
    for i, row in enumerate(rows):
        sample = torch.load(cache / f"sample_{i:03d}.pt", map_location="cpu", weights_only=True)
        expected = [[ACTION_TO_ID[action] for action in row["raw_actions"]]]
        if (sample["target"].shape != (1, 16, 5, 32, 56)
                or sample["control"].shape != (1, 96, 5, 32, 56)
                or sample["actions"].tolist() != expected
                or not bool(torch.isfinite(sample["target"]).all())
                or not bool(torch.isfinite(sample["control"]).all())
                or not bool(torch.all(sample["control"][:, 32:] == 1))):
            raise RuntimeError(f"Invalid cached training sample {i}: {row['global_clip_id']}")
        samples.append(sample)
    return prompt, samples, rows


def save_checkpoint(adapter: ActionTokenAdapter, optimizer: torch.optim.Optimizer,
                    step: int, record: dict, noise_generator: torch.Generator) -> None:
    directory = OUT / "checkpoints" / f"step_{step:04d}"
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "adapter.pt"
    temporary = directory / "adapter.partial.pt"
    state = {name: value.detach().cpu().contiguous() for name, value in adapter.state_dict().items()}
    torch.save(state, temporary)
    temporary.replace(path)
    save_json(directory / "adapter.json", {"step": step, "sha256": digest(path),
        "adapter_parameters": 3_684_864, "state_keys": sorted(state), "last_step": record})
    # Separate exact-resume state; the official adapter.pt remains adapter-only.
    resume = directory / "training_state.pt"
    temporary_resume = directory / "training_state.partial.pt"
    torch.save({"step": step, "optimizer": optimizer.state_dict(),
        "torch_rng": torch.get_rng_state(), "cuda_rng": torch.cuda.get_rng_state_all(),
        "noise_generator_state": noise_generator.get_state()}, temporary_resume)
    temporary_resume.replace(resume)


def run() -> dict:
    from diffusers import UniPCMultistepScheduler, WanVACETransformer3DModel

    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("BF16 CUDA is required")
    train_dir = OUT / "train"
    train_dir.mkdir(parents=True, exist_ok=True)
    log_path = train_dir / "steps.jsonl"
    if log_path.exists() and log_path.stat().st_size:
        raise FileExistsError("M6D training log exists; refusing to overwrite or silently restart")
    prompt_cpu, samples, rows = load_cache()
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    adapter = ActionTokenAdapter().to(device="cuda", dtype=torch.float32).train()
    trainable = sum(p.numel() for p in adapter.parameters() if p.requires_grad)
    if trainable != 3_684_864:
        raise RuntimeError(f"Adapter parameter count changed: {trainable}")
    if any(bool(torch.count_nonzero(site.out_proj.weight)) or bool(torch.count_nonzero(site.out_proj.bias))
           for site in adapter.sites.values()):
        raise RuntimeError("Adapter is not M6B-3 zero initialized")
    initial_path = train_dir / "initial_adapter.pt"
    torch.save({name: value.detach().cpu() for name, value in adapter.state_dict().items()}, initial_path)
    transformer = WanVACETransformer3DModel.from_pretrained(
        str(MODEL), subfolder="transformer", torch_dtype=torch.bfloat16, local_files_only=True
    ).to("cuda").eval().requires_grad_(False)
    transformer.enable_gradient_checkpointing()
    if any(p.requires_grad or p.grad is not None for p in transformer.parameters()):
        raise RuntimeError("A pretrained VACE parameter is trainable")
    base_versions = {name: p._version for name, p in transformer.named_parameters()}
    optimizer = torch.optim.AdamW(adapter.parameters(), lr=LR)
    if {id(p) for group in optimizer.param_groups for p in group["params"]} != {id(p) for p in adapter.parameters()}:
        raise RuntimeError("Optimizer includes pretrained parameters")
    scheduler = UniPCMultistepScheduler.from_pretrained(str(MODEL), subfolder="scheduler", local_files_only=True)
    scheduler.set_timesteps(1000)
    if scheduler.config.prediction_type != "flow_prediction" or scheduler.config.flow_shift != 3.0:
        raise RuntimeError("Fixed M6B-4 flow scheduler changed")
    timesteps = scheduler.timesteps.to("cuda")
    sigmas = scheduler.sigmas[:1000].to("cuda")
    order_generator = torch.Generator().manual_seed(SEED)
    order = [i for _ in range(math.ceil(STEPS / 128))
             for i in torch.randperm(128, generator=order_generator).tolist()][:STEPS]
    timestep_indices = torch.randint(0, 1000, (STEPS,), generator=torch.Generator().manual_seed(SEED + 1)).tolist()
    noise_generator = torch.Generator(device="cuda").manual_seed(SEED + 2)
    prompt = prompt_cpu.to("cuda")
    mask = temporal_mask(torch.device("cuda"))
    control_scale = torch.ones(len(transformer.config.vace_layers), device="cuda", dtype=torch.bfloat16)
    protocol = {"milestone": "M6D", "subset_manifest": str(OUT / "subset_manifest.json"),
        "subset_manifest_sha256": digest(OUT / "subset_manifest.json"),
        "initial_adapter_sha256": digest(initial_path),
        "initialization": "Fresh ActionTokenAdapter with torch.manual_seed(42), M6B-3 zero output projections; no M6B-4 trained weights loaded",
        "model": str(MODEL), "architecture": "unchanged M6B-3/M6B-4 action-token cross-attention",
        "blocks": list(BLOCKS), "trainable_adapter_parameters": trainable,
        "native_o0_and_mask_behavior_unchanged": True, "prompt": PROMPT,
        "seed": SEED, "optimizer": "AdamW", "learning_rate": LR,
        "batch_size": 1, "gradient_accumulation": 1, "backbone_dtype": "bfloat16",
        "optimizer_steps": STEPS, "future_loss_latent_positions": [1, 2, 3, 4],
        "fixed_checkpoint_steps": list(CHECKPOINTS), "official_checkpoint_step": 2000,
        "sampling": "Seed-42 independent shuffled 128-sample epochs, truncated after 2000 steps",
        "timestep_sampling": "Seed-43 randint from fixed 1000-step flow schedule",
        "diffusion_noise": "Seed-44 CUDA generator; BF16 noise as in M6B-4"}
    save_json(train_dir / "protocol.json", protocol)
    started = time.monotonic()
    torch.cuda.reset_peak_memory_stats()
    counts = collections.Counter()
    with log_path.open("w", encoding="utf-8") as log:
        for step, sample_index in enumerate(order, start=1):
            step_started = time.monotonic()
            sample = samples[sample_index]
            target = sample["target"].to("cuda")
            control = sample["control"].to("cuda")
            actions = sample["actions"].to("cuda")
            timestep_index = timestep_indices[step - 1]
            sigma = sigmas[timestep_index].float()
            timestep = timesteps[timestep_index].expand(1)
            noise = torch.randn(target.shape, device="cuda", dtype=torch.bfloat16, generator=noise_generator)
            noisy = ((1 - sigma) * target.float() + sigma * noise.float()).to(torch.bfloat16)
            velocity = noise.float() - target.float()
            optimizer.zero_grad(set_to_none=True)
            action_tokens = adapter.tokens(actions)
            hook_calls = collections.Counter()
            handles = []
            try:
                for block in BLOCKS:
                    def hook(_module, _inputs, hidden, *, site=block):
                        hook_calls[site] += 1
                        with torch.autocast("cuda", dtype=torch.bfloat16):
                            return adapter.sites[str(site)](hidden, action_tokens, mask)
                    handles.append(transformer.blocks[block].register_forward_hook(hook))
                prediction = transformer(hidden_states=noisy, timestep=timestep,
                    encoder_hidden_states=prompt, control_hidden_states=control,
                    control_hidden_states_scale=control_scale, return_dict=False)[0]
                if prediction.shape != target.shape:
                    raise RuntimeError(f"Unexpected denoiser shape at step {step}")
                loss = F.mse_loss(prediction.float()[:, :, 1:], velocity[:, :, 1:])
                if not bool(torch.isfinite(loss)):
                    raise FloatingPointError(f"Nonfinite future-only loss at step {step}")
                loss.backward()
            finally:
                for handle in handles:
                    handle.remove()
            if any(hook_calls[block] != 2 for block in BLOCKS):
                raise RuntimeError(f"Action hook skipped at step {step}: {hook_calls}")
            squares = 0.0
            by_block = collections.defaultdict(float)
            for name, parameter in adapter.named_parameters():
                gradient = parameter.grad
                if gradient is None:
                    if step == 1 and "out_proj" not in name:
                        continue
                    raise FloatingPointError(f"Missing adapter gradient at step {step}: {name}")
                if not bool(torch.isfinite(gradient).all()):
                    raise FloatingPointError(f"Nonfinite adapter gradient at step {step}: {name}")
                part = float(torch.sum(gradient.float().square()))
                squares += part
                group = name.split(".")[1] if name.startswith("sites.") else name.split(".")[0]
                by_block[group] += part
            grad_norm = math.sqrt(squares)
            if not math.isfinite(grad_norm) or grad_norm <= 0:
                raise FloatingPointError(f"Zero/nonfinite adapter gradient at step {step}")
            optimizer.step()
            torch.cuda.synchronize()
            counts[sample_index] += 1
            record = {"step": step, "sample_index": sample_index,
                "sample_id": rows[sample_index]["global_clip_id"],
                "loss": float(loss.detach()), "lr": optimizer.param_groups[0]["lr"],
                "adapter_gradient_norm": grad_norm,
                "per_block_gradient_norm": {key: math.sqrt(value) for key, value in by_block.items()},
                "timestep_index": timestep_index, "sigma": float(sigma),
                "step_runtime_seconds": round(time.monotonic() - step_started, 3),
                "elapsed_seconds": round(time.monotonic() - started, 3),
                "gpu_allocated_mib": round(torch.cuda.memory_allocated() / 2**20, 1),
                "gpu_reserved_mib": round(torch.cuda.memory_reserved() / 2**20, 1),
                "gpu_peak_allocated_mib": round(torch.cuda.max_memory_allocated() / 2**20, 1)}
            log.write(json.dumps(record) + "\n")
            log.flush()
            if step % 100 == 0 or step in CHECKPOINTS:
                print("M6D_STEP " + json.dumps({"step": step, "loss": record["loss"],
                    "grad_norm": grad_norm, "elapsed_seconds": record["elapsed_seconds"]}), flush=True)
            if step in CHECKPOINTS:
                _save_count = sum(counts.values())
                if _save_count != step:
                    raise RuntimeError("Training sample count changed")
                save_checkpoint(adapter, optimizer, step, record, noise_generator)
            del target, control, actions, noise, noisy, velocity, prediction, loss, action_tokens
    if sum(counts.values()) != STEPS or min(counts.values()) < 15 or max(counts.values()) > 16:
        raise RuntimeError("128-sample training order is unbalanced")
    if any(p.grad is not None or p._version != base_versions[name]
           for name, p in transformer.named_parameters()):
        raise RuntimeError("A pretrained VACE parameter changed or acquired gradients")
    summary = {"status": "complete", "optimizer_steps": STEPS,
        "official_checkpoint_step": 2000, "adapter_trainable_parameters": trainable,
        "pretrained_vace_unchanged": True, "vae_text_not_loaded_or_trained": True,
        "sample_counts": {str(i): counts[i] for i in range(128)},
        "runtime_seconds": round(time.monotonic() - started, 3),
        "peak_gpu_allocated_mib": round(torch.cuda.max_memory_allocated() / 2**20, 1),
        "peak_gpu_reserved_mib": round(torch.cuda.max_memory_reserved() / 2**20, 1),
        "checkpoints": list(CHECKPOINTS), "log_path": str(log_path),
        "official_checkpoint": str(OUT / "checkpoints/step_2000/adapter.pt")}
    save_json(train_dir / "summary.json", summary)
    return summary


if __name__ == "__main__":
    print(json.dumps(run(), indent=2), flush=True)
