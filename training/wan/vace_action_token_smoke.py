"""M6B-4: exactly 32 optimizer steps for the frozen-VACE action-token adapter."""

from __future__ import annotations

import collections
import hashlib
import json
import math
import random
import time
from pathlib import Path

import torch
import torch.nn.functional as F

from training.wan.vace_action_adapter import ACTION_TO_ID
from training.wan.vace_action_token_prototype import ActionTokenAdapter, BLOCKS, MODEL, temporal_mask
from training.wan.vace_action_train import MANIFEST, OUTPUT as OLD_OUTPUT, _records, _sha256

ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / "data/experiments/m6b4_action_token_smoke"
STEPS, SEED, LR = 32, 42, 2e-4


def save_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def load_cache() -> tuple[torch.Tensor, list[dict], list[dict]]:
    rows = _records()
    directory = OLD_OUTPUT / "train/cache"
    index = json.loads((directory / "index.json").read_text())
    if (index["manifest_sha256"] != _sha256(MANIFEST)
            or index["sample_ids"] != [row["global_clip_id"] for row in rows]
            or index["prompt"] != "A first-person gameplay view in a retro 3D stone maze."
            or index["native_vace_mask_unchanged"] is not True):
        raise RuntimeError("Existing cache index or fixed prompt changed")
    prompt = torch.load(directory / "prompt.pt", map_location="cpu", weights_only=True)
    if prompt.shape != (1, 512, 4096):
        raise RuntimeError("Unexpected cached generic-prompt shape")
    samples = []
    for i, row in enumerate(rows):
        sample = torch.load(directory / f"sample_{i:02d}.pt", map_location="cpu", weights_only=True)
        expected = [[ACTION_TO_ID[action] for action in row["raw_actions"]]]
        if (sample["target"].shape != (1, 16, 5, 32, 56)
                or sample["control"].shape != (1, 96, 5, 32, 56)
                or sample["actions"].tolist() != expected
                or not bool(torch.isfinite(sample["target"]).all())
                or not bool(torch.isfinite(sample["control"]).all())
                or not bool(torch.all(sample["control"][:, 32:] == 1))):
            raise RuntimeError(f"Invalid cached sample/actions: {row['global_clip_id']}")
        samples.append(sample)
    return prompt, samples, rows


def run() -> dict:
    from diffusers import UniPCMultistepScheduler, WanVACETransformer3DModel

    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("BF16 CUDA is required")
    prototype = json.loads((ROOT / "data/experiments/m6b3_action_token_cross_attention/prototype.json").read_text())
    if not prototype["zero_init_exact_equal_to_unhooked_vace"] or prototype["adapter_parameters"] != 3_684_864:
        raise RuntimeError("M6B-3 zero-init audit did not pass")
    if OUTPUT.exists() and any(OUTPUT.iterdir()):
        raise FileExistsError("M6B-4 smoke output already exists; refusing to overwrite or train more")
    OUTPUT.mkdir(parents=True)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    random.seed(SEED)
    started = time.monotonic()
    prompt_cpu, samples, rows = load_cache()
    prompt = prompt_cpu.to("cuda")
    adapter = ActionTokenAdapter().to("cuda", dtype=torch.float32).train()
    trainable = sum(p.numel() for p in adapter.parameters() if p.requires_grad)
    if trainable != 3_684_864 or any(not p.requires_grad for p in adapter.parameters()):
        raise RuntimeError(f"Wrong trainable adapter parameters: {trainable}")
    if any(bool(torch.count_nonzero(site.out_proj.weight)) or bool(torch.count_nonzero(site.out_proj.bias))
           for site in adapter.sites.values()):
        raise RuntimeError("Action output projections were not zero initialized")
    transformer = WanVACETransformer3DModel.from_pretrained(
        str(MODEL), subfolder="transformer", torch_dtype=torch.bfloat16, local_files_only=True
    ).to("cuda").eval().requires_grad_(False)
    if len(transformer.blocks) != 30 or list(transformer.config.vace_layers) != list(range(0, 30, 2)):
        raise RuntimeError("VACE block layout changed")
    if any(p.requires_grad or p.grad is not None for p in transformer.parameters()):
        raise RuntimeError("A pretrained VACE parameter is trainable")
    transformer.enable_gradient_checkpointing()
    base_versions = {name: parameter._version for name, parameter in transformer.named_parameters()}
    mask = temporal_mask(torch.device("cuda"))
    control_scale = torch.ones(len(transformer.config.vace_layers), device="cuda", dtype=torch.bfloat16)
    optimizer = torch.optim.AdamW(adapter.parameters(), lr=LR)
    if {id(p) for group in optimizer.param_groups for p in group["params"]} != {id(p) for p in adapter.parameters()}:
        raise RuntimeError("Optimizer includes a pretrained parameter")
    scheduler = UniPCMultistepScheduler.from_pretrained(str(MODEL), subfolder="scheduler", local_files_only=True)
    scheduler.set_timesteps(1000)
    if scheduler.config.prediction_type != "flow_prediction" or scheduler.config.flow_shift != 3.0:
        raise RuntimeError("Pinned flow scheduler changed")
    timesteps = scheduler.timesteps.to("cuda")
    sigmas = scheduler.sigmas[:1000].to("cuda")
    order_generator = torch.Generator().manual_seed(SEED)
    order = torch.randperm(16, generator=order_generator).tolist() + torch.randperm(16, generator=order_generator).tolist()
    indices = torch.randint(0, 1000, (STEPS,), generator=torch.Generator().manual_seed(SEED + 1)).tolist()
    noise_generator = torch.Generator(device="cuda").manual_seed(SEED + 2)

    def denoise(noisy: torch.Tensor, timestep: torch.Tensor, control: torch.Tensor) -> torch.Tensor:
        return transformer(hidden_states=noisy, timestep=timestep, encoder_hidden_states=prompt,
                           control_hidden_states=control, control_hidden_states_scale=control_scale,
                           return_dict=False)[0]

    # Repeat the M6B-3 equivalence check on the exact model/adapter used for training.
    sample0 = samples[0]
    fixed_target = sample0["target"].to("cuda")
    fixed_control = sample0["control"].to("cuda")
    fixed_noise = torch.randn(fixed_target.shape, generator=torch.Generator(device="cuda").manual_seed(SEED),
                              device="cuda", dtype=torch.float32)
    fixed_sigma = sigmas[774].float()
    fixed_noisy = ((1 - fixed_sigma) * fixed_target.float() + fixed_sigma * fixed_noise).to(torch.bfloat16)
    fixed_timestep = timesteps[774].expand(1)
    with torch.inference_mode():
        baseline = denoise(fixed_noisy, fixed_timestep, fixed_control)
        zero_outputs = {}
        for name, ids in {"original": sample0["actions"],
                          "all_noop": torch.full((1, 16), 3, dtype=torch.long)}.items():
            action_tokens = adapter.tokens(ids.to("cuda"))
            handles = []
            try:
                for block in BLOCKS:
                    def hook(_module, _inputs, hidden, *, site=block):
                        with torch.autocast("cuda", dtype=torch.bfloat16):
                            return adapter.sites[str(site)](hidden, action_tokens, mask)
                    handles.append(transformer.blocks[block].register_forward_hook(hook))
                zero_outputs[name] = denoise(fixed_noisy, fixed_timestep, fixed_control)
            finally:
                for handle in handles:
                    handle.remove()
        if not (torch.equal(baseline, zero_outputs["original"])
                and torch.equal(baseline, zero_outputs["all_noop"])):
            raise RuntimeError("Zero-initialized adapter changed VACE output")
    del baseline, zero_outputs, fixed_noise, fixed_target, fixed_control, fixed_noisy
    torch.cuda.empty_cache()

    save_json(OUTPUT / "protocol.json", {
        "model": str(MODEL), "manifest_sha256": _sha256(MANIFEST),
        "cache_index_sha256": _sha256(OLD_OUTPUT / "train/cache/index.json"),
        "prompt": "A first-person gameplay view in a retro 3D stone maze.",
        "steps": STEPS, "seed": SEED, "optimizer": "AdamW", "lr": LR,
        "batch_size": 1, "gradient_accumulation": 1, "backbone_dtype": "bfloat16",
        "trainable_parameters": trainable, "blocks": list(BLOCKS),
        "native_o0_and_latent_mask_unchanged": True,
        "future_only_loss_latent_positions": [1, 2, 3, 4],
        "all_16_cached_samples_and_actions_valid": True,
        "zero_init_exact_baseline_equality_rechecked": True,
    })
    torch.cuda.reset_peak_memory_stats()
    counts = collections.Counter()
    all_grad_nonzero = True
    max_loss = -math.inf
    min_loss = math.inf
    log_path = OUTPUT / "steps.jsonl"
    with log_path.open("w", encoding="utf-8") as log:
        for step, sample_index in enumerate(order, start=1):
            step_started = time.monotonic()
            sample = samples[sample_index]
            target = sample["target"].to("cuda")
            control = sample["control"].to("cuda")
            actions = sample["actions"].to("cuda")
            timestep_index = indices[step - 1]
            sigma = sigmas[timestep_index].float()
            timestep = timesteps[timestep_index].expand(1)
            noise = torch.randn(target.shape, device="cuda", dtype=torch.bfloat16, generator=noise_generator)
            noisy = ((1 - sigma) * target.float() + sigma * noise.float()).to(torch.bfloat16)
            velocity = noise.float() - target.float()
            optimizer.zero_grad(set_to_none=True)
            action_tokens = adapter.tokens(actions)
            handles = []
            hook_calls = collections.Counter()
            try:
                for block in BLOCKS:
                    def hook(_module, _inputs, hidden, *, site=block):
                        hook_calls[site] += 1
                        with torch.autocast("cuda", dtype=torch.bfloat16):
                            return adapter.sites[str(site)](hidden, action_tokens, mask)
                    handles.append(transformer.blocks[block].register_forward_hook(hook))
                prediction = denoise(noisy, timestep, control)
                if prediction.shape != target.shape:
                    raise RuntimeError("Unexpected denoiser output shape")
                loss = F.mse_loss(prediction.float()[:, :, 1:], velocity[:, :, 1:])
                if not bool(torch.isfinite(loss)):
                    raise FloatingPointError(f"Nonfinite loss at step {step}")
                loss.backward()
            finally:
                for handle in handles:
                    handle.remove()
            if any(hook_calls[block] == 0 for block in BLOCKS):
                raise RuntimeError(f"Action hook skipped at step {step}: {hook_calls}")
            squared_norm = 0.0
            nonzero_count = 0
            for name, parameter in adapter.named_parameters():
                grad = parameter.grad
                if grad is None:
                    # Zero output projections gate Q/K/V and embeddings until later steps.
                    if step == 1 and ("out_proj" not in name):
                        continue
                    raise FloatingPointError(f"Missing gradient at step {step}: {name}")
                if not bool(torch.isfinite(grad).all()):
                    raise FloatingPointError(f"Nonfinite gradient at step {step}: {name}")
                squared_norm += float(torch.sum(grad.float().square()))
                nonzero_count += int(bool(torch.count_nonzero(grad)))
            grad_norm = math.sqrt(squared_norm)
            if not math.isfinite(grad_norm) or grad_norm <= 0 or nonzero_count == 0:
                raise FloatingPointError(f"Zero/nonfinite adapter gradient at step {step}")
            optimizer.step()
            torch.cuda.synchronize()
            counts[sample_index] += 1
            loss_value = float(loss.detach())
            min_loss, max_loss = min(min_loss, loss_value), max(max_loss, loss_value)
            record = {"step": step, "sample_index": sample_index,
                      "sample_id": rows[sample_index]["global_clip_id"], "loss": loss_value,
                      "lr": optimizer.param_groups[0]["lr"], "adapter_gradient_norm": grad_norm,
                      "nonzero_gradient_tensors": nonzero_count,
                      "hook_calls": {str(block): hook_calls[block] for block in BLOCKS},
                      "timestep_index": timestep_index, "sigma": float(sigma),
                      "step_runtime_seconds": round(time.monotonic() - step_started, 3),
                      "elapsed_seconds": round(time.monotonic() - started, 3),
                      "gpu_allocated_mib": round(torch.cuda.memory_allocated() / 2**20, 1),
                      "gpu_reserved_mib": round(torch.cuda.memory_reserved() / 2**20, 1),
                      "gpu_peak_allocated_mib": round(torch.cuda.max_memory_allocated() / 2**20, 1)}
            log.write(json.dumps(record) + "\n")
            log.flush()
            print("M6B4_STEP " + json.dumps(record), flush=True)
            del target, control, actions, noise, noisy, velocity, prediction, loss, action_tokens
    if counts != {i: 2 for i in range(16)}:
        raise RuntimeError(f"Unbalanced 32-step schedule: {counts}")
    if any(p.grad is not None or p._version != base_versions[name]
           for name, p in transformer.named_parameters()):
        raise RuntimeError("Pretrained VACE weight changed or acquired a gradient")
    checkpoint = OUTPUT / "step_0032.pt"
    temporary = checkpoint.with_suffix(".partial.pt")
    torch.save({"step": STEPS, "adapter": {name: value.detach().cpu() for name, value in adapter.state_dict().items()},
                "optimizer": optimizer.state_dict(), "torch_rng": torch.get_rng_state(),
                "cuda_rng": torch.cuda.get_rng_state_all(), "noise_generator_state": noise_generator.get_state(),
                "order_generator_state": order_generator.get_state()}, temporary)
    temporary.replace(checkpoint)
    gate = {"passed": True, "steps": STEPS, "finite_losses": True, "finite_gradients": True,
            "nonzero_adapter_gradients": all_grad_nonzero, "no_nan_inf": True, "no_oom": True,
            "pretrained_vace_weights_unchanged": True, "pretrained_vae_text_frozen": True,
            "all_16_samples_and_actions_loaded": True, "zero_init_exact_match": True,
            "adapter_trainable_parameters": trainable,
            "sample_counts": {str(i): counts[i] for i in range(16)},
            "minimum_loss": min_loss, "maximum_loss": max_loss,
            "runtime_seconds": round(time.monotonic() - started, 3),
            "peak_gpu_allocated_mib": round(torch.cuda.max_memory_allocated() / 2**20, 1),
            "peak_gpu_reserved_mib": round(torch.cuda.max_memory_reserved() / 2**20, 1),
            "checkpoint": str(checkpoint), "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
            "step_log": str(log_path)}
    save_json(OUTPUT / "smoke_gate.json", gate)
    return gate


if __name__ == "__main__":
    print(json.dumps(run(), indent=2), flush=True)
