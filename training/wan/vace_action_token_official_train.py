"""M6B-4 official 800-step run, exactly resumed from the validated step-32 gate."""

from __future__ import annotations

import collections
import json
import math
import time
from pathlib import Path

import torch
import torch.nn.functional as F

from training.wan.vace_action_token_prototype import ActionTokenAdapter, BLOCKS, MODEL, temporal_mask
from training.wan.vace_action_token_smoke import OUTPUT as SMOKE, LR, ROOT, SEED, load_cache, save_json
from training.wan.vace_action_train import MANIFEST, _sha256

OUTPUT = ROOT / "data/experiments/m6b4_action_token_800"
CHECKPOINTS = (100, 200, 400, 800)


def _save_adapter(adapter: ActionTokenAdapter, step: int, record: dict) -> None:
    path = OUTPUT / "checkpoints" / f"step_{step:04d}" / "adapter.pt"
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".partial.pt")
    state = {name: tensor.detach().cpu().contiguous() for name, tensor in adapter.state_dict().items()}
    torch.save(state, temporary)
    temporary.replace(path)
    save_json(path.with_suffix(".json"), {"step": step, "adapter_parameters": 3_684_864,
        "sha256": _sha256(path), "keys": sorted(state), "last_step": record})


def run() -> dict:
    from diffusers import UniPCMultistepScheduler, WanVACETransformer3DModel

    if OUTPUT.exists() and any(OUTPUT.iterdir()):
        raise FileExistsError("Official output already exists; refusing to overwrite or silently resume")
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("BF16 CUDA is required")
    checkpoint_path = SMOKE / "step_0032.pt"
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    smoke_gate = json.loads((SMOKE / "smoke_gate.json").read_text())
    smoke_rows = [json.loads(line) for line in (SMOKE / "steps.jsonl").read_text().splitlines()]
    if (checkpoint["step"] != 32 or len(smoke_rows) != 32 or not smoke_gate["passed"]
            or smoke_gate["checkpoint_sha256"] != _sha256(checkpoint_path)
            or checkpoint["optimizer"]["param_groups"][0]["lr"] != LR
            or len(checkpoint["optimizer"]["state"]) != 34):
        raise RuntimeError("Step-32 checkpoint/log failed exact-resume audit")
    order_generator = torch.Generator().manual_seed(SEED)
    previous_order = [i for _ in range(2) for i in torch.randperm(16, generator=order_generator).tolist()]
    if (previous_order != [row["sample_index"] for row in smoke_rows]
            or not torch.equal(order_generator.get_state(), checkpoint["order_generator_state"])):
        raise RuntimeError("Step-32 data-order state cannot be reconstructed exactly")
    order_generator.set_state(checkpoint["order_generator_state"])
    remaining_order = [i for _ in range(48) for i in torch.randperm(16, generator=order_generator).tolist()]
    full_order = previous_order + remaining_order
    if len(full_order) != 800 or collections.Counter(full_order) != {i: 50 for i in range(16)}:
        raise RuntimeError("800-step data order is not balanced")
    indices = torch.randint(0, 1000, (800,), generator=torch.Generator().manual_seed(SEED + 1)).tolist()
    if indices[:32] != [row["timestep_index"] for row in smoke_rows]:
        raise RuntimeError("Step-32 timestep RNG cannot be reconstructed exactly")

    prompt_cpu, samples, rows = load_cache()
    prompt = prompt_cpu.to("cuda")
    transformer = WanVACETransformer3DModel.from_pretrained(
        str(MODEL), subfolder="transformer", torch_dtype=torch.bfloat16, local_files_only=True
    ).to("cuda").eval().requires_grad_(False)
    if any(p.requires_grad for p in transformer.parameters()):
        raise RuntimeError("A pretrained VACE parameter is trainable")
    transformer.enable_gradient_checkpointing()
    base_versions = {name: p._version for name, p in transformer.named_parameters()}
    adapter = ActionTokenAdapter().to("cuda", dtype=torch.float32).train()
    adapter.load_state_dict(checkpoint["adapter"], strict=True)
    if sum(p.numel() for p in adapter.parameters() if p.requires_grad) != 3_684_864:
        raise RuntimeError("Adapter trainable parameter count changed")
    optimizer = torch.optim.AdamW(adapter.parameters(), lr=LR)
    optimizer.load_state_dict(checkpoint["optimizer"])
    if (optimizer.param_groups[0]["lr"] != LR
            or {id(p) for group in optimizer.param_groups for p in group["params"]}
               != {id(p) for p in adapter.parameters()}):
        raise RuntimeError("Optimizer state or parameter membership changed")
    scheduler = UniPCMultistepScheduler.from_pretrained(str(MODEL), subfolder="scheduler", local_files_only=True)
    scheduler.set_timesteps(1000)
    if scheduler.config.prediction_type != "flow_prediction" or scheduler.config.flow_shift != 3.0:
        raise RuntimeError("Fixed training scheduler changed")
    timesteps = scheduler.timesteps.to("cuda")
    sigmas = scheduler.sigmas[:1000].to("cuda")
    noise_generator = torch.Generator(device="cuda")
    noise_generator.set_state(checkpoint["noise_generator_state"])
    torch.set_rng_state(checkpoint["torch_rng"])
    torch.cuda.set_rng_state_all(checkpoint["cuda_rng"])
    mask = temporal_mask(torch.device("cuda"))
    control_scale = torch.ones(len(transformer.config.vace_layers), device="cuda", dtype=torch.bfloat16)
    OUTPUT.mkdir(parents=True)
    save_json(OUTPUT / "resume_audit.json", {
        "decision": "exact_resume_from_step_32", "step32_checkpoint": str(checkpoint_path),
        "step32_checkpoint_sha256": _sha256(checkpoint_path),
        "adapter_weights_loaded": True, "optimizer_state_loaded": True,
        "scheduler": "fixed 1000-timestep UniPC flow schedule; no per-step scheduler state",
        "scheduler_reconstructed": True, "cpu_cuda_and_noise_rng_restored": True,
        "first_32_sample_indices_match_log": True, "saved_order_generator_state_matches": True,
        "first_32_timestep_indices_match_log": True,
        "same_manifest_sha256": _sha256(MANIFEST), "same_16_sample_cache": True,
        "steps_remaining": 768, "expected_total_steps": 800,
    })
    torch.cuda.reset_peak_memory_stats()
    started = time.monotonic()
    counts = collections.Counter(previous_order)
    log_path = OUTPUT / "steps.jsonl"
    with log_path.open("w", encoding="utf-8") as log:
        for row in smoke_rows:
            log.write(json.dumps({**row, "phase": "resumed_smoke_steps_1_to_32"}) + "\n")
        log.flush()
        for step in range(33, 801):
            step_start = time.monotonic()
            sample_index = full_order[step - 1]
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
                prediction = transformer(
                    hidden_states=noisy, timestep=timestep, encoder_hidden_states=prompt,
                    control_hidden_states=control, control_hidden_states_scale=control_scale,
                    return_dict=False,
                )[0]
                if prediction.shape != target.shape:
                    raise RuntimeError(f"Denoiser output shape changed at step {step}")
                loss = F.mse_loss(prediction.float()[:, :, 1:], velocity[:, :, 1:])
                if not bool(torch.isfinite(loss)):
                    raise FloatingPointError(f"Nonfinite loss at step {step}")
                loss.backward()
            finally:
                for handle in handles:
                    handle.remove()
            if any(hook_calls[block] != 2 for block in BLOCKS):
                raise RuntimeError(f"Missing forward or checkpoint-recompute hook at step {step}")
            squares = 0.0
            grad_groups = collections.defaultdict(float)
            for name, parameter in adapter.named_parameters():
                grad = parameter.grad
                if grad is None or not bool(torch.isfinite(grad).all()):
                    raise FloatingPointError(f"Missing/nonfinite adapter gradient at step {step}: {name}")
                part = float(torch.sum(grad.float().square()))
                squares += part
                group = name.split(".")[1] if name.startswith("sites.") else name.split(".")[0]
                grad_groups[group] += part
            grad_norm = math.sqrt(squares)
            if not math.isfinite(grad_norm) or grad_norm <= 0:
                raise FloatingPointError(f"Zero/nonfinite adapter gradient at step {step}")
            optimizer.step()
            torch.cuda.synchronize()
            counts[sample_index] += 1
            record = {"step": step, "phase": "exact_resume_continuation",
                "sample_index": sample_index, "sample_id": rows[sample_index]["global_clip_id"],
                "loss": float(loss.detach()), "lr": optimizer.param_groups[0]["lr"],
                "adapter_gradient_norm": grad_norm,
                "per_block_gradient_norm": {key: math.sqrt(value) for key, value in grad_groups.items()},
                "hook_calls": {str(block): hook_calls[block] for block in BLOCKS},
                "timestep_index": timestep_index, "sigma": float(sigma),
                "step_runtime_seconds": round(time.monotonic() - step_start, 3),
                "continuation_elapsed_seconds": round(time.monotonic() - started, 3),
                "gpu_allocated_mib": round(torch.cuda.memory_allocated() / 2**20, 1),
                "gpu_reserved_mib": round(torch.cuda.memory_reserved() / 2**20, 1),
                "gpu_peak_allocated_mib": round(torch.cuda.max_memory_allocated() / 2**20, 1)}
            log.write(json.dumps(record) + "\n")
            log.flush()
            if step % 25 == 0 or step in CHECKPOINTS:
                print("M6B4_OFFICIAL " + json.dumps({"step": step, "loss": record["loss"],
                    "gradient_norm": grad_norm, "elapsed_seconds": record["continuation_elapsed_seconds"]}), flush=True)
            if step in CHECKPOINTS:
                _save_adapter(adapter, step, record)
            del target, control, actions, noise, noisy, velocity, prediction, loss, action_tokens
    if counts != {i: 50 for i in range(16)}:
        raise RuntimeError(f"Final sample counts are unbalanced: {counts}")
    if any(p.grad is not None or p._version != base_versions[name]
           for name, p in transformer.named_parameters()):
        raise RuntimeError("A pretrained VACE parameter changed or acquired gradients")
    summary = {"status": "complete", "optimizer_steps": 800, "resume_from_step": 32,
        "continuation_steps": 768, "adapter_trainable_parameters": 3_684_864,
        "pretrained_vace_unchanged": True, "pretrained_vae_text_not_loaded_for_training": True,
        "sample_counts": {str(i): counts[i] for i in range(16)},
        "continuation_runtime_seconds": round(time.monotonic() - started, 3),
        "peak_gpu_allocated_mib": round(torch.cuda.max_memory_allocated() / 2**20, 1),
        "peak_gpu_reserved_mib": round(torch.cuda.max_memory_reserved() / 2**20, 1),
        "checkpoint_steps": list(CHECKPOINTS), "log_path": str(log_path)}
    save_json(OUTPUT / "training_summary.json", summary)
    return summary


if __name__ == "__main__":
    print(json.dumps(run(), indent=2), flush=True)
