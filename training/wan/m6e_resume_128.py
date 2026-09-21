"""Exact M6D step-2000 to step-6400 continuation; audit is the default mode.

No optimizer step runs unless --execute is explicitly supplied.
"""

from __future__ import annotations

import argparse
import collections
import json
import math
import time
from pathlib import Path

import torch
import torch.nn.functional as F

from training.wan.m6d_select_128 import OUT, SEED, digest
from training.wan.m6d_train_128 import LR, PROMPT, load_cache, save_checkpoint
from training.wan.vace_action_token_prototype import ActionTokenAdapter, BLOCKS, MODEL, temporal_mask
from training.wan.vace_action_token_smoke import save_json

START_STEP = 2000
END_STEP = 6400
FUTURE_CHECKPOINTS = (3200, 4800, 6400)
CHECKPOINT = OUT / "checkpoints/step_2000"


def fixed_streams() -> tuple[list[int], list[int]]:
    order_generator = torch.Generator().manual_seed(SEED)
    order = [sample for _ in range(math.ceil(END_STEP / 128))
             for sample in torch.randperm(128, generator=order_generator).tolist()][:END_STEP]
    timestep_indices = torch.randint(
        0, 1000, (END_STEP,), generator=torch.Generator().manual_seed(SEED + 1)).tolist()
    return order, timestep_indices


def audit() -> dict:
    """Read-only proof that all states needed for an exact resume are available."""
    from diffusers import UniPCMultistepScheduler

    subset_path = OUT / "subset_manifest.json"
    subset = json.loads(subset_path.read_text())
    protocol = json.loads((OUT / "train/protocol.json").read_text())
    summary = json.loads((OUT / "train/summary.json").read_text())
    rows = [json.loads(line) for line in (OUT / "train/steps.jsonl").read_text().splitlines()]
    if (subset["clip_count"] != 128 or subset["episode_count"] != 41
            or protocol["subset_manifest_sha256"] != digest(subset_path)
            or protocol["seed"] != SEED or protocol["learning_rate"] != LR
            or protocol["optimizer_steps"] != START_STEP
            or protocol["prompt"] != PROMPT
            or summary["optimizer_steps"] != START_STEP
            or summary["official_checkpoint_step"] != START_STEP
            or not summary["pretrained_vace_unchanged"]
            or len(rows) != START_STEP
            or [row["step"] for row in rows] != list(range(1, START_STEP + 1))):
        raise RuntimeError("M6D source subset, protocol, summary, or step log changed")
    adapter_path = CHECKPOINT / "adapter.pt"
    sidecar = json.loads((CHECKPOINT / "adapter.json").read_text())
    if sidecar["step"] != START_STEP or sidecar["sha256"] != digest(adapter_path):
        raise RuntimeError("Step-2000 adapter checksum or step changed")
    weights = torch.load(adapter_path, map_location="cpu", weights_only=True)
    adapter = ActionTokenAdapter()
    adapter.load_state_dict(weights, strict=True)
    if sum(p.numel() for p in adapter.parameters()) != 3_684_864:
        raise RuntimeError("Action adapter parameter count changed")
    state_path = CHECKPOINT / "training_state.pt"
    state = torch.load(state_path, map_location="cpu", weights_only=True)
    optimizer_state = state["optimizer"]
    if (state["step"] != START_STEP or len(optimizer_state["state"]) != 34
            or optimizer_state["param_groups"][0]["lr"] != LR
            or {int(part["step"]) for part in optimizer_state["state"].values()} != {START_STEP}
            or not isinstance(state["torch_rng"], torch.Tensor)
            or not isinstance(state["cuda_rng"], list) or len(state["cuda_rng"]) != 1
            or not isinstance(state["noise_generator_state"], torch.Tensor)):
        raise RuntimeError("Step-2000 optimizer, current step, or RNG state is incomplete")
    order, timesteps = fixed_streams()
    if ([row["sample_index"] for row in rows] != order[:START_STEP]
            or [row["timestep_index"] for row in rows] != timesteps[:START_STEP]
            or any(not math.isfinite(row["loss"]) or not math.isfinite(row["adapter_gradient_norm"])
                   for row in rows)):
        raise RuntimeError("Seeded sample/timestep streams do not reproduce all 2000 logged steps")
    scheduler = UniPCMultistepScheduler.from_pretrained(
        str(MODEL), subfolder="scheduler", local_files_only=True)
    scheduler.set_timesteps(1000)
    if (scheduler.config.prediction_type != "flow_prediction"
            or scheduler.config.flow_shift != 3.0
            or any(abs(float(scheduler.sigmas[row["timestep_index"]]) - row["sigma"]) > 1e-7
                   for row in rows)):
        raise RuntimeError("Fixed 1000-timestep scheduler cannot reproduce logged training sigmas")
    cache_index = json.loads((OUT / "cache/index.json").read_text())
    cache_summary = json.loads((OUT / "cache/summary.json").read_text())
    if (cache_index["subset_manifest_sha256"] != digest(subset_path)
            or cache_index["sample_ids"] != subset["clip_ids"]
            or cache_summary["status"] != "complete"
            or not cache_summary["native_vace_mask_unchanged"]):
        raise RuntimeError("128-sample cached data or native VACE mask provenance changed")
    return {"ready_for_exact_resume": True,
        "resume_from_step": START_STEP, "target_step": END_STEP,
        "additional_optimizer_steps": END_STEP - START_STEP,
        "future_checkpoint_steps": list(FUTURE_CHECKPOINTS),
        "adapter_path": str(adapter_path), "adapter_sha256": digest(adapter_path),
        "adapter_parameters": 3_684_864,
        "optimizer_state_path": str(state_path), "optimizer_state_sha256": digest(state_path),
        "optimizer_parameter_states": 34, "optimizer_state_step": START_STEP,
        "learning_rate": LR, "optimizer": "AdamW",
        "scheduler": "fixed 1000-timestep UniPC flow schedule; no mutable per-step scheduler state",
        "logged_scheduler_sigmas_match": True,
        "cpu_cuda_and_noise_rng_state_available": True,
        "sample_order_reconstructable_from_seed_and_verified_against_all_logged_steps": True,
        "timestep_order_reconstructable_from_seed_and_verified_against_all_logged_steps": True,
        "next_sample_index": order[START_STEP], "next_timestep_index": timesteps[START_STEP],
        "subset_manifest_sha256": digest(subset_path), "cache_index_sha256": digest(OUT / "cache/index.json"),
        "no_optimizer_step_executed": True}


def execute() -> dict:
    """Future authorized continuation; never called by M6E-0 preparation."""
    from diffusers import UniPCMultistepScheduler, WanVACETransformer3DModel

    proof = audit()
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("BF16 CUDA is required")
    continuation_log = OUT / "train/continuation_2001_6400.jsonl"
    if continuation_log.exists() and continuation_log.stat().st_size:
        raise FileExistsError("Continuation log exists; refusing to silently restart")
    prompt_cpu, samples, rows = load_cache()
    weights = torch.load(CHECKPOINT / "adapter.pt", map_location="cpu", weights_only=True)
    state = torch.load(CHECKPOINT / "training_state.pt", map_location="cpu", weights_only=True)
    adapter = ActionTokenAdapter().to(device="cuda", dtype=torch.float32).train()
    adapter.load_state_dict(weights, strict=True)
    if sum(p.numel() for p in adapter.parameters() if p.requires_grad) != 3_684_864:
        raise RuntimeError("Wrong trainable parameter count")
    transformer = WanVACETransformer3DModel.from_pretrained(
        str(MODEL), subfolder="transformer", torch_dtype=torch.bfloat16, local_files_only=True
    ).to("cuda").eval().requires_grad_(False)
    transformer.enable_gradient_checkpointing()
    if any(p.requires_grad or p.grad is not None for p in transformer.parameters()):
        raise RuntimeError("A pretrained VACE parameter is trainable")
    base_versions = {name: p._version for name, p in transformer.named_parameters()}
    optimizer = torch.optim.AdamW(adapter.parameters(), lr=LR)
    optimizer.load_state_dict(state["optimizer"])
    if (optimizer.param_groups[0]["lr"] != LR
            or {id(p) for group in optimizer.param_groups for p in group["params"]}
               != {id(p) for p in adapter.parameters()}):
        raise RuntimeError("Restored optimizer changed architecture or LR")
    scheduler = UniPCMultistepScheduler.from_pretrained(
        str(MODEL), subfolder="scheduler", local_files_only=True)
    scheduler.set_timesteps(1000)
    if scheduler.config.prediction_type != "flow_prediction" or scheduler.config.flow_shift != 3.0:
        raise RuntimeError("Fixed M6D training scheduler changed")
    order, timestep_indices = fixed_streams()
    noise_generator = torch.Generator(device="cuda")
    noise_generator.set_state(state["noise_generator_state"])
    # Restore only after loading modules so setup cannot consume the resumed RNG stream.
    torch.set_rng_state(state["torch_rng"])
    torch.cuda.set_rng_state_all(state["cuda_rng"])
    timesteps = scheduler.timesteps.to("cuda")
    sigmas = scheduler.sigmas[:1000].to("cuda")
    prompt = prompt_cpu.to("cuda")
    mask = temporal_mask(torch.device("cuda"))
    control_scale = torch.ones(len(transformer.config.vace_layers), device="cuda", dtype=torch.bfloat16)
    protocol = {**proof, "no_optimizer_step_executed": False,
        "batch_size": 1, "gradient_accumulation": 1, "backbone_dtype": "bfloat16",
        "future_loss_latent_positions": [1, 2, 3, 4], "pretrained_weights_frozen": True}
    save_json(OUT / "train/continuation_protocol_6400.json", protocol)
    started = time.monotonic()
    torch.cuda.reset_peak_memory_stats()
    counts = collections.Counter(order[:START_STEP])
    with continuation_log.open("w", encoding="utf-8") as log:
        for step in range(START_STEP + 1, END_STEP + 1):
            step_started = time.monotonic()
            sample_index = order[step - 1]
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
                if gradient is None or not bool(torch.isfinite(gradient).all()):
                    raise FloatingPointError(f"Missing/nonfinite adapter gradient at step {step}: {name}")
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
                "continuation_elapsed_seconds": round(time.monotonic() - started, 3),
                "gpu_allocated_mib": round(torch.cuda.memory_allocated() / 2**20, 1),
                "gpu_reserved_mib": round(torch.cuda.memory_reserved() / 2**20, 1),
                "gpu_peak_allocated_mib": round(torch.cuda.max_memory_allocated() / 2**20, 1)}
            log.write(json.dumps(record) + "\n")
            log.flush()
            if step % 100 == 0 or step in FUTURE_CHECKPOINTS:
                print("M6E_STEP " + json.dumps({"step": step, "loss": record["loss"],
                    "grad_norm": grad_norm, "elapsed_seconds": record["continuation_elapsed_seconds"]}), flush=True)
            if step in FUTURE_CHECKPOINTS:
                save_checkpoint(adapter, optimizer, step, record, noise_generator)
            del target, control, actions, noise, noisy, velocity, prediction, loss, action_tokens
    if sum(counts.values()) != END_STEP or counts != {i: 50 for i in range(128)}:
        raise RuntimeError("6400-step sample order is unbalanced")
    if any(p.grad is not None or p._version != base_versions[name]
           for name, p in transformer.named_parameters()):
        raise RuntimeError("A pretrained VACE parameter changed or acquired gradients")
    summary = {"status": "complete", "resumed_from_step": START_STEP,
        "optimizer_steps": END_STEP, "new_steps": END_STEP - START_STEP,
        "checkpoint_steps": list(FUTURE_CHECKPOINTS),
        "pretrained_vace_unchanged": True,
        "runtime_seconds": round(time.monotonic() - started, 3),
        "peak_gpu_allocated_mib": round(torch.cuda.max_memory_allocated() / 2**20, 1),
        "peak_gpu_reserved_mib": round(torch.cuda.max_memory_reserved() / 2**20, 1),
        "log_path": str(continuation_log)}
    save_json(OUT / "train/continuation_summary_6400.json", summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true", help="Actually train steps 2001 through 6400")
    args = parser.parse_args()
    print(json.dumps(execute() if args.execute else audit(), indent=2), flush=True)


if __name__ == "__main__":
    main()
