"""Controlled 6400-step M6D restart with only the locked action-ranking objective changed."""

from __future__ import annotations

import argparse
import collections
import json
import math
import shutil
import time
from pathlib import Path

import torch
import torch.nn.functional as F

from training.wan.m6d_select_128 import OUT as M6D, ROOT, SEED, digest
from training.wan.m6d_train_128 import LR, PROMPT, load_cache
from training.wan.m6e_resume_128 import fixed_streams
from training.wan.m6f0_action_contrastive_audit import SWAP
from training.wan.vace_action_token_prototype import ActionTokenAdapter, BLOCKS, MODEL, temporal_mask

OUT = ROOT / "data/experiments/m6f2_action_contrastive_128_6400"
STEPS = 6400
CHECKPOINTS = (1600, 3200, 4800, 6400)
MARGIN = 0.01
LAMBDA = 0.25


def save_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def setup() -> dict:
    from diffusers import UniPCMultistepScheduler, WanVACETransformer3DModel

    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("BF16 CUDA is required")
    prompt_cpu, samples, rows = load_cache()
    if len(samples) != 128 or len(rows) != 128:
        raise RuntimeError("The fixed M6D TRAIN subset changed")
    order, timestep_indices = fixed_streams()
    original_logs = [json.loads(line) for file in (
        M6D / "train/steps.jsonl", M6D / "train/continuation_2001_6400.jsonl")
        for line in file.read_text().splitlines()]
    if (len(order) != STEPS or len(timestep_indices) != STEPS or len(original_logs) != STEPS
            or any(row["step"] != i + 1 or row["sample_index"] != order[i]
                   or row["timestep_index"] != timestep_indices[i]
                   for i, row in enumerate(original_logs))):
        raise RuntimeError("Seeded sample/timestep streams differ from the complete M6D/M6E run")
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    adapter = ActionTokenAdapter().to(device="cuda", dtype=torch.float32).train()
    original_initial = M6D / "train/initial_adapter.pt"
    initial = torch.load(original_initial, map_location="cpu", weights_only=True)
    if (sum(p.numel() for p in adapter.parameters() if p.requires_grad) != 3_684_864
            or any(not torch.equal(value.cpu(), initial[name])
                   for name, value in adapter.state_dict().items())
            or any(bool(torch.count_nonzero(site.out_proj.weight))
                   or bool(torch.count_nonzero(site.out_proj.bias))
                   for site in adapter.sites.values())):
        raise RuntimeError("Fresh seed-42 adapter differs from the saved M6D/M6E zero initialization")
    transformer = WanVACETransformer3DModel.from_pretrained(
        str(MODEL), subfolder="transformer", torch_dtype=torch.bfloat16, local_files_only=True
    ).to("cuda").eval().requires_grad_(False)
    transformer.enable_gradient_checkpointing()
    if any(p.requires_grad or p.grad is not None for p in transformer.parameters()):
        raise RuntimeError("A pretrained VACE parameter is trainable")
    base_versions = {name: p._version for name, p in transformer.named_parameters()}
    scheduler = UniPCMultistepScheduler.from_pretrained(
        str(MODEL), subfolder="scheduler", local_files_only=True)
    scheduler.set_timesteps(1000)
    if (scheduler.config.prediction_type != "flow_prediction"
            or scheduler.config.flow_shift != 3.0
            or any(abs(float(scheduler.sigmas[row["timestep_index"]]) - row["sigma"]) > 1e-7
                   for row in original_logs)):
        raise RuntimeError("The fixed M6D flow scheduler changed")
    optimizer = torch.optim.AdamW(adapter.parameters(), lr=LR)
    if {id(p) for group in optimizer.param_groups for p in group["params"]} != {
            id(p) for p in adapter.parameters()}:
        raise RuntimeError("The optimizer includes pretrained weights")
    noise_generator = torch.Generator(device="cuda").manual_seed(SEED + 2)
    correct_ids = [sample["actions"] for sample in samples]
    wrong_ids = []
    for actions in correct_ids:
        wrong = actions.clone()
        for source, destination in SWAP.items():
            wrong[actions == source] = destination
        wrong_ids.append(wrong)
    eligible = [not torch.equal(correct, wrong)
                for correct, wrong in zip(correct_ids, wrong_ids, strict=True)]
    if sum(eligible) != 124 or sum(not value for value in eligible) != 4:
        raise RuntimeError("The fixed subset's swappable-action coverage changed")
    return {"prompt": prompt_cpu.to("cuda"), "samples": samples, "rows": rows,
        "adapter": adapter, "transformer": transformer, "base_versions": base_versions,
        "scheduler": scheduler, "optimizer": optimizer, "noise_generator": noise_generator,
        "order": order, "timestep_indices": timestep_indices,
        "wrong_ids": wrong_ids, "eligible": eligible,
        "mask": temporal_mask(torch.device("cuda")),
        "control_scale": torch.ones(len(transformer.config.vace_layers),
            device="cuda", dtype=torch.bfloat16),
        "initial_adapter_path": original_initial, "initial_adapter_sha256": digest(original_initial)}


def train_step(context: dict, step: int, *, optimizer_update: bool) -> dict:
    adapter, transformer = context["adapter"], context["transformer"]
    sample_index = context["order"][step - 1]
    sample = context["samples"][sample_index]
    target = sample["target"].to("cuda")
    control = sample["control"].to("cuda")
    correct = sample["actions"].to("cuda")
    wrong = context["wrong_ids"][sample_index].to("cuda")
    eligible = context["eligible"][sample_index]
    timestep_index = context["timestep_indices"][step - 1]
    scheduler = context["scheduler"]
    sigma = scheduler.sigmas[timestep_index].to("cuda").float()
    timestep = scheduler.timesteps[timestep_index].to("cuda").expand(1)
    noise = torch.randn(target.shape, device="cuda", dtype=torch.bfloat16,
                        generator=context["noise_generator"])
    noisy = ((1 - sigma) * target.float() + sigma * noise.float()).to(torch.bfloat16)
    velocity = noise.float() - target.float()
    optimizer = context["optimizer"]
    optimizer.zero_grad(set_to_none=True)
    versions_before_pair = tuple(p._version for p in adapter.parameters())

    def forward_loss(actions: torch.Tensor, mode: str,
                     wrong_reference: float | None = None) -> tuple[float, bool]:
        tokens = adapter.tokens(actions)
        hook_calls = collections.Counter()
        handles = []
        try:
            for block in BLOCKS:
                def hook(_module, _inputs, hidden, *, site=block, action_tokens=tokens):
                    hook_calls[site] += 1
                    with torch.autocast("cuda", dtype=torch.bfloat16):
                        return adapter.sites[str(site)](hidden, action_tokens, context["mask"])
                handles.append(transformer.blocks[block].register_forward_hook(hook))
            predicted = transformer(hidden_states=noisy, timestep=timestep,
                encoder_hidden_states=context["prompt"], control_hidden_states=control,
                control_hidden_states_scale=context["control_scale"], return_dict=False)[0]
            if predicted.shape != target.shape:
                raise RuntimeError(f"Denoiser output shape changed at optimizer step {step}")
            loss = F.mse_loss(predicted.float()[:, :, 1:], velocity[:, :, 1:])
            if not bool(torch.isfinite(loss)):
                raise FloatingPointError(f"Nonfinite future-only loss at step {step}")
            value = float(loss.detach())
            active = bool(wrong_reference is not None and MARGIN + value - wrong_reference > 0)
            if mode == "correct":
                (loss * (1 + LAMBDA if active else 1)).backward()
            elif mode == "wrong_gradient":
                (-LAMBDA * loss).backward()
            elif mode != "wrong_value":
                raise ValueError(f"Unknown paired-forward mode: {mode}")
        finally:
            for handle in handles:
                handle.remove()
        required_calls = 1 if mode == "wrong_value" else 2
        if any(hook_calls[block] != required_calls for block in BLOCKS):
            raise RuntimeError(f"Action hook skipped at step {step}: {hook_calls}")
        return value, active

    wrong_value = None
    if eligible:
        with torch.no_grad():
            wrong_value, _ = forward_loss(wrong, "wrong_value")
    correct_value, hinge_active = forward_loss(correct, "correct", wrong_value)
    if eligible and hinge_active:
        wrong_gradient_value, _ = forward_loss(wrong, "wrong_gradient")
        if abs(wrong_gradient_value - wrong_value) > 1e-6:
            raise RuntimeError(f"Wrong-action measurement and gradient forwards differ at step {step}")
    if tuple(p._version for p in adapter.parameters()) != versions_before_pair:
        raise RuntimeError(f"Adapter weights changed between paired forwards at step {step}")
    rank = max(0.0, MARGIN + correct_value - wrong_value) if eligible else 0.0
    total = correct_value + LAMBDA * rank
    squares = 0.0
    per_block = collections.defaultdict(float)
    for name, parameter in adapter.named_parameters():
        gradient = parameter.grad
        if gradient is None:
            if step == 1 and "out_proj" not in name:
                continue
            raise FloatingPointError(f"Missing adapter gradient at step {step}: {name}")
        if not bool(torch.isfinite(gradient).all()):
            raise FloatingPointError(f"Nonfinite adapter gradient at step {step}: {name}")
        value = float(gradient.float().square().sum())
        squares += value
        key = name.split(".")[1] if name.startswith("sites.") else name.split(".")[0]
        per_block[key] += value
    grad_norm = math.sqrt(squares)
    if not math.isfinite(grad_norm) or grad_norm <= 0:
        raise FloatingPointError(f"Zero/nonfinite action-adapter gradient at step {step}")
    if optimizer_update:
        optimizer.step()
    torch.cuda.synchronize()
    if any(p.grad is not None or p.requires_grad for p in transformer.parameters()):
        raise RuntimeError(f"Frozen VACE acquired a gradient at step {step}")
    return {"step": step, "sample_index": sample_index,
        "sample_id": context["rows"][sample_index]["global_clip_id"],
        "eligible_counterfactual": eligible, "hinge_active": hinge_active,
        "loss_correct": correct_value, "loss_wrong": wrong_value,
        "delta_wrong_minus_correct": wrong_value - correct_value if eligible else None,
        "ranking_loss": rank, "total_loss": total,
        "adapter_gradient_norm": grad_norm,
        "per_block_gradient_norm": {key: math.sqrt(value) for key, value in per_block.items()},
        "lr": optimizer.param_groups[0]["lr"], "timestep_index": timestep_index,
        "sigma": float(sigma),
        "gpu_allocated_mib": round(torch.cuda.memory_allocated() / 2**20, 1),
        "gpu_reserved_mib": round(torch.cuda.memory_reserved() / 2**20, 1),
        "gpu_peak_allocated_mib": round(torch.cuda.max_memory_allocated() / 2**20, 1)}


def save_checkpoint(context: dict, step: int, record: dict) -> None:
    directory = OUT / "checkpoints" / f"step_{step:04d}"
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "adapter.pt"
    temporary = directory / "adapter.partial.pt"
    torch.save({name: value.detach().cpu().contiguous()
                for name, value in context["adapter"].state_dict().items()}, temporary)
    temporary.replace(path)
    save_json(directory / "adapter.json", {"step": step, "sha256": digest(path),
        "adapter_parameters": 3_684_864, "last_step": record})
    temporary_state = directory / "training_state.partial.pt"
    torch.save({"step": step, "optimizer": context["optimizer"].state_dict(),
        "torch_rng": torch.get_rng_state(), "cuda_rng": torch.cuda.get_rng_state_all(),
        "noise_generator_state": context["noise_generator"].get_state()}, temporary_state)
    temporary_state.replace(directory / "training_state.pt")


def run(*, preflight: bool = False) -> dict:
    context = setup()
    if preflight:
        first_eligible_step = next(i + 1 for i, sample_index in enumerate(context["order"])
                                   if context["eligible"][sample_index])
        record = train_step(context, first_eligible_step, optimizer_update=False)
        if (not record["eligible_counterfactual"] or not record["hinge_active"]
                or record["loss_wrong"] is None or record["ranking_loss"] <= 0):
            raise RuntimeError("Contrastive branch did not execute in the no-update preflight")
        result = {"status": "pass; no optimizer step", "tested_step": first_eligible_step,
            "record": record, "initial_adapter_sha256": context["initial_adapter_sha256"],
            "pretrained_frozen": True, "optimizer_steps": 0}
        save_json(OUT / "preflight.json", result)
        return result
    train_dir = OUT / "train"
    train_dir.mkdir(parents=True, exist_ok=True)
    log_path = train_dir / "steps.jsonl"
    if log_path.exists() and log_path.stat().st_size:
        raise FileExistsError("M6F-2 log exists; refusing to overwrite or silently restart")
    copied_initial = train_dir / "initial_adapter.pt"
    if copied_initial.exists():
        if digest(copied_initial) != context["initial_adapter_sha256"]:
            raise RuntimeError("M6F-2 initial adapter copy changed")
    else:
        shutil.copyfile(context["initial_adapter_path"], copied_initial)
    protocol = {"milestone": "M6F-2", "training": True,
        "subset_manifest": str(M6D / "subset_manifest.json"),
        "subset_manifest_sha256": digest(M6D / "subset_manifest.json"),
        "initial_adapter": str(copied_initial),
        "initial_adapter_sha256": digest(copied_initial),
        "initialization": "same seed-42 zero-initialized M6D/M6E adapter, never loaded trained M6E weights",
        "architecture": "unchanged M6B-4 action-token cross-attention",
        "model": str(MODEL), "blocks": list(BLOCKS),
        "adapter_trainable_parameters": 3_684_864,
        "pretrained_vace_vae_text_frozen": True,
        "native_o0_and_mask_behavior_unchanged": True,
        "prompt": PROMPT, "optimizer": "AdamW", "learning_rate": LR,
        "batch_size": 1, "gradient_accumulation": 1, "seed": SEED,
        "backbone_dtype": "bfloat16", "optimizer_steps": STEPS,
        "future_loss_latent_positions": [1, 2, 3, 4],
        "counterfactual": "swap TURN_LEFT/RIGHT and MOVE_FORWARD_LEFT/RIGHT; keep MOVE_FORWARD/NOOP",
        "eligible_clips": 124, "unchanged_clips": 4,
        "margin": MARGIN, "lambda": LAMBDA,
        "objective": "Lcorrect + lambda * max(0, margin + Lcorrect - Lwrong), or Lcorrect if no swap",
        "sample_order_and_timestep_indices": "exact M6D/M6E seed-42/43 streams verified against all 6400 baseline log rows",
        "diffusion_noise": "seed-44 CUDA generator, BF16 noise as in M6D/M6E",
        "fixed_checkpoint_steps": list(CHECKPOINTS),
        "predetermined_official_checkpoint_step": 6400}
    save_json(train_dir / "protocol.json", protocol)
    started = time.monotonic()
    torch.cuda.reset_peak_memory_stats()
    counts = collections.Counter()
    eligible_count = active_count = 0
    with log_path.open("w", encoding="utf-8") as stream:
        for step in range(1, STEPS + 1):
            step_started = time.monotonic()
            record = train_step(context, step, optimizer_update=True)
            counts[record["sample_index"]] += 1
            eligible_count += int(record["eligible_counterfactual"])
            active_count += int(record["hinge_active"])
            record["active_hinge_fraction_eligible_cumulative"] = active_count / eligible_count
            record["step_runtime_seconds"] = round(time.monotonic() - step_started, 3)
            record["elapsed_seconds"] = round(time.monotonic() - started, 3)
            stream.write(json.dumps(record, allow_nan=False) + "\n")
            stream.flush()
            if step % 100 == 0:
                print("M6F2_STEP " + json.dumps({"step": step,
                    "loss_correct": record["loss_correct"], "ranking_loss": record["ranking_loss"],
                    "total_loss": record["total_loss"],
                    "active_hinge_fraction": record["active_hinge_fraction_eligible_cumulative"],
                    "grad_norm": record["adapter_gradient_norm"],
                    "elapsed_seconds": record["elapsed_seconds"]}), flush=True)
            if step in CHECKPOINTS:
                save_checkpoint(context, step, record)
    if counts != {index: 50 for index in range(128)} or eligible_count != 6200:
        raise RuntimeError("The deterministic 6400-step subset presentation count changed")
    if any(p.requires_grad or p.grad is not None or p._version != context["base_versions"][name]
           for name, p in context["transformer"].named_parameters()):
        raise RuntimeError("A pretrained VACE weight changed during training")
    summary = {"status": "complete", "optimizer_steps": STEPS,
        "official_checkpoint_step": 6400, "fixed_checkpoint_steps": list(CHECKPOINTS),
        "sample_counts": {str(i): counts[i] for i in range(128)},
        "eligible_presentations": eligible_count, "active_hinge_presentations": active_count,
        "active_hinge_fraction_eligible": active_count / eligible_count,
        "pretrained_vace_unchanged": True, "vae_text_not_loaded_or_trained": True,
        "runtime_seconds": round(time.monotonic() - started, 3),
        "peak_gpu_allocated_mib": round(torch.cuda.max_memory_allocated() / 2**20, 1),
        "peak_gpu_reserved_mib": round(torch.cuda.max_memory_reserved() / 2**20, 1),
        "log_path": str(log_path),
        "official_checkpoint": str(OUT / "checkpoints/step_6400/adapter.pt")}
    save_json(train_dir / "summary.json", summary)
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("preflight", "train"))
    args = parser.parse_args()
    print(json.dumps(run(preflight=args.mode == "preflight"), indent=2), flush=True)
