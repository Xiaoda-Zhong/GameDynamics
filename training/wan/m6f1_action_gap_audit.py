"""No-update correct-versus-swapped-action loss audit on the fixed M6D TRAIN subset."""

from __future__ import annotations

import csv
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from training.wan.m6d_select_128 import OUT as M6D, ROOT, digest
from training.wan.m6d_train_128 import PROMPT, load_cache
from training.wan.m6f0_action_contrastive_audit import CHECKPOINT, SWAP
from training.wan.vace_action_adapter import ACTION_TO_ID
from training.wan.vace_action_token_prototype import ActionTokenAdapter, BLOCKS, MODEL, temporal_mask

OUT = ROOT / "data/experiments/m6f1_action_contrastive_gap_audit"
REGIONS = {"early": 100, "middle": 500, "late": 900}
NOISE_SEED_BASE = 44


def stats(rows: list[dict]) -> dict:
    if not rows:
        return {"pairs": 0}
    gaps = np.asarray([row["delta"] for row in rows], dtype=np.float64)
    return {"pairs": len(rows), "fraction_delta_positive": float(np.mean(gaps > 0)),
        "mean_delta": float(np.mean(gaps)), "median_delta": float(np.median(gaps)),
        "std_delta_population": float(np.std(gaps, ddof=0)),
        "p10_delta": float(np.percentile(gaps, 10)),
        "p25_delta": float(np.percentile(gaps, 25)),
        "p50_delta": float(np.percentile(gaps, 50)),
        "p75_delta": float(np.percentile(gaps, 75)),
        "p90_delta": float(np.percentile(gaps, 90)),
        "mean_correct_loss": float(np.mean([row["loss_correct"] for row in rows])),
        "mean_wrong_loss": float(np.mean([row["loss_wrong"] for row in rows]))}


def run() -> dict:
    from diffusers import UniPCMultistepScheduler, WanVACETransformer3DModel

    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("BF16 CUDA is required for the frozen M6F-1 audit")
    sidecar = json.loads((CHECKPOINT.parent / "adapter.json").read_text())
    checkpoint_sha = digest(CHECKPOINT)
    if sidecar["step"] != 6400 or sidecar["sha256"] != checkpoint_sha:
        raise RuntimeError("Official frozen step-6400 adapter changed")
    prompt_cpu, samples, manifest_rows = load_cache()
    if len(manifest_rows) != 128 or len(samples) != 128:
        raise RuntimeError("The fixed 128-clip TRAIN subset changed")
    adapter = ActionTokenAdapter().to("cuda", dtype=torch.float32).eval().requires_grad_(False)
    adapter.load_state_dict(torch.load(CHECKPOINT, map_location="cpu", weights_only=True), strict=True)
    if sum(p.numel() for p in adapter.parameters()) != 3_684_864:
        raise RuntimeError("M6B-4 action adapter shape changed")
    adapter_versions = {name: p._version for name, p in adapter.named_parameters()}
    transformer = WanVACETransformer3DModel.from_pretrained(
        str(MODEL), subfolder="transformer", torch_dtype=torch.bfloat16, local_files_only=True
    ).to("cuda").eval().requires_grad_(False)
    base_versions = {name: p._version for name, p in transformer.named_parameters()}
    scheduler = UniPCMultistepScheduler.from_pretrained(
        str(MODEL), subfolder="scheduler", local_files_only=True)
    scheduler.set_timesteps(1000)
    if scheduler.config.prediction_type != "flow_prediction" or scheduler.config.flow_shift != 3.0:
        raise RuntimeError("Fixed M6D diffusion schedule changed")
    device = torch.device("cuda")
    prompt = prompt_cpu.to(device)
    mask = temporal_mask(device)
    control_scale = torch.ones(len(transformer.config.vace_layers), device=device, dtype=torch.bfloat16)
    if (prompt.shape != (1, 512, 4096) or mask.shape != (1, 1, 1792, 16)
            or any(p.requires_grad or p.grad is not None for p in transformer.parameters())):
        raise RuntimeError("Frozen VACE, text cache, or M6B-4 attention mask changed")

    def forward_loss(actions: torch.Tensor, target: torch.Tensor, control: torch.Tensor,
                     noisy: torch.Tensor, velocity: torch.Tensor,
                     timestep: torch.Tensor) -> float:
        tokens = adapter.tokens(actions)
        calls = {block: 0 for block in BLOCKS}
        handles = []
        try:
            for block in BLOCKS:
                def hook(_module, _inputs, hidden, *, site=block, action_tokens=tokens):
                    calls[site] += 1
                    with torch.autocast("cuda", dtype=torch.bfloat16):
                        return adapter.sites[str(site)](hidden, action_tokens, mask)
                handles.append(transformer.blocks[block].register_forward_hook(hook))
            predicted = transformer(hidden_states=noisy, timestep=timestep,
                encoder_hidden_states=prompt, control_hidden_states=control,
                control_hidden_states_scale=control_scale, return_dict=False)[0]
            if predicted.shape != target.shape:
                raise RuntimeError("The VACE denoiser output shape changed")
            loss = F.mse_loss(predicted.float()[:, :, 1:], velocity[:, :, 1:])
            value = float(loss)
        finally:
            for handle in handles:
                handle.remove()
        if not math.isfinite(value) or any(count != 1 for count in calls.values()):
            raise RuntimeError(f"Nonfinite future loss or missing action hook: {calls}")
        return value

    records = []
    excluded = []
    started = time.monotonic()
    torch.cuda.reset_peak_memory_stats()
    with torch.inference_mode():
        for index, (sample, metadata) in enumerate(zip(samples, manifest_rows, strict=True)):
            correct_cpu = sample["actions"]
            wrong_cpu = correct_cpu.clone()
            for source, destination in SWAP.items():
                wrong_cpu[correct_cpu == source] = destination
            swapped = int((wrong_cpu != correct_cpu).sum())
            noop_count = int((correct_cpu == ACTION_TO_ID["NOOP"]).sum())
            if swapped == 0:
                if not torch.equal(wrong_cpu, correct_cpu):
                    raise RuntimeError("Zero-swap clip unexpectedly changed")
                excluded.append({"sample_index": index, "sample_id": metadata["global_clip_id"],
                                 "reason": "left/right swap leaves all 16 action IDs unchanged"})
                continue
            if (correct_cpu.shape != (1, 16) or wrong_cpu.shape != (1, 16)
                    or any(bool(torch.any(wrong_cpu[correct_cpu == fixed] != fixed))
                           for fixed in (ACTION_TO_ID["MOVE_FORWARD"], ACTION_TO_ID["NOOP"]))):
                raise RuntimeError(f"Invalid deterministic action swap on sample {index}")
            target = sample["target"].to(device)
            control = sample["control"].to(device)
            correct = correct_cpu.to(device)
            wrong = wrong_cpu.to(device)
            if (target.shape != (1, 16, 5, 32, 56)
                    or control.shape != (1, 96, 5, 32, 56)
                    or not bool(torch.all(control[:, 32:] == 1))
                    or float(control[:, :16, 0].float().norm()) <= 0):
                raise RuntimeError(f"Native O0 control or target changed on sample {index}")
            for region_index, (region, timestep_index) in enumerate(REGIONS.items()):
                seed = NOISE_SEED_BASE + 3 * index + region_index
                timestep = scheduler.timesteps[timestep_index].to(device).expand(1)
                sigma = scheduler.sigmas[timestep_index].to(device).float()
                noise = torch.randn(target.shape, device=device, dtype=torch.bfloat16,
                    generator=torch.Generator(device="cuda").manual_seed(seed))
                noisy = ((1 - sigma) * target.float() + sigma * noise.float()).to(torch.bfloat16)
                velocity = noise.float() - target.float()
                # These same tensor objects are passed to both forwards; only action IDs differ.
                correct_loss = forward_loss(correct, target, control, noisy, velocity, timestep)
                wrong_loss = forward_loss(wrong, target, control, noisy, velocity, timestep)
                records.append({"sample_index": index, "sample_id": metadata["global_clip_id"],
                    "episode_id": metadata["episode_name"], "region": region,
                    "timestep_index": timestep_index, "timestep": int(timestep.item()),
                    "sigma": float(sigma), "noise_seed": seed,
                    "swapped_action_count": swapped, "noop_count": noop_count,
                    "noop_fraction": noop_count / 16,
                    "correct_action_ids": correct_cpu[0].tolist(),
                    "wrong_action_ids": wrong_cpu[0].tolist(),
                    "loss_correct": correct_loss, "loss_wrong": wrong_loss,
                    "delta": wrong_loss - correct_loss})
                del noise, noisy, velocity
            del target, control, correct, wrong
            if (index + 1) % 16 == 0 or index == 127:
                print("M6F1_PROGRESS " + json.dumps({"clips_seen": index + 1,
                    "eligible_pairs": len(records), "excluded_clips": len(excluded),
                    "elapsed_seconds": round(time.monotonic() - started, 1)}), flush=True)
    if len(excluded) != 4 or len(records) != (128 - len(excluded)) * len(REGIONS):
        raise RuntimeError("The fixed subset's eligible/unchanged count changed")
    if any(p.requires_grad or p.grad is not None or p._version != base_versions[name]
           for name, p in transformer.named_parameters()):
        raise RuntimeError("A pretrained VACE weight changed or acquired a gradient")
    if any(p.requires_grad or p.grad is not None or p._version != adapter_versions[name]
           for name, p in adapter.named_parameters()):
        raise RuntimeError("The frozen adapter changed or acquired a gradient")
    if digest(CHECKPOINT) != checkpoint_sha:
        raise RuntimeError("The official adapter checkpoint changed during the audit")

    summary = {"milestone": "M6F-1", "status": "complete; inference-only loss audit",
        "subset_manifest": str(M6D / "subset_manifest.json"),
        "subset_manifest_sha256": digest(M6D / "subset_manifest.json"),
        "checkpoint": str(CHECKPOINT), "checkpoint_sha256": checkpoint_sha,
        "eligible_clips": 128 - len(excluded), "unchanged_clips": len(excluded),
        "paired_forward_count": len(records), "denoiser_forward_count": 2 * len(records),
        "regions": {name: {"timestep_index": index,
            "timestep": int(scheduler.timesteps[index]), "sigma": float(scheduler.sigmas[index])}
            for name, index in REGIONS.items()},
        "noise_seed_rule": "44 + 3 * zero_based_subset_index + region_index(early=0,middle=1,late=2)",
        "same_o0_target_timestep_noise_text_and_model_within_each_pair": True,
        "only_action_ids_differ_within_each_pair": True,
        "future_only_loss_latent_positions": [1, 2, 3, 4],
        "native_vace_o0_and_mask_behavior_unchanged": True,
        "pretrained_vace_and_adapter_frozen_unchanged": True,
        "vae_and_text_encoder_not_loaded": True, "optimizer_steps": 0,
        "global": stats(records),
        "by_region": {name: stats([row for row in records if row["region"] == name])
                      for name in REGIONS},
        "excluded_clips": excluded,
        "runtime_seconds": round(time.monotonic() - started, 3),
        "peak_gpu_allocated_mib": round(torch.cuda.max_memory_allocated() / 2**20, 1),
        "peak_gpu_reserved_mib": round(torch.cuda.max_memory_reserved() / 2**20, 1)}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "raw_results.json").write_text(json.dumps(records, indent=2, allow_nan=False) + "\n")
    with (OUT / "raw_results.csv").open("w", newline="", encoding="utf-8") as stream:
        columns = [key for key in records[0] if key not in ("correct_action_ids", "wrong_action_ids")]
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows({key: row[key] for key in columns} for row in records)
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    return summary


if __name__ == "__main__":
    result = run()
    print(json.dumps({key: result[key] for key in ("eligible_clips", "unchanged_clips",
        "paired_forward_count", "global", "by_region", "runtime_seconds",
        "peak_gpu_allocated_mib", "peak_gpu_reserved_mib")}, indent=2), flush=True)
