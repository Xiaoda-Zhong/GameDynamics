#!/usr/bin/env python3
"""Run the fixed 800-step single-clip Wan LoRA memorization diagnostic."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import shutil
import time
from pathlib import Path

import torch

from training.wan.finetrainers_data import _verify_exact_frames
from training.wan.finetrainers_overfit import (
    _eval_settings,
    _fixed_sample,
    _generate_one,
)
from training.wan.finetrainers_smoke import GPUMemorySampler, MANIFEST, OUTPUT, run_training


ROOT = OUTPUT / "single_sample_0800"
TRAIN_OUTPUT = ROOT / "train"
EVAL_OUTPUT = ROOT / "eval"
STEPS = 800
CHECKPOINT_STEPS = (100, 200, 400, 800)
LOSS_WINDOWS = ((1, 100), (101, 200), (201, 400), (401, 800))


def _manifest_hash() -> str:
    return hashlib.sha256(MANIFEST.read_bytes()).hexdigest()


def _loss_statistics() -> tuple[dict[str, float], list[dict]]:
    rows = [json.loads(line) for line in (TRAIN_OUTPUT / "steps.jsonl").read_text().splitlines()]
    if len(rows) != STEPS or [row["step"] for row in rows] != list(range(1, STEPS + 1)):
        raise ValueError(f"Expected {STEPS} ordered optimizer-step metrics")
    if not all(
        math.isfinite(row["loss"])
        and math.isfinite(row["lora_grad_norm"])
        and row["lora_grad_norm"] > 0
        and row["learning_rate"] == 5e-5
        for row in rows
    ):
        raise FloatingPointError("Loss, gradient norm, or learning rate became invalid")
    intervals = []
    for start in range(1, STEPS + 1, 100):
        end = start + 99
        intervals.append({
            "steps": f"{start}-{end}",
            "mean_loss": sum(row["loss"] for row in rows[start - 1:end]) / 100,
        })
    (TRAIN_OUTPUT / "loss_100_step_intervals.jsonl").write_text(
        "".join(json.dumps(item) + "\n" for item in intervals), encoding="utf-8"
    )
    requested = {
        f"{start}-{end}": sum(row["loss"] for row in rows[start - 1:end]) / (end - start + 1)
        for start, end in LOSS_WINDOWS
    }
    return requested, intervals


def run_train() -> dict:
    sample = _fixed_sample()
    before = _manifest_hash()
    result = run_training(
        Path("/tmp/finetrainers-wan-5c"), TRAIN_OUTPUT, STEPS, CHECKPOINT_STEPS,
        step_label="SINGLE_SAMPLE_STEP", sample_id=sample["global_clip_id"],
    )
    if result["dataset_size"] != 1 or result["target_module_count"] != 240:
        raise RuntimeError("Single-sample count or verified LoRA target count changed")
    if _manifest_hash() != before:
        raise RuntimeError("Source manifest changed during training")
    result["source_manifest_sha256"] = before
    result["mean_loss_by_step_range"], result["mean_loss_by_100_step_interval"] = _loss_statistics()
    (TRAIN_OUTPUT / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def _training_summary() -> dict:
    path = TRAIN_OUTPUT / "summary.json"
    if not path.is_file():
        raise FileNotFoundError(f"800-step training summary is missing: {path}")
    summary = json.loads(path.read_text())
    if summary.get("steps") != STEPS or tuple(map(int, summary["checkpoints"])) != CHECKPOINT_STEPS:
        raise ValueError("Training summary does not have the requested four checkpoints")
    if summary.get("dataset_size") != 1 or summary.get("training_sample_id") != _fixed_sample()["global_clip_id"]:
        raise ValueError("Training summary does not describe the selected single clip")
    if summary.get("source_manifest_sha256") != _manifest_hash():
        raise RuntimeError("Source manifest hash has changed")
    return summary


def _save_source_video(sample: dict) -> Path:
    mapping_path = TRAIN_OUTPUT / "derived_dataset/source_map.jsonl"
    mapping = [json.loads(line) for line in mapping_path.read_text().splitlines()]
    if len(mapping) != 1 or mapping[0]["global_clip_id"] != sample["global_clip_id"]:
        raise ValueError("Derived dataset does not contain exactly the selected clip")
    if mapping[0]["prompt"] != sample["prompt"] or mapping[0]["source_rgb_frames"] != sample["rgb_frames"]:
        raise ValueError("Derived sample differs from source manifest")
    video_path = EVAL_OUTPUT / "source_17_frames_lossless.mp4"
    if not video_path.exists():
        shutil.copy2(mapping[0]["video_path"], video_path)
    _verify_exact_frames(video_path, sample["rgb_frames"])
    (EVAL_OUTPUT / "source_17_frames_lossless.mp4.json").write_text(json.dumps({
        "global_clip_id": sample["global_clip_id"],
        "source_rgb_frames": sample["rgb_frames"],
        "raw_actions": sample["raw_actions"],
        "prompt": sample["prompt"],
        "fps": sample["fps"],
        "num_frames": 17,
        "pixel_exact_to_source_pngs": True,
        "video_path": str(video_path),
    }, indent=2) + "\n")
    return video_path


def run_eval() -> dict:
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1 or not torch.cuda.is_bf16_supported():
        raise RuntimeError("Exactly one BF16-capable CUDA GPU is required")
    sample = _fixed_sample()
    training = _training_summary()
    EVAL_OUTPUT.mkdir(parents=True, exist_ok=True)
    settings = _eval_settings(sample["prompt"])
    config_path = EVAL_OUTPUT / "comparison_config.json"
    if config_path.exists() and json.loads(config_path.read_text()) != settings:
        raise ValueError("Existing comparison configuration differs")
    config_path.write_text(json.dumps(settings, indent=2) + "\n")
    source_video = _save_source_video(sample)
    videos = {}
    verified = {}
    started = time.monotonic()
    with GPUMemorySampler() as sampler:
        for step in (0, *CHECKPOINT_STEPS):
            checkpoint = None if step == 0 else Path(training["checkpoints"][str(step)]["path"])
            path, loaded, duration = _generate_one(
                step, sample["prompt"], checkpoint, eval_output=EVAL_OUTPUT,
            )
            videos[str(step)] = str(path)
            if step:
                verified[str(step)] = loaded
            print(f"EVAL_VIDEO step={step} path={path} seconds={duration:.2f}", flush=True)
    if not all(verified.values()) or tuple(map(int, verified)) != CHECKPOINT_STEPS:
        raise RuntimeError("One or more checkpoints did not load with WanPipeline.load_lora_weights")
    result = {
        "settings": settings,
        "source_video": str(source_video),
        "videos": videos,
        "checkpoint_load_verified": verified,
        "runtime_seconds": round(time.monotonic() - started, 2),
        "peak_gpu_vram_mib_observed": sampler.peak_mib,
    }
    (EVAL_OUTPUT / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def run_all() -> dict:
    started = time.monotonic()
    training = run_train()
    gc.collect()
    torch.cuda.empty_cache()
    evaluation = run_eval()
    result = {
        "training": training,
        "evaluation": evaluation,
        "total_runtime_seconds": round(time.monotonic() - started, 2),
        "peak_gpu_vram_mib_observed": max(
            training["peak_gpu_vram_mib_observed"], evaluation["peak_gpu_vram_mib_observed"]
        ),
    }
    (ROOT / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", nargs="?", default="all", choices=("all", "train", "eval"))
    phase = parser.parse_args().phase
    if phase == "all":
        result = run_all()
    elif phase == "train":
        result = run_train()
    else:
        result = run_eval()
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
