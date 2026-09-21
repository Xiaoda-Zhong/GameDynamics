#!/usr/bin/env python3
"""Run the M5C-4 single-clip scene+action prompt diagnostic."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import time
from pathlib import Path

import torch

from training.wan.finetrainers_data import _encode_lossless_rgb, _verify_exact_frames
from training.wan.finetrainers_overfit import _eval_settings, _fixed_sample, _generate_one
from training.wan.finetrainers_smoke import GPUMemorySampler, MANIFEST, OUTPUT, run_training


ROOT = OUTPUT / "single_sample_scene_action_0800"
TRAIN_OUTPUT = ROOT / "train"
EVAL_OUTPUT = ROOT / "eval"
OLD_ROOT = OUTPUT / "single_sample_0800"
STEPS = 800
CHECKPOINT_STEPS = (100, 200, 400, 800)
LOSS_WINDOWS = ((1, 100), (101, 200), (201, 400), (401, 800))
SCENE_CONTEXT = "A first-person gameplay view in a retro 3D stone corridor with a red wall."


def _prompts() -> tuple[dict, str, str]:
    sample = _fixed_sample()
    action_prompt = sample["prompt"]
    expected = (
        "The player moves forward while turning left for 2 steps, stays still for "
        "8 steps, then moves forward while turning right for 6 steps."
    )
    if action_prompt != expected:
        raise ValueError("Source action prompt has changed")
    # Finetrainers uses one prompt per line; the supplied line break is one space.
    return sample, action_prompt, f"{SCENE_CONTEXT} {action_prompt}"


def _manifest_hash() -> str:
    return hashlib.sha256(MANIFEST.read_bytes()).hexdigest()


def _save_source_video(sample: dict, scene_prompt: str) -> Path:
    EVAL_OUTPUT.mkdir(parents=True, exist_ok=True)
    path = EVAL_OUTPUT / "source_17_frames_lossless.mp4"
    if not path.exists():
        _encode_lossless_rgb(sample["rgb_frames"], sample["fps"], path)
    _verify_exact_frames(path, sample["rgb_frames"])
    (EVAL_OUTPUT / "source_17_frames_lossless.mp4.json").write_text(json.dumps({
        "global_clip_id": sample["global_clip_id"],
        "source_rgb_frames": sample["rgb_frames"],
        "raw_actions": sample["raw_actions"],
        "source_prompt": sample["prompt"],
        "training_prompt": scene_prompt,
        "fps": sample["fps"],
        "num_frames": 17,
        "pixel_exact_to_source_pngs": True,
        "video_path": str(path),
    }, indent=2) + "\n", encoding="utf-8")
    return path


def _check_old_experiment(action_prompt: str) -> dict:
    old_config = json.loads((OLD_ROOT / "eval/comparison_config.json").read_text())
    if old_config != _eval_settings(action_prompt):
        raise ValueError("M5C-3 action-only inference settings differ from M5C-4")
    old_summary = json.loads((OLD_ROOT / "train/summary.json").read_text())
    if old_summary["source_manifest_sha256"] != _manifest_hash():
        raise ValueError("M5C-3 source manifest differs from the current manifest")
    if old_summary["steps"] != STEPS or old_summary["target_module_count"] != 240:
        raise ValueError("M5C-3 training settings are not the expected baseline")
    return old_summary


def run_base() -> dict:
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1 or not torch.cuda.is_bf16_supported():
        raise RuntimeError("Exactly one BF16-capable CUDA GPU is required")
    sample, action_prompt, scene_prompt = _prompts()
    _check_old_experiment(action_prompt)
    source_video = _save_source_video(sample, scene_prompt)
    prompts = {"action_only": action_prompt, "scene_action": scene_prompt}
    videos = {}
    started = time.monotonic()
    with GPUMemorySampler() as sampler:
        for label, prompt in prompts.items():
            directory = EVAL_OUTPUT / f"base_{label}"
            directory.mkdir(parents=True, exist_ok=True)
            video, loaded, duration = _generate_one(0, prompt, None, eval_output=directory)
            if loaded:
                raise RuntimeError("Base control unexpectedly loaded a LoRA")
            videos[label] = str(video)
            print(f"BASE_CONTROL prompt={label} video={video} seconds={duration:.2f}", flush=True)
    result = {
        "source_manifest_sha256": _manifest_hash(),
        "source_video": str(source_video),
        "prompts": prompts,
        "inference_settings": {label: _eval_settings(prompt) for label, prompt in prompts.items()},
        "videos": videos,
        "runtime_seconds": round(time.monotonic() - started, 2),
        "peak_gpu_vram_mib_observed": sampler.peak_mib,
    }
    (EVAL_OUTPUT / "base_controls.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def _base_summary(scene_prompt: str) -> dict:
    summary = json.loads((EVAL_OUTPUT / "base_controls.json").read_text())
    if summary["prompts"]["scene_action"] != scene_prompt or summary["source_manifest_sha256"] != _manifest_hash():
        raise ValueError("Base controls do not match this experiment")
    if not all(Path(path).is_file() for path in summary["videos"].values()):
        raise FileNotFoundError("One or more base controls are missing")
    return summary


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
    intervals = [{
        "steps": f"{start}-{start + 99}",
        "mean_loss": sum(row["loss"] for row in rows[start - 1:start + 99]) / 100,
    } for start in range(1, STEPS + 1, 100)]
    (TRAIN_OUTPUT / "loss_100_step_intervals.jsonl").write_text(
        "".join(json.dumps(item) + "\n" for item in intervals), encoding="utf-8",
    )
    requested = {
        f"{start}-{end}": sum(row["loss"] for row in rows[start - 1:end]) / (end - start + 1)
        for start, end in LOSS_WINDOWS
    }
    return requested, intervals


def run_train() -> dict:
    sample, _, scene_prompt = _prompts()
    _base_summary(scene_prompt)
    before = _manifest_hash()
    result = run_training(
        Path("/tmp/finetrainers-wan-5c"), TRAIN_OUTPUT, STEPS, CHECKPOINT_STEPS,
        step_label="SCENE_ACTION_STEP", sample_id=sample["global_clip_id"],
        prompt_override=scene_prompt,
    )
    if result["dataset_size"] != 1 or result["target_module_count"] != 240:
        raise RuntimeError("Single-sample count or verified LoRA target count changed")
    mapping = [json.loads(line) for line in (TRAIN_OUTPUT / "derived_dataset/source_map.jsonl").read_text().splitlines()]
    if len(mapping) != 1 or any((
        mapping[0]["global_clip_id"] != sample["global_clip_id"],
        mapping[0]["source_rgb_frames"] != sample["rgb_frames"],
        mapping[0]["raw_actions"] != sample["raw_actions"],
        mapping[0]["source_prompt"] != sample["prompt"],
        mapping[0]["prompt"] != scene_prompt,
    )):
        raise ValueError("Derived training sample differs from the source or requested prompt")
    if (TRAIN_OUTPUT / "derived_dataset/prompt.txt").read_text() != scene_prompt + "\n":
        raise ValueError("Finetrainers prompt file differs from requested scene+action prompt")
    _verify_exact_frames(Path(mapping[0]["video_path"]), sample["rgb_frames"])
    if _manifest_hash() != before:
        raise RuntimeError("Source manifest changed during training")
    result["source_manifest_sha256"] = before
    result["training_prompt"] = scene_prompt
    result["source_prompt"] = sample["prompt"]
    result["mean_loss_by_step_range"], result["mean_loss_by_100_step_interval"] = _loss_statistics()
    (TRAIN_OUTPUT / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def _training_summary(scene_prompt: str) -> dict:
    summary = json.loads((TRAIN_OUTPUT / "summary.json").read_text())
    if (
        summary.get("steps") != STEPS
        or tuple(map(int, summary["checkpoints"])) != CHECKPOINT_STEPS
        or summary.get("dataset_size") != 1
        or summary.get("training_sample_id") != _fixed_sample()["global_clip_id"]
        or summary.get("training_prompt") != scene_prompt
        or summary.get("source_manifest_sha256") != _manifest_hash()
    ):
        raise ValueError("Training summary does not match this experiment")
    return summary


def run_eval() -> dict:
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1 or not torch.cuda.is_bf16_supported():
        raise RuntimeError("Exactly one BF16-capable CUDA GPU is required")
    sample, _, scene_prompt = _prompts()
    base = _base_summary(scene_prompt)
    training = _training_summary(scene_prompt)
    settings = _eval_settings(scene_prompt)
    config_path = EVAL_OUTPUT / "comparison_config.json"
    if config_path.exists() and json.loads(config_path.read_text()) != settings:
        raise ValueError("Existing comparison configuration differs")
    config_path.write_text(json.dumps(settings, indent=2) + "\n")
    source_video = _save_source_video(sample, scene_prompt)
    videos = {}
    verified = {}
    started = time.monotonic()
    with GPUMemorySampler() as sampler:
        for step in CHECKPOINT_STEPS:
            checkpoint = Path(training["checkpoints"][str(step)]["path"])
            path, loaded, duration = _generate_one(
                step, scene_prompt, checkpoint, eval_output=EVAL_OUTPUT,
            )
            videos[str(step)] = str(path)
            verified[str(step)] = loaded
            print(f"EVAL_VIDEO step={step} path={path} seconds={duration:.2f}", flush=True)
    if not all(verified.values()) or tuple(map(int, verified)) != CHECKPOINT_STEPS:
        raise RuntimeError("One or more checkpoints did not load with WanPipeline.load_lora_weights")
    result = {
        "settings": settings,
        "source_video": str(source_video),
        "base_videos": base["videos"],
        "videos": videos,
        "checkpoint_load_verified": verified,
        "runtime_seconds": round(time.monotonic() - started, 2),
        "peak_gpu_vram_mib_observed": sampler.peak_mib,
    }
    (EVAL_OUTPUT / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def run_all() -> dict:
    started = time.monotonic()
    base = run_base()
    gc.collect()
    torch.cuda.empty_cache()
    training = run_train()
    gc.collect()
    torch.cuda.empty_cache()
    evaluation = run_eval()
    result = {
        "base_controls": base,
        "training": training,
        "evaluation": evaluation,
        "total_runtime_seconds": round(time.monotonic() - started, 2),
        "peak_gpu_vram_mib_observed": max(
            base["peak_gpu_vram_mib_observed"],
            training["peak_gpu_vram_mib_observed"],
            evaluation["peak_gpu_vram_mib_observed"],
        ),
    }
    (ROOT / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", nargs="?", default="all", choices=("all", "base", "train", "eval"))
    phase = parser.parse_args().phase
    functions = {"all": run_all, "base": run_base, "train": run_train, "eval": run_eval}
    print(json.dumps(functions[phase](), indent=2))


if __name__ == "__main__":
    main()
