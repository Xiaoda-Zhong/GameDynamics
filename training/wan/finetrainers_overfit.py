#!/usr/bin/env python3
"""Train 320 tiny-dataset Wan LoRA steps and make controlled train-sample videos."""

from __future__ import annotations

import argparse
import gc
import json
import math
import time
from pathlib import Path

import torch
from decord import VideoReader, cpu

from training.wan.dataset import ActionCaptionVideoDataset
from training.wan.finetrainers_smoke import (
    GPUMemorySampler,
    MANIFEST,
    MODEL,
    OUTPUT,
    run_training,
)


TRAIN_OUTPUT = OUTPUT / "train_0320"
EVAL_OUTPUT = OUTPUT / "eval"
STEPS = 320
CHECKPOINT_STEPS = (40, 80, 160, 320)
LOSS_WINDOWS = ((1, 40), (41, 80), (81, 160), (161, 320))
SAMPLE_ID = "my_way_home_episode_0028_clip_000011"
SEED = 42
HEIGHT = 256
WIDTH = 448
FRAMES = 17
FPS = 8
INFERENCE_STEPS = 30
GUIDANCE_SCALE = 5.0
FLOW_SHIFT = 3.0


def _fixed_sample() -> dict:
    dataset = ActionCaptionVideoDataset(MANIFEST, expected_num_frames=FRAMES)
    matches = [sample for sample in dataset.records if sample["global_clip_id"] == SAMPLE_ID]
    if len(matches) != 1 or len(matches[0]["raw_actions"]) != 16:
        raise ValueError(f"Expected one 16-action training sample {SAMPLE_ID}")
    return matches[0]


def _training_summary() -> dict:
    path = TRAIN_OUTPUT / "summary.json"
    if not path.is_file():
        raise FileNotFoundError(f"320-step training summary is missing: {path}")
    summary = json.loads(path.read_text(encoding="utf-8"))
    if summary.get("steps") != STEPS or tuple(int(k) for k in summary["checkpoints"]) != CHECKPOINT_STEPS:
        raise ValueError("Training summary does not contain the four requested checkpoints")
    return summary


def _loss_statistics() -> dict[str, float]:
    path = TRAIN_OUTPUT / "steps.jsonl"
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    if len(rows) != STEPS or [row["step"] for row in rows] != list(range(1, STEPS + 1)):
        raise ValueError(f"Expected {STEPS} ordered optimizer-step metrics")
    if not all(math.isfinite(row["loss"]) and math.isfinite(row["lora_grad_norm"])
               and row["lora_grad_norm"] > 0 for row in rows):
        raise FloatingPointError("Loss or LoRA gradient became invalid")
    return {
        f"{start}-{end}": sum(rows[index - 1]["loss"] for index in range(start, end + 1)) / (end - start + 1)
        for start, end in LOSS_WINDOWS
    }


def _eval_settings(prompt: str) -> dict:
    return {
        "training_sample_id": SAMPLE_ID,
        "prompt": prompt,
        "seed": SEED,
        "height": HEIGHT,
        "width": WIDTH,
        "num_frames": FRAMES,
        "fps": FPS,
        "num_inference_steps": INFERENCE_STEPS,
        "guidance_scale": GUIDANCE_SCALE,
        "flow_shift": FLOW_SHIFT,
        "negative_prompt": None,
        "base_model": str(MODEL),
    }


def _generate_one(
    step: int, prompt: str, checkpoint: Path | None, *, eval_output: Path = EVAL_OUTPUT,
) -> tuple[Path, bool, float]:
    from diffusers import AutoencoderKLWan, WanPipeline
    from diffusers.schedulers import UniPCMultistepScheduler
    from diffusers.utils import export_to_video

    name = "step_000_base.mp4" if step == 0 else f"step_{step:03d}.mp4"
    video_path = eval_output / name
    metadata_path = eval_output / f"{name}.json"
    if video_path.exists():
        if not metadata_path.is_file():
            raise FileExistsError(f"Comparison video exists without metadata: {video_path}")
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        expected = {
            **_eval_settings(prompt), "step": step,
            "checkpoint": str(checkpoint) if checkpoint is not None else None,
            "output_path": str(video_path),
        }
        if any(metadata.get(key) != value for key, value in expected.items()):
            raise ValueError(f"Existing comparison metadata differs: {metadata_path}")
        video = VideoReader(str(video_path), ctx=cpu(0))
        if len(video) != FRAMES or tuple(video[0].shape[:2]) != (HEIGHT, WIDTH):
            raise RuntimeError(f"Existing comparison video is malformed: {video_path}")
        if step and not metadata.get("lora_load_verified"):
            raise RuntimeError(f"Existing LoRA load was not verified: {metadata_path}")
        return video_path, bool(metadata.get("lora_load_verified")), 0.0

    started = time.monotonic()
    vae = AutoencoderKLWan.from_pretrained(
        str(MODEL), subfolder="vae", torch_dtype=torch.float32, local_files_only=True,
    )
    pipeline = WanPipeline.from_pretrained(
        str(MODEL), vae=vae, torch_dtype=torch.bfloat16, local_files_only=True,
    )
    pipeline.scheduler = UniPCMultistepScheduler.from_config(
        pipeline.scheduler.config, flow_shift=FLOW_SHIFT,
    )
    loaded = False
    if checkpoint is not None:
        pipeline.load_lora_weights(str(checkpoint), adapter_name=f"step_{step:03d}")
        pipeline.set_adapters([f"step_{step:03d}"], [1.0])
        loaded = True
    pipeline.enable_model_cpu_offload()

    generator = torch.Generator(device="cuda").manual_seed(SEED)
    result = pipeline(
        prompt=prompt,
        negative_prompt=None,
        height=HEIGHT,
        width=WIDTH,
        num_frames=FRAMES,
        num_inference_steps=INFERENCE_STEPS,
        guidance_scale=GUIDANCE_SCALE,
        generator=generator,
    )
    frames = result.frames[0]
    if len(frames) != FRAMES:
        raise RuntimeError(f"Step {step} generated {len(frames)} frames, expected {FRAMES}")
    export_to_video(frames, str(video_path), fps=FPS)
    video = VideoReader(str(video_path), ctx=cpu(0))
    if len(video) != FRAMES or tuple(video[0].shape[:2]) != (HEIGHT, WIDTH):
        raise RuntimeError(f"Saved comparison video is malformed: {video_path}")
    duration = time.monotonic() - started
    metadata_path.write_text(json.dumps({
        **_eval_settings(prompt),
        "step": step,
        "checkpoint": str(checkpoint) if checkpoint is not None else None,
        "lora_load_verified": loaded,
        "output_path": str(video_path),
        "runtime_seconds": round(duration, 2),
    }, indent=2) + "\n", encoding="utf-8")
    del result, frames, video, pipeline, vae
    gc.collect()
    torch.cuda.empty_cache()
    return video_path, loaded, duration


def run_evaluation() -> dict:
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1 or not torch.cuda.is_bf16_supported():
        raise RuntimeError("Exactly one BF16-capable CUDA GPU is required")
    sample = _fixed_sample()
    summary = _training_summary()
    prompt = sample["prompt"]
    settings = _eval_settings(prompt)
    EVAL_OUTPUT.mkdir(parents=True, exist_ok=True)
    settings_path = EVAL_OUTPUT / "comparison_config.json"
    if settings_path.exists() and json.loads(settings_path.read_text(encoding="utf-8")) != settings:
        raise ValueError("Existing comparison settings differ from the fixed experiment settings")
    settings_path.write_text(json.dumps(settings, indent=2) + "\n", encoding="utf-8")

    videos = {}
    load_verified = {}
    started = time.monotonic()
    with GPUMemorySampler() as sampler:
        for step in (0, *CHECKPOINT_STEPS):
            checkpoint = None if step == 0 else Path(summary["checkpoints"][str(step)]["path"])
            path, loaded, duration = _generate_one(step, prompt, checkpoint)
            videos[str(step)] = str(path)
            if step:
                load_verified[str(step)] = loaded
            print(f"EVAL_VIDEO step={step} path={path} seconds={duration:.2f}", flush=True)
    eval_summary = {
        "settings": settings,
        "videos": videos,
        "checkpoint_load_verified": load_verified,
        "runtime_seconds": round(time.monotonic() - started, 2),
        "peak_gpu_vram_mib_observed": sampler.peak_mib,
    }
    (EVAL_OUTPUT / "summary.json").write_text(json.dumps(eval_summary, indent=2) + "\n", encoding="utf-8")
    return eval_summary


def run_all() -> dict:
    started = time.monotonic()
    training = run_training(
        Path("/tmp/finetrainers-wan-5c"), TRAIN_OUTPUT, STEPS, CHECKPOINT_STEPS,
        step_label="OVERFIT_STEP",
    )
    training["mean_loss_by_step_range"] = _loss_statistics()
    (TRAIN_OUTPUT / "summary.json").write_text(json.dumps(training, indent=2) + "\n", encoding="utf-8")
    gc.collect()
    torch.cuda.empty_cache()
    evaluation = run_evaluation()
    result = {
        "training": training,
        "evaluation": evaluation,
        "total_runtime_seconds": round(time.monotonic() - started, 2),
        "peak_gpu_vram_mib_observed": max(
            training["peak_gpu_vram_mib_observed"], evaluation["peak_gpu_vram_mib_observed"]
        ),
    }
    (OUTPUT / "overfit_0320_summary.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", nargs="?", default="all", choices=("all", "train", "eval"))
    options = parser.parse_args()
    if options.phase == "all":
        print(json.dumps(run_all(), indent=2))
    elif options.phase == "train":
        result = run_training(
            Path("/tmp/finetrainers-wan-5c"), TRAIN_OUTPUT, STEPS, CHECKPOINT_STEPS,
            step_label="OVERFIT_STEP",
        )
        result["mean_loss_by_step_range"] = _loss_statistics()
        (TRAIN_OUTPUT / "summary.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(result, indent=2))
    else:
        print(json.dumps(run_evaluation(), indent=2))


if __name__ == "__main__":
    main()
