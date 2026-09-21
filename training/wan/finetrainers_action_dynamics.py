#!/usr/bin/env python3
"""Generate and score the M5C-5 action-dynamics counterfactuals without training."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
from decord import VideoReader, cpu

from data_pipeline.build_action_captions import generate_action_prompt, run_length_encode_actions
from training.wan.action_dynamics_metrics import (
    ACTION_SIGNAL, FLOW_PARAMETERS, FRAMES, HEIGHT, LEFT, NOOP, RIGHT, SEEDS, WIDTH,
    calibrate_source, evaluate_video, flip_test, measure_video,
)
from training.wan.finetrainers_overfit import _eval_settings, _fixed_sample
from training.wan.finetrainers_scene_action_single_sample import SCENE_CONTEXT
from training.wan.finetrainers_smoke import GPUMemorySampler, MANIFEST, MODEL, OUTPUT


M5C4_ROOT = OUTPUT / "single_sample_scene_action_0800"
ROOT = OUTPUT / "single_sample_action_dynamics_0800"
VIDEOS = ROOT / "videos"
METRICS = ROOT / "metrics"
CHECKPOINT = M5C4_ROOT / "train/lora_weights/000800"
SOURCE_VIDEO = M5C4_ROOT / "eval/source_17_frames_lossless.mp4"
CONDITIONS = ("ORIGINAL", "REVERSED", "NOOP")
LORA_SCALE = 1.0
THRESHOLDS = {
    "aggregate_turn_direction_accuracy_minimum": 0.70,
    "median_r_noop_maximum": 0.50,
    "median_absolute_action_motion_correlation_minimum": 0.50,
    "counterfactual_flip_successes_minimum_of_5": 4,
}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _condition_actions() -> dict[str, list[str]]:
    return {
        "ORIGINAL": [LEFT] * 2 + [NOOP] * 8 + [RIGHT] * 6,
        "REVERSED": [RIGHT] * 2 + [NOOP] * 8 + [LEFT] * 6,
        "NOOP": [NOOP] * 16,
    }


def _protocol() -> dict:
    sample = _fixed_sample()
    actions = _condition_actions()
    if sample["raw_actions"] != actions["ORIGINAL"]:
        raise ValueError("M5C-4 source actions differ from the requested original sequence")
    if not (CHECKPOINT / "pytorch_lora_weights.safetensors").is_file() or not SOURCE_VIDEO.is_file():
        raise FileNotFoundError("M5C-4 step-800 adapter or exact source video is missing")
    original_action_prompt = generate_action_prompt(run_length_encode_actions(actions["ORIGINAL"]))
    prompts = {
        label: f"{SCENE_CONTEXT} {generate_action_prompt(run_length_encode_actions(sequence))}"
        for label, sequence in actions.items()
    }
    m5c4 = json.loads((M5C4_ROOT / "eval/comparison_config.json").read_text())
    if m5c4 != _eval_settings(prompts["ORIGINAL"]) or sample["prompt"] != original_action_prompt:
        raise ValueError("M5C-4 prompt or inference settings differ from the diagnostic baseline")
    prior_video = json.loads((M5C4_ROOT / "eval/step_800.mp4.json").read_text())
    if prior_video["checkpoint"] != str(CHECKPOINT) or not prior_video["lora_load_verified"]:
        raise ValueError("M5C-4 step-800 checkpoint was not validated as expected")
    common_inference = {key: value for key, value in m5c4.items() if key not in ("prompt", "seed")}
    return {
        "milestone": "M5C-5 action-dynamics diagnostic",
        "sample_id": sample["global_clip_id"],
        "source_manifest": str(MANIFEST),
        "source_manifest_sha256": _sha256(MANIFEST),
        "source_video": str(SOURCE_VIDEO),
        "source_video_sha256": _sha256(SOURCE_VIDEO),
        "checkpoint": str(CHECKPOINT),
        "checkpoint_weights_sha256": _sha256(CHECKPOINT / "pytorch_lora_weights.safetensors"),
        "lora_scale": LORA_SCALE,
        "seeds": list(SEEDS),
        "scene_prefix": SCENE_CONTEXT,
        "conditions": {
            label: {"raw_actions": sequence, "action_prompt": generate_action_prompt(run_length_encode_actions(sequence)), "prompt": prompts[label]}
            for label, sequence in actions.items()
        },
        "inference_settings_without_seed_or_prompt": common_inference,
        "vae_dtype": "float32",
        "pipeline_dtype": "bfloat16",
        "scheduler": "UniPCMultistepScheduler with flow_shift=3.0",
        "flow_parameters": FLOW_PARAMETERS,
        "metric_rules": {
            "direction_and_correlation_flow": "mean_u per transition, multiplied by the sign that makes source left turns positive",
            "turn_direction_accuracy": "fraction of turn transitions whose signed mean_u has the expected strict sign; zero is incorrect",
            "r_noop_within_video": "mean NOOP flow magnitude / mean TURN flow magnitude for ORIGINAL and REVERSED",
            "r_noop_noop_condition": "undefined within a 16-NOOP video; also report its mean magnitude divided by the matched seed's mean TURN magnitude across ORIGINAL and REVERSED",
            "correlation": "Pearson r between actions (+1 left, 0 NOOP, -1 right) and calibrated mean_u; undefined for all-NOOP",
            "aggregate_undefined_correlation": "count as |r|=0 for the pass criterion if a turn-containing video has constant flow",
            "flip_success": "for each seed, ORIGINAL first >0 and last <0, REVERSED first <0 and last >0 in calibrated segment-mean u; all four required",
            "aggregate_population": "all 10 turn-containing videos, 8 turn transitions each; NOOP-only videos are secondary suppression controls",
        },
        "thresholds": THRESHOLDS,
    }


def _write_protocol() -> dict:
    protocol = _protocol()
    ROOT.mkdir(parents=True, exist_ok=True)
    path = ROOT / "protocol.json"
    if path.exists() and json.loads(path.read_text()) != protocol:
        raise ValueError("Existing M5C-5 protocol differs from current inputs or rules")
    path.write_text(json.dumps(protocol, indent=2) + "\n", encoding="utf-8")
    return protocol


def _video_paths(condition: str, seed: int) -> tuple[Path, Path]:
    path = VIDEOS / f"{condition.lower()}_seed_{seed}.mp4"
    return path, path.with_suffix(".mp4.json")


def _check_video(path: Path) -> None:
    video = VideoReader(str(path), ctx=cpu(0))
    if len(video) != FRAMES or tuple(video[0].shape) != (HEIGHT, WIDTH, 3):
        raise ValueError(f"Generated video has unexpected frames or dimensions: {path}")


def _matches_m5c4_reference(path: Path) -> bool:
    reference = VideoReader(str(M5C4_ROOT / "eval/step_800.mp4"), ctx=cpu(0))
    current = VideoReader(str(path), ctx=cpu(0))
    return len(reference) == len(current) == FRAMES and all(
        np.array_equal(reference[index].asnumpy(), current[index].asnumpy())
        for index in range(FRAMES)
    )


def generate() -> dict:
    """Use one fixed M5C-4 LoRA pipeline for all 15 matched-seed generations."""
    from diffusers import AutoencoderKLWan, WanPipeline
    from diffusers.schedulers import UniPCMultistepScheduler
    from diffusers.utils import export_to_video

    protocol = _write_protocol()
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1 or not torch.cuda.is_bf16_supported():
        raise RuntimeError("Exactly one BF16-capable CUDA GPU is required")
    VIDEOS.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    outputs = []
    with GPUMemorySampler() as sampler:
        vae = AutoencoderKLWan.from_pretrained(
            str(MODEL), subfolder="vae", torch_dtype=torch.float32, local_files_only=True,
        )
        pipeline = WanPipeline.from_pretrained(
            str(MODEL), vae=vae, torch_dtype=torch.bfloat16, local_files_only=True,
        )
        pipeline.scheduler = UniPCMultistepScheduler.from_config(
            pipeline.scheduler.config, flow_shift=3.0,
        )
        pipeline.load_lora_weights(str(CHECKPOINT), adapter_name="m5c4_step_800")
        pipeline.set_adapters(["m5c4_step_800"], [LORA_SCALE])
        pipeline.enable_model_cpu_offload()
        for condition in CONDITIONS:
            details = protocol["conditions"][condition]
            for seed in SEEDS:
                path, metadata_path = _video_paths(condition, seed)
                expected = {
                    "condition": condition,
                    "seed": seed,
                    "prompt": details["prompt"],
                    "raw_actions": details["raw_actions"],
                    "checkpoint": str(CHECKPOINT),
                    "lora_scale": LORA_SCALE,
                    "inference_settings": {
                        **protocol["inference_settings_without_seed_or_prompt"],
                        "seed": seed, "prompt": details["prompt"],
                    },
                    "video_path": str(path),
                }
                if path.exists():
                    if not metadata_path.is_file():
                        raise FileExistsError(f"Video exists without metadata: {path}")
                    metadata = json.loads(metadata_path.read_text())
                    if any(metadata.get(key) != value for key, value in expected.items()):
                        raise ValueError(f"Existing video metadata differs: {metadata_path}")
                    _check_video(path)
                else:
                    if metadata_path.exists():
                        raise FileExistsError(f"Metadata exists without video: {metadata_path}")
                    began = time.monotonic()
                    generator = torch.Generator(device="cuda").manual_seed(seed)
                    result = pipeline(
                        prompt=details["prompt"], negative_prompt=None,
                        height=HEIGHT, width=WIDTH, num_frames=FRAMES,
                        num_inference_steps=30, guidance_scale=5.0,
                        generator=generator,
                    )
                    frames = result.frames[0]
                    if len(frames) != FRAMES:
                        raise RuntimeError(f"Generated {len(frames)} frames instead of {FRAMES}")
                    temporary = path.with_name(path.stem + ".partial.mp4")
                    try:
                        export_to_video(frames, str(temporary), fps=8)
                        _check_video(temporary)
                        temporary.replace(path)
                    except BaseException:
                        temporary.unlink(missing_ok=True)
                        raise
                    duration = round(time.monotonic() - began, 2)
                    metadata_path.write_text(json.dumps({
                        **expected,
                        "runtime_seconds": duration,
                        "lora_load_verified": True,
                    }, indent=2) + "\n", encoding="utf-8")
                    del result, frames
                    print(f"GENERATED condition={condition} seed={seed} seconds={duration} path={path}", flush=True)
                if condition == "ORIGINAL" and seed == 42 and not _matches_m5c4_reference(path):
                    raise RuntimeError("ORIGINAL seed 42 does not reproduce M5C-4 step-800 video pixel for pixel")
                outputs.append(str(path))
        del pipeline, vae
        gc.collect()
        torch.cuda.empty_cache()
    if len(outputs) != 15:
        raise RuntimeError(f"Expected 15 videos, found {len(outputs)}")
    summary = {
        "checkpoint": str(CHECKPOINT),
        "checkpoint_load_verified": True,
        "original_seed_42_pixel_identical_to_m5c4": True,
        "videos": outputs,
        "runtime_seconds": round(time.monotonic() - started, 2),
        "peak_gpu_vram_mib_observed": sampler.peak_mib,
    }
    (ROOT / "generation_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


def _transition_record(
    row: dict, condition: str, seed: int | None, action: str,
    video_path: Path, left_positive_multiplier: int,
) -> dict:
    calibrated_u = left_positive_multiplier * row["mean_u"]
    signal = ACTION_SIGNAL[action]
    return {
        **row,
        "condition": condition,
        "seed": seed,
        "action": action,
        "video_path": str(video_path),
        "calibrated_mean_u": calibrated_u,
        "inferred_turn_direction": "LEFT" if calibrated_u > 0 else "RIGHT" if calibrated_u < 0 else "ZERO",
        "turn_direction_correct": signal * calibrated_u > 0 if signal else None,
    }


def _median_or_none(values: list[float | None]) -> float | None:
    if not values or any(value is None or not math.isfinite(value) for value in values):
        return None
    return float(np.median(values))


def analyze() -> dict:
    """Measure the source first, then all 15 videos using the same fixed flow rule."""
    import cv2

    protocol = _write_protocol()
    generation = json.loads((ROOT / "generation_summary.json").read_text())
    if len(generation["videos"]) != 15 or not generation["checkpoint_load_verified"]:
        raise ValueError("Generation summary does not contain the 15 validated LoRA videos")
    if cv2.__version__ != "4.12.0":
        raise ValueError(f"Expected isolated OpenCV 4.12.0, found {cv2.__version__}")
    METRICS.mkdir(parents=True, exist_ok=True)
    source_actions = protocol["conditions"]["ORIGINAL"]["raw_actions"]
    source_rows = measure_video(SOURCE_VIDEO)
    calibration = calibrate_source(source_rows, source_actions)
    source_metric = evaluate_video(source_rows, source_actions, calibration["left_positive_multiplier"])
    source_metric.update({"condition": "SOURCE", "seed": None, "video_path": str(SOURCE_VIDEO)})
    transition_records = [
        _transition_record(
            row, "SOURCE", None, source_actions[index], SOURCE_VIDEO,
            calibration["left_positive_multiplier"],
        )
        for index, row in enumerate(source_rows)
    ]
    video_records = [source_metric]
    by_condition_seed = {}
    for condition in CONDITIONS:
        actions = protocol["conditions"][condition]["raw_actions"]
        for seed in SEEDS:
            path, _ = _video_paths(condition, seed)
            if str(path) not in generation["videos"]:
                raise ValueError(f"Generation summary omitted {path}")
            rows = measure_video(path)
            metric = evaluate_video(rows, actions, calibration["left_positive_multiplier"])
            metric.update({"condition": condition, "seed": seed, "video_path": str(path)})
            by_condition_seed[(condition, seed)] = metric
            video_records.append(metric)
            transition_records.extend(
                _transition_record(
                    row, condition, seed, actions[index], path,
                    calibration["left_positive_multiplier"],
                ) for index, row in enumerate(rows)
            )
            print(f"FLOW condition={condition} seed={seed} direction={metric['turn_direction_accuracy']} r={metric['action_motion_r_signed']}", flush=True)
    for seed in SEEDS:
        noop = by_condition_seed[("NOOP", seed)]
        turn_magnitude = float(np.mean([
            by_condition_seed[(condition, seed)]["mean_turn_magnitude"]
            for condition in ("ORIGINAL", "REVERSED")
        ]))
        noop["r_noop_matched_seed_turn_baseline"] = (
            noop["mean_noop_magnitude"] / turn_magnitude if turn_magnitude > 0 else None
        )
    flips = []
    for seed in SEEDS:
        record = flip_test(by_condition_seed[("ORIGINAL", seed)], by_condition_seed[("REVERSED", seed)])
        flips.append({"seed": seed, **record})
    actionable = [by_condition_seed[(condition, seed)] for condition in ("ORIGINAL", "REVERSED") for seed in SEEDS]
    direction_correct = sum(record["turn_correct"] for record in actionable)
    direction_total = sum(record["turn_count"] for record in actionable)
    median_ratio = _median_or_none([record["r_noop_within_video"] for record in actionable])
    median_abs_r = float(np.median([
        record["action_motion_r_absolute"] if record["action_motion_r_absolute"] is not None else 0.0
        for record in actionable
    ]))
    flip_successes = sum(record["success"] for record in flips)
    criteria = {
        "turn_direction_accuracy": direction_correct / direction_total >= THRESHOLDS["aggregate_turn_direction_accuracy_minimum"],
        "noop_suppression": median_ratio is not None and median_ratio <= THRESHOLDS["median_r_noop_maximum"],
        "action_motion_correlation": median_abs_r >= THRESHOLDS["median_absolute_action_motion_correlation_minimum"],
        "counterfactual_flip": flip_successes >= THRESHOLDS["counterfactual_flip_successes_minimum_of_5"],
    }
    condition_aggregates = {}
    for condition in CONDITIONS:
        records = [by_condition_seed[(condition, seed)] for seed in SEEDS]
        condition_aggregates[condition] = {
            "turn_direction_accuracy": (
                sum(record["turn_correct"] for record in records) / sum(record["turn_count"] for record in records)
                if condition != "NOOP" else None
            ),
            "median_r_noop_within_video": _median_or_none([record["r_noop_within_video"] for record in records]),
            "median_action_motion_r_signed": _median_or_none([record["action_motion_r_signed"] for record in records]),
            "median_action_motion_r_absolute": _median_or_none([record["action_motion_r_absolute"] for record in records]),
            "median_matched_seed_noop_ratio": (
                _median_or_none([record.get("r_noop_matched_seed_turn_baseline") for record in records])
                if condition == "NOOP" else None
            ),
        }
    summary = {
        "opencv_version": cv2.__version__,
        "flow_parameters": FLOW_PARAMETERS,
        "source_calibration": calibration,
        "source_metrics": source_metric,
        "source_metrics_path": str(METRICS / "source.json"),
        "per_transition_metrics_path": str(METRICS / "transitions.jsonl"),
        "per_video_metrics_path": str(METRICS / "video_metrics.jsonl"),
        "condition_aggregates": condition_aggregates,
        "turn_direction_correct": direction_correct,
        "turn_direction_total": direction_total,
        "aggregate_turn_direction_accuracy": direction_correct / direction_total,
        "median_r_noop_within_turn_containing_videos": median_ratio,
        "median_absolute_action_motion_correlation": median_abs_r,
        "flip_tests": flips,
        "counterfactual_flip_successes": flip_successes,
        "counterfactual_flip_success_rate": flip_successes / len(SEEDS),
        "thresholds": THRESHOLDS,
        "criteria_met": criteria,
        "pass_natural_language_action_conditioning": all(criteria.values()),
    }
    (METRICS / "transitions.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in transition_records), encoding="utf-8",
    )
    (METRICS / "video_metrics.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in video_records), encoding="utf-8",
    )
    (METRICS / "source.json").write_text(json.dumps({
        "calibration": calibration, "metrics": source_metric,
        "transitions": transition_records[:16],
    }, indent=2) + "\n")
    (METRICS / "aggregate.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", nargs="?", default="all", choices=("all", "generate", "analyze"))
    phase = parser.parse_args().phase
    if phase == "generate":
        result = generate()
    elif phase == "analyze":
        result = analyze()
    else:
        generate()
        result = analyze()
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
