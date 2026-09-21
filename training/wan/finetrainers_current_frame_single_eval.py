#!/usr/bin/env python3
"""M6A-2: one-clip Control LoRA overfit and O0 counterfactual evaluation."""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw

from training.wan.current_frame_metrics import (
    FRAMES, HEIGHT, WIDTH, generated_frames_rgb, mean_absolute_rgb_error,
    source_frame_rgb, ssim_rgb,
)
from training.wan.finetrainers_current_frame_eval import (
    FLOW_SHIFT, FPS, GUIDANCE, SEED, SOURCE, STEPS, _control_latent, _load_pipeline,
)
from training.wan.finetrainers_current_frame_train import (
    SINGLE_ROOT, SINGLE_SAMPLE_ID, SINGLE_TRAIN, TRAIN as M6A_TRAIN, _hash_manifest,
)
from training.wan.finetrainers_smoke import GPUMemorySampler, _require_official_source


EVAL = SINGLE_ROOT / "eval"
CHECKPOINTS = (100, 200, 400, 800)


def _training_row() -> dict:
    rows = [json.loads(line) for line in (SINGLE_TRAIN / "derived_dataset/source_map.jsonl").read_text().splitlines()]
    if len(rows) != 1 or rows[0]["global_clip_id"] != SINGLE_SAMPLE_ID:
        raise ValueError("M6A-2 must train on exactly the requested source clip")
    if len(rows[0]["source_rgb_frames"]) != FRAMES or len(rows[0]["raw_actions"]) != 16:
        raise ValueError("M6A-2 source must retain 17 observations and 16 actions")
    return rows[0]


def _other_sources() -> list[dict]:
    rows = [json.loads(line) for line in (M6A_TRAIN / "derived_dataset/source_map.jsonl").read_text().splitlines()]
    if len(rows) != 16 or len({row["global_clip_id"] for row in rows}) != 16:
        raise ValueError("Expected the existing 16-clip M6A source mapping")
    matches = [row for row in rows if row["global_clip_id"] == SINGLE_SAMPLE_ID]
    if len(matches) != 1 or matches[0]["source_rgb_frames"] != _training_row()["source_rgb_frames"]:
        raise ValueError("M6A and M6A-2 original state references differ")
    return [row for row in rows if row["global_clip_id"] != SINGLE_SAMPLE_ID]


def _select_counterfactuals(a: dict) -> list[dict]:
    """Select five real O0 states by deterministic max-min RGB distance, before generation."""
    candidates = _other_sources()
    a_frame = source_frame_rgb(a["source_rgb_frames"][0])
    frames = [source_frame_rgb(row["source_rgb_frames"][0]) for row in candidates]
    selected: list[int] = []
    for _ in range(5):
        ranked = []
        for index, frame in enumerate(frames):
            if index in selected or ssim_rgb(frame, a_frame) >= 0.70:
                continue
            references = [a_frame, *(frames[j] for j in selected)]
            min_mae = min(mean_absolute_rgb_error(frame, reference) for reference in references)
            ranked.append((min_mae, index))
        if not ranked:
            raise ValueError("Could not select five clearly different real initial states")
        selected.append(max(ranked, key=lambda item: (item[0], -item[1]))[1])
    return [{
        "sample_id": candidates[index]["global_clip_id"],
        "source_o0": candidates[index]["source_rgb_frames"][0],
        "source_video": candidates[index]["video_path"],
        "o0_ssim_vs_training_a": ssim_rgb(frames[index], a_frame),
        "o0_mae_vs_training_a_rgb_0_255": mean_absolute_rgb_error(frames[index], a_frame),
    } for index in selected]


def _protocol() -> dict:
    summary = json.loads((SINGLE_TRAIN / "summary.json").read_text())
    if (summary["steps"] != 800 or summary["dataset_size"] != 1
            or summary["training_sample_id"] != SINGLE_SAMPLE_ID
            or summary["source_manifest_sha256"] != _hash_manifest()
            or tuple(map(int, summary["checkpoint_paths"])) != CHECKPOINTS
            or summary["rank"] != 128 or summary["lora_alpha"] != 128
            or summary["target_module_count"] != 241):
        raise ValueError("M6A-2 training summary does not match the declared protocol")
    a = _training_row()
    return {
        "training_sample_id": SINGLE_SAMPLE_ID,
        "source_manifest_sha256": _hash_manifest(),
        "training_o0": a["source_rgb_frames"][0],
        "source_video": a["video_path"],
        "prompt": a["prompt"],
        "counterfactuals": _select_counterfactuals(a),
        "checkpoints": {str(k): summary["checkpoint_paths"][str(k)]["path"] for k in CHECKPOINTS},
        "untrained_control_definition": (
            "Expanded patch embedding and real O0 control hook, with all Control LoRA adapters disabled; "
            "the added base patch channels are zero initialized"
        ),
        "seed": SEED, "height": HEIGHT, "width": WIDTH, "frames": FRAMES,
        "fps": FPS, "inference_steps": STEPS, "guidance_scale": GUIDANCE,
        "flow_shift": FLOW_SHIFT, "negative_prompt": None, "lora_scale": 1.0,
        "model_dtype": "bfloat16", "inference_vae_dtype": "float32",
        "source_preprocessing": "RGB PNG; /127.5-1; bicubic 256x448 align_corners=False",
        "ssim": "11x11 Gaussian sigma=1.5, valid pixels, RGB-channel mean, range=1",
        "mae": "Mean absolute RGB error in 0-255 units",
        "primary_pass": "step800 SSIM>=0.75 and gain over untrained>=0.30 and MAE<=0.5*untrained MAE",
        "counterfactual_pass": "at least four of five B outputs have SSIM to supplied B greater than to training A",
    }


def run_generate() -> dict:
    _require_official_source(SOURCE)
    from diffusers.utils import export_to_video
    from finetrainers.patches.dependencies.diffusers.control import control_channel_concat

    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("BF16 CUDA GPU required")
    EVAL.mkdir(parents=True, exist_ok=True)
    protocol = _protocol()
    protocol_path = EVAL / "protocol.json"
    if protocol_path.exists() and json.loads(protocol_path.read_text()) != protocol:
        raise ValueError("Existing M6A-2 protocol differs")
    protocol_path.write_text(json.dumps(protocol, indent=2) + "\n")
    started = time.monotonic()
    with GPUMemorySampler() as sampler:
        pipe, loaded = _load_pipeline(SINGLE_TRAIN, CHECKPOINTS, active_step=800)
        layer = pipe.transformer.patch_embedding
        if torch.count_nonzero(layer.base_layer.weight[:, 16:]).item() != 0:
            raise RuntimeError("Untrained control patch channels are not zero initialized")
        pipe.vae.to("cuda")
        latent_frames = (FRAMES - 1) // 4 + 1
        frames = {"training_a": source_frame_rgb(protocol["training_o0"])}
        frames.update({f"counter_{i:02d}": source_frame_rgb(item["source_o0"])
                       for i, item in enumerate(protocol["counterfactuals"], 1)})
        controls = {name: _control_latent(pipe, frame, latent_frames)
                    for name, frame in frames.items()}
        pipe.vae.to("cpu")
        torch.cuda.empty_cache()
        pipe.enable_model_cpu_offload()
        fixed_latents = pipe.prepare_latents(
            1, 16, HEIGHT, WIDTH, FRAMES, torch.bfloat16, torch.device("cuda"),
            torch.Generator(device="cuda").manual_seed(SEED),
        )
        videos = []
        conditions = [("untrained_control", "training_a", None)]
        conditions += [(f"step_{step:04d}", "training_a", step) for step in CHECKPOINTS]
        conditions += [(f"counter_{index:02d}", f"counter_{index:02d}", 800)
                       for index in range(1, 6)]
        for name, state, step in conditions:
            if step is None:
                pipe.disable_lora()
                if not layer.disable_adapters:
                    raise RuntimeError("Untrained control baseline still has active LoRA weights")
            else:
                pipe.enable_lora()
                adapter = f"step_{step}"
                pipe.set_adapters([adapter], [1.0])
                if layer.disable_adapters or layer.active_adapters != [adapter]:
                    raise RuntimeError(f"Wrong active Control LoRA adapter at step {step}")
            control = controls[state].to("cuda")
            path = EVAL / "videos" / f"{name}.mp4"
            metadata_path = path.with_suffix(".json")
            metadata = {
                "condition": name, "source_state": state, "checkpoint_step": step,
                "prompt": protocol["prompt"], "seed": SEED,
                "control_latent_norm": float(control.float().norm()),
                "control_nonzero_temporal_indices": [0], "control_hook_active": True,
                "adapter_disabled": step is None,
                "video_path": str(path), "frame_count_verified": FRAMES,
            }
            path.parent.mkdir(parents=True, exist_ok=True)
            if path.exists():
                if not metadata_path.is_file() or json.loads(metadata_path.read_text()) != metadata:
                    raise ValueError(f"Existing video metadata differs: {path}")
                generated_frames_rgb(path)
            else:
                with torch.no_grad(), control_channel_concat(
                    pipe.transformer, ["hidden_states"], [control], dims=[1],
                ):
                    result = pipe(
                        prompt=protocol["prompt"], negative_prompt=None,
                        height=HEIGHT, width=WIDTH, num_frames=FRAMES,
                        num_inference_steps=STEPS, guidance_scale=GUIDANCE,
                        generator=torch.Generator(device="cuda").manual_seed(SEED),
                        latents=fixed_latents.clone(),
                    )
                if len(result.frames[0]) != FRAMES:
                    raise RuntimeError(f"Wrong generated frame count: {name}")
                export_to_video(result.frames[0], str(path), fps=FPS)
                generated_frames_rgb(path)
                metadata_path.write_text(json.dumps(metadata, indent=2) + "\n")
                del result
                torch.cuda.empty_cache()
            videos.append(metadata)
            print("GENERATED " + json.dumps(metadata), flush=True)
        result = {
            "protocol": str(protocol_path), "checkpoints_loaded": loaded,
            "videos": videos, "video_count": len(videos),
            "runtime_seconds": round(time.monotonic() - started, 2),
            "peak_gpu_vram_mib_observed": sampler.peak_mib,
        }
    (EVAL / "generation_summary.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def _contact_sheet(protocol: dict, video_rows: dict[str, dict]) -> Path:
    tile_w, tile_h, header = 224, 128, 24
    sheet = Image.new("RGB", (6 * tile_w, 8 * (tile_h + header)), "#202020")
    draw = ImageDraw.Draw(sheet)
    source_a = [source_frame_rgb(path) for path in _training_row()["source_rgb_frames"][:4]]
    trajectory = [("Real A", source_a)] + [
        (name, generated_frames_rgb(video_rows[name]["video_path"]))
        for name in ("untrained_control", "step_0100", "step_0200", "step_0400", "step_0800")
    ]
    def paste(tensor: torch.Tensor, col: int, row: int, label: str) -> None:
        array = (tensor.clamp(0, 1).permute(1, 2, 0).numpy() * 255).round().astype(np.uint8)
        image = Image.fromarray(array).resize((tile_w, tile_h))
        x, y = col * tile_w, row * (tile_h + header)
        sheet.paste(image, (x, y + header))
        draw.text((x + 3, y + 3), label, fill="white")
    for row_index, frame_index in enumerate((0, 1, 3)):
        for col, (name, frames) in enumerate(trajectory):
            paste(frames[frame_index], col, row_index, f"{name} f{frame_index}")
    trained_a = trajectory[-1][1][0]
    for offset, candidate in enumerate(protocol["counterfactuals"]):
        row = 3 + offset
        b_frame = source_frame_rgb(candidate["source_o0"])
        generated_b = generated_frames_rgb(video_rows[f"counter_{offset + 1:02d}"]["video_path"])[0]
        for col, (label, frame) in enumerate((
            ("Real A", source_a[0]), ("Trained A", trained_a),
            (f"Real B{offset + 1}", b_frame), (f"Generated B{offset + 1}", generated_b),
        )):
            paste(frame, col, row, label)
    path = EVAL / "contact_sheet.png"
    sheet.save(path)
    return path


def run_analyze() -> dict:
    protocol = json.loads((EVAL / "protocol.json").read_text())
    generation = json.loads((EVAL / "generation_summary.json").read_text())
    if generation["video_count"] != 10:
        raise ValueError("Expected five training-state and five counterfactual videos")
    video_rows = {row["condition"]: row for row in generation["videos"]}
    if len(video_rows) != 10:
        raise ValueError("Duplicate evaluation condition")
    real_a = [source_frame_rgb(path) for path in _training_row()["source_rgb_frames"][:4]]
    trajectory = []
    for name, step in (("untrained_control", None), ("step_0100", 100), ("step_0200", 200),
                       ("step_0400", 400), ("step_0800", 800)):
        frames = generated_frames_rgb(video_rows[name]["video_path"])
        ssims = [ssim_rgb(frames[index], real_a[index]) for index in range(4)]
        trajectory.append({
            "condition": name, "checkpoint_step": step,
            "frame0_ssim": ssims[0],
            "frame0_mae_rgb_0_255": mean_absolute_rgb_error(frames[0], real_a[0]),
            "ssim_frames_0_1_2_3": ssims, "early_mean_ssim": sum(ssims) / 4,
            "video_path": video_rows[name]["video_path"],
        })
    baseline, trained = trajectory[0], trajectory[-1]
    primary_pass = (
        trained["frame0_ssim"] >= 0.75
        and trained["frame0_ssim"] - baseline["frame0_ssim"] >= 0.30
        and trained["frame0_mae_rgb_0_255"] <= 0.5 * baseline["frame0_mae_rgb_0_255"]
    )
    trained_a_frame = generated_frames_rgb(video_rows["step_0800"]["video_path"])[0]
    counterfactuals = []
    for index, candidate in enumerate(protocol["counterfactuals"], 1):
        name = f"counter_{index:02d}"
        generated = generated_frames_rgb(video_rows[name]["video_path"])[0]
        b = source_frame_rgb(candidate["source_o0"])
        b_ssim = ssim_rgb(generated, b)
        a_ssim = ssim_rgb(generated, real_a[0])
        counterfactuals.append({
            "condition": name, **candidate, "generated_to_supplied_b_ssim": b_ssim,
            "generated_to_training_a_ssim": a_ssim, "success": b_ssim > a_ssim,
            "output_change_vs_training_a_mae_rgb_0_255": mean_absolute_rgb_error(
                generated, trained_a_frame),
            "video_path": video_rows[name]["video_path"],
        })
    success_count = sum(row["success"] for row in counterfactuals)
    counter_pass = success_count >= 4
    steps = [json.loads(line) for line in (SINGLE_TRAIN / "steps.jsonl").read_text().splitlines()]
    finite_training = (len(steps) == 800 and [row["step"] for row in steps] == list(range(1, 801))
                       and all(math.isfinite(row["loss"]) and math.isfinite(row["lora_grad_norm"])
                               and row["lora_grad_norm"] > 0 for row in steps))
    checkpoints_valid = (set(generation["checkpoints_loaded"]) == {str(step) for step in CHECKPOINTS}
                         and all(row["lora_layer_count"] == 241 and row["control_injection_b_norm"] > 0
                                 for row in generation["checkpoints_loaded"].values()))
    videos_valid = all(
        row["frame_count_verified"] == FRAMES and row["control_hook_active"]
        and row["control_nonzero_temporal_indices"] == [0]
        and row["control_latent_norm"] > 0 for row in generation["videos"]
    ) and video_rows["untrained_control"]["adapter_disabled"] and all(
        row["output_change_vs_training_a_mae_rgb_0_255"] > 0 for row in counterfactuals
    )
    engineering_pass = finite_training and checkpoints_valid and videos_valid
    classification = (
        "C. trained-state FAIL" if not primary_pass else
        "A. trained-state PASS + counterfactual PASS" if counter_pass else
        "B. trained-state PASS + counterfactual FAIL"
    )
    sheet = _contact_sheet(protocol, video_rows)
    result = {
        "classification": classification,
        "trajectory": trajectory, "counterfactuals": counterfactuals,
        "counterfactual_success_count": success_count,
        "criteria": {
            "trained_state_pass": primary_pass, "counterfactual_pass": counter_pass,
            "engineering_validity_pass": engineering_pass,
            "finite_training": finite_training, "checkpoints_valid": checkpoints_valid,
            "videos_and_o0_control_valid": videos_valid,
        },
        "contact_sheet": str(sheet),
        "protocol": generation["protocol"],
        "generation_summary": str(EVAL / "generation_summary.json"),
    }
    (EVAL / "metrics.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("generate", "analyze"))
    phase = parser.parse_args().phase
    result = run_generate() if phase == "generate" else run_analyze()
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
