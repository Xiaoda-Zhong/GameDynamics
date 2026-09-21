#!/usr/bin/env python3
"""M6A fixed-seed O0 conditioning evaluation with the pinned Wan Control LoRA."""

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
    select_distinct_pairs, source_frame_rgb, ssim_rgb,
)
from training.wan.finetrainers_current_frame_train import ROOT, SOURCE, TRAIN, _hash_manifest
from training.wan.finetrainers_smoke import GPUMemorySampler, MODEL, _require_official_source


EVAL = ROOT / "eval"
SEED = 42
STEPS = 30
GUIDANCE = 5.0
FLOW_SHIFT = 3.0
FPS = 8


def _sources() -> list[dict]:
    path = TRAIN / "derived_dataset/source_map.jsonl"
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    if len(rows) != 16 or len({row["global_clip_id"] for row in rows}) != 16:
        raise ValueError("Expected 16 unique M6A training sources")
    return rows


def _protocol() -> dict:
    summary = json.loads((TRAIN / "summary.json").read_text())
    if (summary["steps"] != 320 or summary["source_manifest_sha256"] != _hash_manifest()
            or summary["frame_conditioning_type"] != "index" or summary["frame_conditioning_index"] != 0):
        raise ValueError("M6A training summary does not match this evaluation")
    rows = _sources()
    pairs = select_distinct_pairs(rows)
    return {
        "source_manifest_sha256": _hash_manifest(),
        "checkpoint": summary["checkpoint_paths"]["320"]["path"],
        "pairs": pairs,
        "baseline_definition": "Same trained step-320 Control LoRA and prompt, with zero control latent",
        "seed": SEED, "height": HEIGHT, "width": WIDTH, "frames": FRAMES,
        "fps": FPS, "inference_steps": STEPS, "guidance_scale": GUIDANCE,
        "flow_shift": FLOW_SHIFT, "negative_prompt": None,
        "lora_scale": 1.0, "dtype": "bfloat16",
        "source_preprocessing": "RGB PNG; /127.5-1; bicubic 256x448 align_corners=False",
        "ssim": "11x11 Gaussian sigma=1.5, valid pixels, RGB-channel mean, range=1",
        "mae": "Mean absolute RGB error in 0-255 units",
    }


def _control_latent(pipe, frame: torch.Tensor, latent_frames: int) -> torch.Tensor:
    from finetrainers.trainer.control_trainer.data import apply_frame_conditioning_on_latents

    device = torch.device("cuda")
    # The pinned trainer uses O0 as the first control video frame and posterior mode.
    control_video = (frame.unsqueeze(0).unsqueeze(2) * 2.0 - 1.0).to(device, pipe.vae.dtype)
    with torch.no_grad():
        latent = pipe.vae.encode(control_video).latent_dist.mode()
        mean = torch.tensor(pipe.vae.config.latents_mean, device=device).view(1, -1, 1, 1, 1)
        inverse_std = 1.0 / torch.tensor(pipe.vae.config.latents_std, device=device).view(1, -1, 1, 1, 1)
        latent = ((latent.float() - mean) * inverse_std).to(torch.bfloat16)
        latent = apply_frame_conditioning_on_latents(
            latent, expected_num_frames=latent_frames, channel_dim=1, frame_dim=2,
            frame_conditioning_type="index", frame_conditioning_index=0,
            concatenate_mask=False,
        )
    if latent.shape != (1, 16, latent_frames, HEIGHT // 8, WIDTH // 8):
        raise ValueError(f"Unexpected conditioned latent shape: {tuple(latent.shape)}")
    if not torch.isfinite(latent.float()).all() or float(latent[:, :, 0].float().norm()) <= 0:
        raise FloatingPointError("Current-frame control latent is invalid")
    if torch.count_nonzero(latent[:, :, 1:]).item() != 0:
        raise ValueError("Future-frame information leaked into O0 control latent")
    return latent.cpu()


def _load_pipeline(
    train_root: Path = TRAIN,
    checkpoint_steps: tuple[int, ...] = (80, 160, 320),
    active_step: int = 320,
):
    from diffusers import AutoencoderKLWan, WanPipeline
    from diffusers.schedulers import UniPCMultistepScheduler
    from finetrainers.models.utils import _expand_conv3d_with_zeroed_weights
    from finetrainers.patches import load_lora_weights

    vae = AutoencoderKLWan.from_pretrained(
        str(MODEL), subfolder="vae", torch_dtype=torch.float32, local_files_only=True,
    )
    pipe = WanPipeline.from_pretrained(
        str(MODEL), vae=vae, torch_dtype=torch.bfloat16, local_files_only=True,
    )
    pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config, flow_shift=FLOW_SHIFT)
    original_channels = pipe.transformer.patch_embedding.in_channels
    if original_channels != 16:
        raise ValueError(f"Expected 16 original Wan patch channels, got {original_channels}")
    pipe.transformer.patch_embedding = _expand_conv3d_with_zeroed_weights(
        pipe.transformer.patch_embedding, new_in_channels=32,
    )
    loaded = {}
    if active_step not in checkpoint_steps:
        raise ValueError("Active Control LoRA checkpoint must be loaded")
    for step in checkpoint_steps:
        checkpoint = train_root / "lora_weights" / f"{step:06d}"
        name = f"step_{step}"
        load_lora_weights(pipe, str(checkpoint), adapter_name=name)
        layer = pipe.transformer.patch_embedding
        if name not in layer.lora_B:
            raise RuntimeError(f"Control injection LoRA missing for checkpoint {step}")
        b_norm = float(layer.lora_B[name].weight.detach().float().norm())
        if not math.isfinite(b_norm) or b_norm <= 0:
            raise FloatingPointError(f"Control injection LoRA is inactive at checkpoint {step}")
        loaded[str(step)] = {
            "checkpoint": str(checkpoint), "adapter_name": name,
            "control_injection_b_norm": b_norm,
            "lora_layer_count": sum(
                1 for module in pipe.transformer.modules() if hasattr(module, "lora_A") and name in module.lora_A
            ),
        }
        if loaded[str(step)]["lora_layer_count"] != 241:
            raise RuntimeError(f"Checkpoint {step} did not load into all 241 LoRA layers")
        print("CHECKPOINT_LOADED " + json.dumps({"step": step, **loaded[str(step)]}), flush=True)
    active_name = f"step_{active_step}"
    pipe.set_adapters([active_name], [1.0])
    if pipe.transformer.patch_embedding.active_adapters != [active_name]:
        raise RuntimeError(f"Step-{active_step} Control LoRA is not the only active adapter")
    return pipe, loaded


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
        raise ValueError("Existing M6A evaluation protocol differs")
    protocol_path.write_text(json.dumps(protocol, indent=2) + "\n")
    rows = _sources()
    started = time.monotonic()
    with GPUMemorySampler() as sampler:
        pipe, checkpoints = _load_pipeline()
        pipe.vae.to("cuda")
        latent_frames = (FRAMES - 1) // 4 + 1
        frames = {index: source_frame_rgb(rows[index]["source_rgb_frames"][0])
                  for pair in protocol["pairs"] for index in (pair["a_index"], pair["b_index"])}
        controls = {index: _control_latent(pipe, frame, latent_frames)
                    for index, frame in frames.items()}
        pipe.vae.to("cpu")
        torch.cuda.empty_cache()
        pipe.enable_model_cpu_offload()
        fixed_latents = pipe.prepare_latents(
            1, 16, HEIGHT, WIDTH, FRAMES, torch.bfloat16, torch.device("cuda"),
            torch.Generator(device="cuda").manual_seed(SEED),
        )
        videos = []
        for pair_index, pair in enumerate(protocol["pairs"], 1):
            prompt = rows[pair["a_index"]]["prompt"]
            if not prompt:
                raise ValueError("Empty scene+action evaluation prompt")
            for condition, index in (("unconditioned", None), ("conditioned_a", pair["a_index"]),
                                     ("conditioned_b", pair["b_index"])):
                control = (torch.zeros_like(controls[pair["a_index"]]) if index is None
                           else controls[index]).to("cuda")
                path = EVAL / "videos" / f"pair_{pair_index:02d}_{condition}.mp4"
                path.parent.mkdir(parents=True, exist_ok=True)
                if path.exists():
                    generated_frames_rgb(path)
                else:
                    with torch.no_grad(), control_channel_concat(
                        pipe.transformer, ["hidden_states"], [control], dims=[1],
                    ):
                        result = pipe(
                            prompt=prompt, negative_prompt=None,
                            height=HEIGHT, width=WIDTH, num_frames=FRAMES,
                            num_inference_steps=STEPS, guidance_scale=GUIDANCE,
                            generator=torch.Generator(device="cuda").manual_seed(SEED),
                            latents=fixed_latents.clone(),
                        )
                    if len(result.frames[0]) != FRAMES:
                        raise RuntimeError(f"Wrong generated frame count: {path}")
                    export_to_video(result.frames[0], str(path), fps=FPS)
                    generated_frames_rgb(path)
                    del result
                    torch.cuda.empty_cache()
                row = {
                    "pair_index": pair_index, "condition": condition,
                    "prompt": prompt, "seed": SEED, "source_index": index,
                    "source_sample_id": rows[index]["global_clip_id"] if index is not None else None,
                    "control_latent_norm": float(control.float().norm()),
                    "control_nonzero_temporal_indices": ([0] if index is not None else []),
                    "control_hook_active": True, "adapter_name": "step_320",
                    "video_path": str(path), "frame_count_verified": FRAMES,
                }
                videos.append(row)
                print("GENERATED " + json.dumps(row), flush=True)
        result = {
            "protocol": str(protocol_path), "checkpoints_loaded": checkpoints,
            "videos": videos, "video_count": len(videos),
            "runtime_seconds": round(time.monotonic() - started, 2),
            "peak_gpu_vram_mib_observed": sampler.peak_mib,
        }
    (EVAL / "generation_summary.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def _contact_sheet(protocol: dict, generation: dict, rows: list[dict]) -> Path:
    by_key = {(row["pair_index"], row["condition"]): row for row in generation["videos"]}
    thumb_w, thumb_h = 224, 128
    sheet = Image.new("RGB", (5 * thumb_w, 15 * (thumb_h + 28)), "#202020")
    draw = ImageDraw.Draw(sheet)
    for pair_index, pair in enumerate(protocol["pairs"], 1):
        candidates = [
            ("Source A", [source_frame_rgb(rows[pair["a_index"]]["source_rgb_frames"][i]) for i in (0, 1, 3)]),
            ("Control A", generated_frames_rgb(by_key[pair_index, "conditioned_a"]["video_path"])),
            ("Source B", [source_frame_rgb(rows[pair["b_index"]]["source_rgb_frames"][i]) for i in (0, 1, 3)]),
            ("Control B", generated_frames_rgb(by_key[pair_index, "conditioned_b"]["video_path"])),
            ("Zero control", generated_frames_rgb(by_key[pair_index, "unconditioned"]["video_path"])),
        ]
        for lane, (name, video) in enumerate(candidates):
            for column, time_index in enumerate((0, 1, 3)):
                y = ((pair_index - 1) * 3 + column) * (thumb_h + 28)
                tensor = video[column] if name.startswith("Source") else video[time_index]
                image = Image.fromarray((tensor.clamp(0, 1).permute(1, 2, 0).numpy() * 255).round().astype(np.uint8))
                sheet.paste(image.resize((thumb_w, thumb_h)), (lane * thumb_w, y + 23))
                draw.text((lane * thumb_w + 3, y + 3), f"Pair {pair_index} {name} f{time_index}", fill="white")
    path = EVAL / "contact_sheet.png"
    sheet.save(path)
    return path


def run_analyze() -> dict:
    protocol = json.loads((EVAL / "protocol.json").read_text())
    generation = json.loads((EVAL / "generation_summary.json").read_text())
    if generation["video_count"] != 15:
        raise ValueError("Expected all 15 M6A evaluation videos")
    rows = _sources()
    by_key = {(row["pair_index"], row["condition"]): row for row in generation["videos"]}
    per_pair = []
    for pair_index, pair in enumerate(protocol["pairs"], 1):
        source_a = [source_frame_rgb(path) for path in rows[pair["a_index"]]["source_rgb_frames"][:4]]
        source_b = source_frame_rgb(rows[pair["b_index"]]["source_rgb_frames"][0])
        generated = {condition: generated_frames_rgb(by_key[pair_index, condition]["video_path"])
                     for condition in ("unconditioned", "conditioned_a", "conditioned_b")}
        metrics = {}
        for condition in ("unconditioned", "conditioned_a"):
            frames = generated[condition]
            ssims = [ssim_rgb(frames[i], source_a[i]) for i in range(4)]
            metrics[condition] = {
                "ssim_frames_0_1_2_3": ssims,
                "first_frame_ssim": ssims[0],
                "first_frame_mae_rgb_0_255": mean_absolute_rgb_error(frames[0], source_a[0]),
                "early_frame_mean_ssim": sum(ssims) / 4,
            }
        cross = {
            "a_output_to_a_input_ssim": ssim_rgb(generated["conditioned_a"][0], source_a[0]),
            "a_output_to_b_input_ssim": ssim_rgb(generated["conditioned_a"][0], source_b),
            "b_output_to_b_input_ssim": ssim_rgb(generated["conditioned_b"][0], source_b),
            "b_output_to_a_input_ssim": ssim_rgb(generated["conditioned_b"][0], source_a[0]),
        }
        cross["success"] = (cross["a_output_to_a_input_ssim"] > cross["a_output_to_b_input_ssim"]
                            and cross["b_output_to_b_input_ssim"] > cross["b_output_to_a_input_ssim"])
        output_change = {
            "zero_vs_conditioned_a_first_frame_mae_rgb_0_255": mean_absolute_rgb_error(
                generated["unconditioned"][0], generated["conditioned_a"][0]),
            "conditioned_a_vs_b_first_frame_mae_rgb_0_255": mean_absolute_rgb_error(
                generated["conditioned_a"][0], generated["conditioned_b"][0]),
        }
        per_pair.append({"pair_index": pair_index, **pair, "identity_and_early": metrics,
                         "counterfactual": cross, "output_change": output_change})
    aggregate = {}
    for condition in ("unconditioned", "conditioned_a"):
        aggregate[condition] = {
            key: sum(item["identity_and_early"][condition][key] for item in per_pair) / 5
            for key in ("first_frame_ssim", "first_frame_mae_rgb_0_255", "early_frame_mean_ssim")
        }
    first_gain = aggregate["conditioned_a"]["first_frame_ssim"] - aggregate["unconditioned"]["first_frame_ssim"]
    early_gain = aggregate["conditioned_a"]["early_frame_mean_ssim"] - aggregate["unconditioned"]["early_frame_mean_ssim"]
    flip_count = sum(item["counterfactual"]["success"] for item in per_pair)
    train_rows = [json.loads(line) for line in (TRAIN / "steps.jsonl").read_text().splitlines()]
    training_valid = (len(train_rows) == 320 and [row["step"] for row in train_rows] == list(range(1, 321))
                      and all(math.isfinite(row["loss"]) and math.isfinite(row["lora_grad_norm"])
                              and row["lora_grad_norm"] > 0 for row in train_rows))
    generation_valid = (all(v["frame_count_verified"] == 17 and v["control_hook_active"] for v in generation["videos"])
                        and all(v["control_latent_norm"] > 0 for v in generation["videos"] if v["source_index"] is not None)
                        and all(v["control_latent_norm"] == 0 for v in generation["videos"] if v["source_index"] is None))
    generation_valid = generation_valid and all(
        all(value > 0 for value in item["output_change"].values()) for item in per_pair
    )
    checkpoints_valid = (set(generation["checkpoints_loaded"]) == {"80", "160", "320"}
                         and all(v["lora_layer_count"] == 241 and v["control_injection_b_norm"] > 0
                                 for v in generation["checkpoints_loaded"].values()))
    sheet = _contact_sheet(protocol, generation, rows)
    result = {
        "per_pair": per_pair,
        "aggregate": aggregate,
        "first_frame_ssim_improvement": first_gain,
        "early_frame_mean_ssim_improvement": early_gain,
        "counterfactual_success_count": flip_count,
        "counterfactual_pair_count": 5,
        "criteria": {
            "first_frame_identity": {"pass": aggregate["conditioned_a"]["first_frame_ssim"] >= 0.75 and first_gain >= 0.20,
                                     "required_conditioned_mean_ssim": 0.75, "required_ssim_gain": 0.20},
            "early_scene_retention": {"pass": early_gain >= 0.10, "required_ssim_gain": 0.10},
            "counterfactual_current_state": {"pass": flip_count >= 4, "required_successes": 4},
            "engineering_validity": {"pass": training_valid and generation_valid and checkpoints_valid,
                                     "finite_training": training_valid, "generation_valid": generation_valid,
                                     "checkpoints_load": checkpoints_valid},
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
