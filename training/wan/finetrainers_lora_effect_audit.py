#!/usr/bin/env python3
"""Measure the numerical inference effect of the saved step-800 Wan LoRA."""

from __future__ import annotations

import hashlib
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
from decord import VideoReader, cpu
from safetensors import safe_open

from training.wan.finetrainers_overfit import (
    FPS, FRAMES, GUIDANCE_SCALE, HEIGHT, INFERENCE_STEPS, SAMPLE_ID, SEED, WIDTH,
    _fixed_sample,
)
from training.wan.finetrainers_smoke import GPUMemorySampler, MODEL, OUTPUT


RUN_ROOT = OUTPUT / "single_sample_0800"
CHECKPOINT = RUN_ROOT / "train/lora_weights/000800"
AUDIT_ROOT = RUN_ROOT / "lora_effect_audit"
EXISTING_EVAL = RUN_ROOT / "eval"
SCALES = (0.0, 0.5, 1.0, 2.0, 4.0)
FRAME_METRIC = "decoded_rgb_mean_absolute_difference"


def _tensor_stats() -> dict:
    path = CHECKPOINT / "pytorch_lora_weights.safetensors"
    if not path.is_file():
        raise FileNotFoundError(path)
    accumulators = {
        "A": {"tensor_count": 0, "nonzero_tensors": 0, "nonzero_elements": 0,
              "element_count": 0, "abs_sum": 0.0, "squared_sum": 0.0, "max_abs": 0.0},
        "B": {"tensor_count": 0, "nonzero_tensors": 0, "nonzero_elements": 0,
              "element_count": 0, "abs_sum": 0.0, "squared_sum": 0.0, "max_abs": 0.0},
    }
    with safe_open(path, framework="pt", device="cpu") as file:
        keys = list(file.keys())
        for key in keys:
            if ".lora_A.weight" in key:
                group = "A"
            elif ".lora_B.weight" in key:
                group = "B"
            else:
                raise ValueError(f"Unexpected non-LoRA tensor in adapter: {key}")
            tensor = file.get_tensor(key).to(torch.float64)
            magnitude = tensor.abs()
            accumulator = accumulators[group]
            nonzero = int(torch.count_nonzero(tensor))
            accumulator["tensor_count"] += 1
            accumulator["nonzero_tensors"] += int(nonzero > 0)
            accumulator["nonzero_elements"] += nonzero
            accumulator["element_count"] += tensor.numel()
            accumulator["abs_sum"] += magnitude.sum().item()
            accumulator["squared_sum"] += tensor.square().sum().item()
            accumulator["max_abs"] = max(accumulator["max_abs"], magnitude.max().item())
    if len(keys) != 480 or any(value["tensor_count"] != 240 for value in accumulators.values()):
        raise ValueError("Step-800 checkpoint does not contain the expected 240 A/B tensor pairs")
    result = {}
    for group, accumulator in accumulators.items():
        result[group] = {
            "tensor_count": accumulator["tensor_count"],
            "nonzero_tensors": accumulator["nonzero_tensors"],
            "nonzero_elements": accumulator["nonzero_elements"],
            "element_count": accumulator["element_count"],
            "mean_absolute_value": accumulator["abs_sum"] / accumulator["element_count"],
            "max_absolute_value": accumulator["max_abs"],
            "global_l2_norm": math.sqrt(accumulator["squared_sum"]),
        }
    return {"checkpoint": str(path), "tensor_count": len(keys), "groups": result}


def _tensor_difference(base: torch.Tensor, changed: torch.Tensor) -> dict:
    if base.shape != changed.shape:
        raise ValueError(f"Tensor shapes differ: {base.shape} versus {changed.shape}")
    a = base.to(torch.float64).flatten()
    b = changed.to(torch.float64).flatten()
    difference = b - a
    absolute_l2 = torch.linalg.vector_norm(difference).item()
    base_l2 = torch.linalg.vector_norm(a).item()
    changed_l2 = torch.linalg.vector_norm(b).item()
    if base_l2 == 0 or changed_l2 == 0:
        raise ValueError("Transformer prediction has zero norm")
    result = {
        "shape": list(base.shape),
        "base_l2_norm": base_l2,
        "changed_l2_norm": changed_l2,
        "absolute_l2_difference": absolute_l2,
        "relative_l2_difference": absolute_l2 / base_l2,
        "cosine_similarity": torch.dot(a, b).item() / (base_l2 * changed_l2),
        "max_absolute_element_difference": difference.abs().max().item(),
    }
    if not all(math.isfinite(value) for value in result.values() if isinstance(value, float)):
        raise FloatingPointError("Nonfinite transformer comparison metric")
    return result


def _fingerprint(tensor: torch.Tensor) -> dict:
    contiguous = tensor.detach().cpu().contiguous()
    return {
        "shape": list(contiguous.shape),
        "dtype": str(contiguous.dtype),
        "sha256": hashlib.sha256(contiguous.view(torch.uint8).numpy().tobytes()).hexdigest(),
    }


def _frame_difference(reference: Path, candidate: Path) -> dict:
    first = VideoReader(str(reference), ctx=cpu(0))
    second = VideoReader(str(candidate), ctx=cpu(0))
    if len(first) != FRAMES or len(second) != FRAMES:
        raise ValueError(f"Video must have {FRAMES} frames: {reference}, {candidate}")
    pixel_sum = 0
    element_count = 0
    max_difference = 0
    for index in range(FRAMES):
        a = first[index].asnumpy().astype(np.int16)
        b = second[index].asnumpy().astype(np.int16)
        if a.shape != (HEIGHT, WIDTH, 3) or b.shape != a.shape:
            raise ValueError(f"Video frame dimensions differ at frame {index}")
        difference = np.abs(a - b)
        pixel_sum += int(difference.sum(dtype=np.int64))
        element_count += difference.size
        max_difference = max(max_difference, int(difference.max()))
    mean = pixel_sum / element_count
    return {
        "metric": FRAME_METRIC,
        "frames": FRAMES,
        "mean_absolute_rgb_difference_0_255": mean,
        "mean_absolute_rgb_difference_0_1": mean / 255.0,
        "max_absolute_channel_difference_0_255": max_difference,
    }


def _captured_input(kwargs: dict, output: object) -> dict:
    if not isinstance(output, (tuple, list)) or len(output) != 1:
        raise TypeError("Expected Wan transformer return_dict=False output tuple")
    names = ("hidden_states", "timestep", "encoder_hidden_states")
    return {
        **{name: kwargs[name].detach().cpu().clone() for name in names},
        "prediction": output[0].detach().cpu().clone(),
    }


def _same_inputs(reference: dict, candidate: dict) -> bool:
    return all(
        torch.equal(reference[name], candidate[name])
        for name in ("hidden_states", "timestep", "encoder_hidden_states")
    )


def run_audit() -> dict:
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1 or not torch.cuda.is_bf16_supported():
        raise RuntimeError("Exactly one BF16-capable CUDA GPU is required")
    sample = _fixed_sample()
    if sample["global_clip_id"] != SAMPLE_ID:
        raise ValueError("Unexpected training clip")
    if AUDIT_ROOT.exists() and any(AUDIT_ROOT.glob("scale_*.mp4")):
        raise FileExistsError(f"Existing scale videos would be overwritten: {AUDIT_ROOT}")
    AUDIT_ROOT.mkdir(parents=True, exist_ok=True)
    stats = _tensor_stats()
    (AUDIT_ROOT / "adapter_tensor_stats.json").write_text(json.dumps(stats, indent=2) + "\n")

    from diffusers import AutoencoderKLWan, WanPipeline
    from diffusers.schedulers import UniPCMultistepScheduler
    from diffusers.utils import export_to_video

    started = time.monotonic()
    vae = AutoencoderKLWan.from_pretrained(
        str(MODEL), subfolder="vae", torch_dtype=torch.float32, local_files_only=True,
    )
    pipeline = WanPipeline.from_pretrained(
        str(MODEL), vae=vae, torch_dtype=torch.bfloat16, local_files_only=True,
    )
    pipeline.scheduler = UniPCMultistepScheduler.from_config(
        pipeline.scheduler.config, flow_shift=3.0,
    )
    pipeline.load_lora_weights(str(CHECKPOINT), adapter_name="step_800_audit")
    if pipeline.get_list_adapters().get("transformer") != ["step_800_audit"]:
        raise RuntimeError("Step-800 LoRA was not installed on the Wan transformer")
    pipeline.enable_model_cpu_offload()

    captures: dict[str, dict] = {}
    current_scale: str | None = None

    def capture_first_conditional(_module: torch.nn.Module, _args: tuple, kwargs: dict,
                                  output: object) -> None:
        if current_scale is not None and current_scale not in captures:
            captures[current_scale] = _captured_input(kwargs, output)

    handle = pipeline.transformer.register_forward_hook(capture_first_conditional, with_kwargs=True)
    videos: dict[str, str] = {}
    durations: dict[str, float] = {}
    base_prediction: torch.Tensor | None = None
    with GPUMemorySampler() as sampler:
        for scale in SCALES:
            key = f"{scale:g}"
            pipeline.enable_lora()
            pipeline.set_adapters(["step_800_audit"], adapter_weights=[scale])
            current_scale = key
            generated_started = time.monotonic()
            result = pipeline(
                prompt=sample["prompt"], negative_prompt=None,
                height=HEIGHT, width=WIDTH, num_frames=FRAMES,
                num_inference_steps=INFERENCE_STEPS, guidance_scale=GUIDANCE_SCALE,
                generator=torch.Generator(device="cuda").manual_seed(SEED),
            )
            current_scale = None
            if key not in captures or len(result.frames[0]) != FRAMES:
                raise RuntimeError(f"Scale {key} did not capture a valid 17-frame generation")
            video_path = AUDIT_ROOT / f"scale_{key.replace('.', 'p')}.mp4"
            export_to_video(result.frames[0], str(video_path), fps=FPS)
            video = VideoReader(str(video_path), ctx=cpu(0))
            if len(video) != FRAMES or tuple(video[0].shape) != (HEIGHT, WIDTH, 3):
                raise RuntimeError(f"Malformed generated video: {video_path}")
            videos[key] = str(video_path)
            durations[key] = round(time.monotonic() - generated_started, 2)
            print(f"AUDIT_VIDEO scale={key} path={video_path} seconds={durations[key]:.2f}", flush=True)
            del result, video

            if scale == 0.0:
                # Official disable_lora() supplies the actual base-model comparison.
                # It runs on the exact first-step latent, timestep, and text embedding
                # captured from the scale-zero pipeline call.
                pipeline.disable_lora()
                fixed = captures[key]
                with torch.no_grad(), pipeline.transformer.cache_context("cond"):
                    base_prediction = pipeline.transformer(
                        hidden_states=fixed["hidden_states"].to("cuda"),
                        timestep=fixed["timestep"].to("cuda"),
                        encoder_hidden_states=fixed["encoder_hidden_states"].to("cuda"),
                        attention_kwargs=None, return_dict=False,
                    )[0].detach().cpu()
                pipeline.enable_lora()

    handle.remove()
    if base_prediction is None:
        raise RuntimeError("Base-only transformer prediction was not computed")
    reference = captures["0"]
    inputs_equal = {key: _same_inputs(reference, capture) for key, capture in captures.items()}
    if not all(inputs_equal.values()):
        raise RuntimeError(f"Scale runs did not use identical first-step inputs: {inputs_equal}")
    fixed_input = {
        name: _fingerprint(reference[name])
        for name in ("hidden_states", "timestep", "encoder_hidden_states")
    }
    prediction = {
        "conditioning": "first conditional Wan denoising call",
        "fixed_input": fixed_input,
        "inputs_equal_across_scales": inputs_equal,
        "base_only_vs_scale_0": _tensor_difference(base_prediction, reference["prediction"]),
        "base_only_vs_scale_1": _tensor_difference(base_prediction, captures["1"]["prediction"]),
        "base_only_vs_other_scales": {
            key: _tensor_difference(base_prediction, captures[key]["prediction"])
            for key in ("0.5", "2", "4")
        },
    }
    video_reference = Path(videos["0"])
    video_metrics = {
        key: _frame_difference(video_reference, Path(videos[key])) for key in videos
    }
    existing = {
        "0": EXISTING_EVAL / "step_000_base.mp4",
        "100": EXISTING_EVAL / "step_100.mp4",
        "200": EXISTING_EVAL / "step_200.mp4",
        "400": EXISTING_EVAL / "step_400.mp4",
        "800": EXISTING_EVAL / "step_800.mp4",
    }
    existing_metrics = {
        key: _frame_difference(existing["0"], path) for key, path in existing.items()
    }
    result = {
        "model": str(MODEL),
        "checkpoint": str(CHECKPOINT),
        "training_sample_id": SAMPLE_ID,
        "prompt": sample["prompt"],
        "seed": SEED,
        "frames": FRAMES,
        "width": WIDTH,
        "height": HEIGHT,
        "inference_steps": INFERENCE_STEPS,
        "fps": FPS,
        "guidance_scale": GUIDANCE_SCALE,
        "flow_shift": 3.0,
        "adapter_scale_mechanism": "WanPipeline.set_adapters([name], adapter_weights=[scale]); base-only direct pass uses disable_lora()",
        "adapter_tensors": stats,
        "transformer_prediction": prediction,
        "scale_videos": videos,
        "scale_video_runtime_seconds": durations,
        "scale_video_difference_from_scale_0": video_metrics,
        "scale_0_vs_existing_step_0": _frame_difference(existing["0"], video_reference),
        "scale_1_vs_existing_step_800": _frame_difference(existing["800"], Path(videos["1"])),
        "existing_videos": {key: str(path) for key, path in existing.items()},
        "existing_video_difference_from_step_0": existing_metrics,
        "runtime_seconds": round(time.monotonic() - started, 2),
        "peak_gpu_vram_mib_observed": sampler.peak_mib,
    }
    (AUDIT_ROOT / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def main() -> None:
    print(json.dumps(run_audit(), indent=2))


if __name__ == "__main__":
    main()
