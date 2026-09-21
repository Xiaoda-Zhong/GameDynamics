#!/usr/bin/env python3
"""M6B-1 zero-init sanity and controlled-action VACE inference.

Native first-frame video/mask preparation and all generation settings follow
M6A-4. Only the integer action IDs change through the M6B-0 adapter hook.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import time
from pathlib import Path

import numpy as np
import torch

from training.wan.current_frame_metrics import FRAMES, HEIGHT, WIDTH, generated_frames_rgb
from training.wan.finetrainers_smoke import GPUMemorySampler
from training.wan.vace_action_adapter import ACTION_TO_ID, VACEActionAdapter
from training.wan.vace_hard_first_frame_smoke import (
    CONDITIONING_SCALE,
    FLOW_SHIFT,
    FPS,
    GUIDANCE,
    INFERENCE_STEPS,
    MODEL,
    MODEL_ID,
    MODEL_REVISION,
    PROMPT,
    _conditioning_images,
    _model_ready,
    _save_lossless_video,
    _selected_states,
    _verify_pipeline_mask,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
ROOT = REPO_ROOT / "data/experiments/m6b1_vace_action_tiny"
DEFAULT_CHECKPOINT = ROOT / "checkpoints/step_0800/adapter.pt"
SOURCE_SAMPLE_ID = "my_way_home_episode_0028_clip_000011"
ORIGINAL = (4, 4, 3, 3, 3, 3, 3, 3, 3, 3, 5, 5, 5, 5, 5, 5)
REVERSE_ID = {0: 0, 1: 2, 2: 1, 3: 3, 4: 5, 5: 4}
REVERSED = tuple(REVERSE_ID[action] for action in ORIGINAL)
ALL_NOOP = (3,) * 16
CONDITIONS = {"original": ORIGINAL, "reversed": REVERSED, "all_noop": ALL_NOOP}
SEEDS = (42, 43, 44, 45, 46)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _source_state() -> dict:
    state = _selected_states()[0]
    recorded = tuple(ACTION_TO_ID[action] for action in state["raw_actions"])
    if state["sample_id"] != SOURCE_SAMPLE_ID or recorded != ORIGINAL:
        raise ValueError("M6B-1 source clip or its exact recorded actions changed")
    return state


def _assert_adapter(adapter: VACEActionAdapter, zero_init: bool) -> None:
    count = sum(parameter.numel() for parameter in adapter.parameters())
    if count != 198_336:
        raise RuntimeError(f"Expected exactly 198,336 adapter parameters, got {count}")
    if zero_init and (torch.count_nonzero(adapter.projection.weight).item() != 0
                      or torch.count_nonzero(adapter.projection.bias).item() != 0):
        raise RuntimeError("The new action projection is not exactly zero initialized")


def _load_pipeline():
    from diffusers import AutoencoderKLWan, WanVACEPipeline
    from diffusers.schedulers import UniPCMultistepScheduler

    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("A BF16-capable CUDA GPU is required")
    if not _model_ready():
        raise FileNotFoundError("Pinned Wan VACE weights are incomplete")
    vae = AutoencoderKLWan.from_pretrained(
        str(MODEL), subfolder="vae", torch_dtype=torch.float32, local_files_only=True,
    )
    pipe = WanVACEPipeline.from_pretrained(
        str(MODEL), vae=vae, torch_dtype=torch.bfloat16, local_files_only=True,
    )
    if (pipe.__class__.__name__ != "WanVACEPipeline"
            or pipe.transformer.__class__.__name__ != "WanVACETransformer3DModel"
            or hasattr(pipe.transformer, "peft_config")
            or not pipe.transformer.config.vace_layers):
        raise RuntimeError("Expected the unmodified native VACE transformer")
    for module in (pipe.transformer, pipe.vae, pipe.text_encoder):
        module.requires_grad_(False)
        module.eval()
    pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config, flow_shift=FLOW_SHIFT)
    pipe.enable_model_cpu_offload()

    original_prepare = pipe.prepare_video_latents
    audit: dict = {}

    def checked_prepare(video, mask, reference_images, generator, device):
        if (torch.count_nonzero(mask[:, :, 0]).item() != 0
                or torch.count_nonzero(mask[:, :, 1:] != 1).item() != 0):
            raise RuntimeError("The native VACE first-frame pixel mask changed")
        result = original_prepare(video, mask, reference_images, generator, device)
        norm = float(result[:, :16, 0].float().norm())
        if not math.isfinite(norm) or norm <= 0:
            raise RuntimeError("The supplied O0 has no active VACE conditioning latent")
        audit["native_pixel_mask_verified"] = True
        audit["o0_conditioning_latent_norm"] = norm
        return result

    pipe.prepare_video_latents = checked_prepare
    return pipe, audit


def _generate(pipe, audit: dict, adapter: VACEActionAdapter, video, mask,
              action_ids: tuple[int, ...], seed: int):
    audit.clear()
    actions = torch.tensor([action_ids], dtype=torch.long, device="cuda")
    hook_calls = 0

    def count_hook(_module, _inputs, _output):
        nonlocal hook_calls
        hook_calls += 1

    torch.cuda.reset_peak_memory_stats()
    started = time.monotonic()
    with torch.inference_mode(), adapter.inject(pipe.transformer, actions):
        counter = pipe.transformer.patch_embedding.register_forward_hook(count_hook)
        try:
            result = pipe(
                prompt=PROMPT, negative_prompt=None, video=video, mask=mask,
                reference_images=None, conditioning_scale=CONDITIONING_SCALE,
                height=HEIGHT, width=WIDTH, num_frames=FRAMES,
                num_inference_steps=INFERENCE_STEPS, guidance_scale=GUIDANCE,
                generator=torch.Generator(device="cuda").manual_seed(seed),
                output_type="pil",
            )
        finally:
            counter.remove()
    frames = result.frames[0]
    if len(frames) != FRAMES or hook_calls < 1 or not audit.get("native_pixel_mask_verified"):
        raise RuntimeError("Wrong output frame count, inactive adapter, or inactive O0 conditioning")
    runtime = time.monotonic() - started
    metadata = {
        "action_hook_calls": hook_calls,
        "conditioning_audit": dict(audit),
        "runtime_seconds": round(runtime, 3),
        "peak_gpu_allocated_mib": round(torch.cuda.max_memory_allocated() / 2**20, 2),
        "post_generation_first_frame_replacement": False,
    }
    return frames, metadata


def _common_protocol(state: dict) -> dict:
    return {
        "milestone": "M6B-1 explicit action adapter tiny pilot",
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "model_local_path": str(MODEL),
        "pipeline": "diffusers.WanVACEPipeline",
        "diffusers_version": importlib.metadata.version("diffusers"),
        "source_sample_id": SOURCE_SAMPLE_ID,
        "source_o0": state["source_o0"],
        "source_o0_sha256": _sha256(Path(state["source_o0"])),
        "prompt": PROMPT,
        "negative_prompt": None,
        "height": HEIGHT, "width": WIDTH, "num_frames": FRAMES, "fps": FPS,
        "inference_steps": INFERENCE_STEPS, "guidance_scale": GUIDANCE,
        "flow_shift": FLOW_SHIFT, "conditioning_scale": CONDITIONING_SCALE,
        "dtype": "BF16 transformer/text; FP32 VAE; BF16 adapter during inference",
        "memory_policy": "Diffusers model CPU offload, unchanged from M6A-4",
        "native_vace_conditioning": (
            "Exact real O0 bicubic-resized in memory to 448x256; RGB(128,128,128) "
            "placeholders for frames 1..16; 17 full-frame masks with black O0 and "
            "white frames 1..16. Native latent-mask compression is unchanged."
        ),
        "action_injection": (
            "Integer IDs are embedded and grouped as four ordered future groups; "
            "the projected bias is added only to frozen noisy-latent patch embedding "
            "output before token flattening. Latent 0 receives an exact zero vector."
        ),
        "output_policy": "No generated-frame replacement; lossless RGB H.264 MP4",
        "action_to_id": ACTION_TO_ID,
        "conditions": {name: list(ids) for name, ids in CONDITIONS.items()},
        "seeds": list(SEEDS),
    }


def run_zero_init() -> dict:
    """Generate X/Y with identical O0, seed, prompt and noise before training."""
    state = _source_state()
    adapter = VACEActionAdapter()
    _assert_adapter(adapter, zero_init=True)
    adapter.to(device="cuda", dtype=torch.bfloat16).eval()
    video, mask = _conditioning_images(state)
    output_dir = ROOT / "zero_init"
    output_dir.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    with GPUMemorySampler() as sampler:
        pipe, audit = _load_pipeline()
        mask_check = _verify_pipeline_mask(pipe, video, mask)
        outputs = {}
        for name in ("original", "all_noop"):
            frames, run = _generate(pipe, audit, adapter, video, mask, CONDITIONS[name], seed=42)
            arrays = [np.asarray(frame.convert("RGB"), dtype=np.uint8) for frame in frames]
            outputs[name] = {"arrays": arrays, "run": run}
            del frames
        differences = [np.abs(x.astype(np.int16) - y.astype(np.int16))
                       for x, y in zip(outputs["original"]["arrays"], outputs["all_noop"]["arrays"])]
        max_difference = max(int(array.max()) for array in differences)
        if max_difference != 0:
            raise AssertionError(f"Zero-initialized action conditions changed pre-encoded pixels: {max_difference}")
        checksums = {
            name: hashlib.sha256(b"".join(array.tobytes() for array in data["arrays"])).hexdigest()
            for name, data in outputs.items()
        }
        report = {
            "passed": True,
            "comparison": "pre-video-encoding RGB PIL output pixels, all 17 frames",
            "max_absolute_pixel_difference": max_difference,
            "identical_sha256": checksums["original"] == checksums["all_noop"],
            "sha256_by_condition": checksums,
            "conditions": {name: list(CONDITIONS[name]) for name in outputs},
            "seed": 42,
            "protocol": _common_protocol(state),
            "mask_preprocessing": mask_check,
            "per_generation": {name: data["run"] for name, data in outputs.items()},
            "runtime_seconds": round(time.monotonic() - started, 3),
            "peak_gpu_vram_mib_observed": sampler.peak_mib,
        }
    (output_dir / "sanity.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def run_evaluate(checkpoint: Path = DEFAULT_CHECKPOINT) -> dict:
    """Generate only missing action/seed pairs using the step-800 adapter."""
    state = _source_state()
    checkpoint = checkpoint.resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Missing adapter checkpoint: {checkpoint}")
    if not (ROOT / "zero_init/sanity.json").is_file():
        raise FileNotFoundError("Run the pretraining zero-init sanity check first")
    sanity = json.loads((ROOT / "zero_init/sanity.json").read_text(encoding="utf-8"))
    if not sanity.get("passed") or sanity.get("max_absolute_pixel_difference") != 0:
        raise RuntimeError("Pretraining zero-init sanity check did not pass")
    adapter = VACEActionAdapter()
    _assert_adapter(adapter, zero_init=True)
    state_dict = torch.load(checkpoint, map_location="cpu", weights_only=True)
    adapter.load_state_dict(state_dict, strict=True)
    _assert_adapter(adapter, zero_init=False)
    adapter.to(device="cuda", dtype=torch.bfloat16).eval()
    video, mask = _conditioning_images(state)
    eval_dir = ROOT / "eval"
    videos_dir = eval_dir / "videos"
    videos_dir.mkdir(parents=True, exist_ok=True)
    protocol = {**_common_protocol(state), "adapter_checkpoint": str(checkpoint),
                "adapter_checkpoint_sha256": _sha256(checkpoint)}
    protocol_path = eval_dir / "protocol.json"
    if protocol_path.exists() and json.loads(protocol_path.read_text()) != protocol:
        raise ValueError("Existing M6B-1 evaluation protocol differs")
    protocol_path.write_text(json.dumps(protocol, indent=2) + "\n", encoding="utf-8")

    started = time.monotonic()
    rows = []
    with GPUMemorySampler() as sampler:
        pipe, audit = _load_pipeline()
        mask_check = _verify_pipeline_mask(pipe, video, mask)
        for condition, ids in CONDITIONS.items():
            for seed in SEEDS:
                path = videos_dir / f"{condition}_seed_{seed}.mp4"
                sidecar = path.with_suffix(".json")
                expected = {
                    "condition": condition, "seed": seed, "action_ids": list(ids),
                    "prompt": PROMPT, "checkpoint": str(checkpoint),
                    "checkpoint_sha256": protocol["adapter_checkpoint_sha256"],
                    "resolution": [WIDTH, HEIGHT], "num_frames": FRAMES,
                    "inference_steps": INFERENCE_STEPS, "video_path": str(path),
                }
                if path.exists():
                    if not sidecar.is_file():
                        raise ValueError(f"Existing action video has no sidecar: {path}")
                    metadata = json.loads(sidecar.read_text(encoding="utf-8"))
                    if any(metadata.get(key) != value for key, value in expected.items()):
                        raise ValueError(f"Existing video metadata differs: {path}")
                    generated_frames_rgb(path)
                else:
                    frames, run = _generate(pipe, audit, adapter, video, mask, ids, seed)
                    _save_lossless_video(frames, path)
                    metadata = {**expected, **run, "frame_count_verified": FRAMES,
                                "mask_preprocessing": mask_check}
                    sidecar.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
                    del frames
                    torch.cuda.empty_cache()
                rows.append(metadata)
                print("M6B1_GENERATED " + json.dumps(metadata), flush=True)
        summary = {
            "protocol": str(protocol_path),
            "video_count": len(rows), "videos": rows,
            "runtime_seconds": round(time.monotonic() - started, 3),
            "peak_gpu_vram_mib_observed": sampler.peak_mib,
        }
    (eval_dir / "generation_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", required=True, choices=("zero-init", "evaluate"))
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    args = parser.parse_args()
    result = run_zero_init() if args.mode == "zero-init" else run_evaluate(args.checkpoint)
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
