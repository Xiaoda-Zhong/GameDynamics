"""M6B-4 final and checkpoint-trajectory inference with the frozen native VACE path."""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

import torch

from training.wan.current_frame_metrics import FRAMES, HEIGHT, WIDTH, generated_frames_rgb
from training.wan.finetrainers_smoke import GPUMemorySampler
from training.wan.vace_action_infer import (
    CONDITIONS, SEEDS, _common_protocol, _load_pipeline, _source_state, _sha256,
)
from training.wan.vace_action_token_official_train import OUTPUT
from training.wan.vace_action_token_prototype import ActionTokenAdapter, BLOCKS, temporal_mask
from training.wan.vace_hard_first_frame_smoke import (
    CONDITIONING_SCALE, GUIDANCE, INFERENCE_STEPS, PROMPT, _conditioning_images,
    _save_lossless_video, _verify_pipeline_mask,
)


def _load_adapter(adapter: ActionTokenAdapter, step: int) -> Path:
    path = OUTPUT / "checkpoints" / f"step_{step:04d}" / "adapter.pt"
    adapter.load_state_dict(torch.load(path, map_location="cpu", weights_only=True), strict=True)
    if sum(p.numel() for p in adapter.parameters()) != 3_684_864:
        raise RuntimeError("Adapter shape or count changed")
    adapter.to(device="cuda", dtype=torch.bfloat16).eval().requires_grad_(False)
    return path


def _generate(pipe, adapter: ActionTokenAdapter, video, masks, action_ids: tuple[int, ...], seed: int) -> tuple[list, dict]:
    actions = torch.tensor([action_ids], device="cuda", dtype=torch.long)
    action_tokens = adapter.tokens(actions)
    allow_mask = temporal_mask(torch.device("cuda"))
    counts = {index: 0 for index in BLOCKS}
    handles = []
    torch.cuda.reset_peak_memory_stats()
    started = time.monotonic()
    try:
        for index in BLOCKS:
            def hook(_module, _inputs, hidden, *, site=index):
                counts[site] += 1
                return adapter.sites[str(site)](hidden, action_tokens, allow_mask)
            handles.append(pipe.transformer.blocks[index].register_forward_hook(hook))
        with torch.inference_mode():
            result = pipe(prompt=PROMPT, negative_prompt=None, video=video, mask=masks,
                reference_images=None, conditioning_scale=CONDITIONING_SCALE,
                height=HEIGHT, width=WIDTH, num_frames=FRAMES,
                num_inference_steps=INFERENCE_STEPS, guidance_scale=GUIDANCE,
                generator=torch.Generator(device="cuda").manual_seed(seed), output_type="pil")
    finally:
        for handle in handles:
            handle.remove()
    frames = result.frames[0]
    if len(frames) != FRAMES or any(count == 0 for count in counts.values()):
        raise RuntimeError(f"Wrong frame count or missing action injection: {counts}")
    return frames, {"action_hook_calls": {str(i): counts[i] for i in BLOCKS},
        "runtime_seconds": round(time.monotonic() - started, 3),
        "peak_gpu_allocated_mib": round(torch.cuda.max_memory_allocated() / 2**20, 2),
        "post_generation_first_frame_replacement": False}


def run() -> dict:
    summary = json.loads((OUTPUT / "training_summary.json").read_text())
    if summary["optimizer_steps"] != 800 or not summary["pretrained_vace_unchanged"]:
        raise RuntimeError("Official 800-step training is incomplete")
    state = _source_state()
    video, masks = _conditioning_images(state)
    eval_dir = OUTPUT / "eval"
    videos_dir = eval_dir / "videos"
    trajectory_dir = eval_dir / "checkpoint_trajectory/videos"
    videos_dir.mkdir(parents=True, exist_ok=True)
    trajectory_dir.mkdir(parents=True, exist_ok=True)
    adapter = ActionTokenAdapter()
    adapter.to(device="cuda", dtype=torch.bfloat16).eval().requires_grad_(False)
    protocol = {**_common_protocol(state),
        "milestone": "M6B-4 action-token cross-attention official pilot",
        "architecture": "16 separate action-plus-time tokens; masked four-head cross-attention residual after main blocks 3,11,19,27",
        "trainable_adapter_parameters": 3_684_864,
        "checkpoint_steps": [100, 200, 400, 800],
        "trajectory_conditions": ["original", "reversed"], "trajectory_seed": 42,
        "native_vace_mask_behavior_unchanged": True,
        "action_injection": "Per-block cross-attention to four aligned action tokens for each future temporal latent; observed latent 0 receives no direct residual.",
    }
    for step in (100, 200, 400, 800):
        checkpoint = OUTPUT / "checkpoints" / f"step_{step:04d}" / "adapter.pt"
        protocol.setdefault("checkpoints", {})[str(step)] = {"path": str(checkpoint), "sha256": _sha256(checkpoint)}
    protocol_path = eval_dir / "protocol.json"
    if protocol_path.exists() and json.loads(protocol_path.read_text()) != protocol:
        raise RuntimeError("Existing evaluation protocol differs")
    protocol_path.write_text(json.dumps(protocol, indent=2) + "\n")

    started = time.monotonic()
    rows = []
    trajectory_rows = []
    with GPUMemorySampler() as sampler:
        pipe, audit = _load_pipeline()
        mask_check = _verify_pipeline_mask(pipe, video, masks)
        # All final conditions use the same loaded model, settings, O0, and per-seed noise.
        _load_adapter(adapter, 800)
        for condition, action_ids in CONDITIONS.items():
            for seed in SEEDS:
                path = videos_dir / f"{condition}_seed_{seed}.mp4"
                sidecar = path.with_suffix(".json")
                expected = {"condition": condition, "seed": seed, "action_ids": list(action_ids),
                    "prompt": PROMPT, "checkpoint_step": 800,
                    "checkpoint_sha256": protocol["checkpoints"]["800"]["sha256"],
                    "resolution": [WIDTH, HEIGHT], "num_frames": FRAMES,
                    "inference_steps": INFERENCE_STEPS, "video_path": str(path)}
                if path.exists():
                    if not sidecar.exists():
                        raise RuntimeError(f"Existing video lacks metadata: {path}")
                    metadata = json.loads(sidecar.read_text())
                    if any(metadata.get(key) != value for key, value in expected.items()):
                        raise RuntimeError(f"Existing video settings differ: {path}")
                else:
                    frames, run = _generate(pipe, adapter, video, masks, action_ids, seed)
                    if not audit.get("native_pixel_mask_verified"):
                        raise RuntimeError("Native O0 conditioning inactive")
                    _save_lossless_video(frames, path)
                    metadata = {**expected, **run, "conditioning_audit": dict(audit),
                        "mask_preprocessing": mask_check, "frame_count_verified": FRAMES}
                    sidecar.write_text(json.dumps(metadata, indent=2) + "\n")
                if len(generated_frames_rgb(path)) != FRAMES:
                    raise RuntimeError(f"Wrong decoded video length: {path}")
                rows.append(metadata)
                print("M6B4_GENERATED " + json.dumps({"condition": condition, "seed": seed, "runtime": metadata["runtime_seconds"]}), flush=True)
                torch.cuda.empty_cache()
        for step in (100, 200, 400, 800):
            _load_adapter(adapter, step)
            for condition in ("original", "reversed"):
                seed = 42
                path = trajectory_dir / f"step_{step:04d}_{condition}_seed_42.mp4"
                sidecar = path.with_suffix(".json")
                expected = {"checkpoint_step": step, "checkpoint_sha256": protocol["checkpoints"][str(step)]["sha256"],
                    "condition": condition, "seed": seed, "action_ids": list(CONDITIONS[condition]),
                    "prompt": PROMPT, "resolution": [WIDTH, HEIGHT], "num_frames": FRAMES,
                    "inference_steps": INFERENCE_STEPS, "video_path": str(path)}
                if path.exists():
                    metadata = json.loads(sidecar.read_text())
                    if any(metadata.get(key) != value for key, value in expected.items()):
                        raise RuntimeError(f"Existing trajectory settings differ: {path}")
                elif step == 800:
                    source = videos_dir / f"{condition}_seed_42.mp4"
                    os.link(source, path)
                    final_metadata = json.loads(source.with_suffix(".json").read_text())
                    metadata = {**expected, "reuses_identical_final_video": str(source),
                        "runtime_seconds": 0.0, "action_hook_calls": final_metadata["action_hook_calls"],
                        "conditioning_audit": final_metadata["conditioning_audit"],
                        "mask_preprocessing": mask_check, "frame_count_verified": FRAMES}
                    sidecar.write_text(json.dumps(metadata, indent=2) + "\n")
                else:
                    frames, run = _generate(pipe, adapter, video, masks, CONDITIONS[condition], seed)
                    _save_lossless_video(frames, path)
                    metadata = {**expected, **run, "conditioning_audit": dict(audit),
                        "mask_preprocessing": mask_check, "frame_count_verified": FRAMES}
                    sidecar.write_text(json.dumps(metadata, indent=2) + "\n")
                if len(generated_frames_rgb(path)) != FRAMES:
                    raise RuntimeError(f"Wrong trajectory decoded length: {path}")
                trajectory_rows.append(metadata)
                print("M6B4_TRAJECTORY " + json.dumps({"step": step, "condition": condition}), flush=True)
                torch.cuda.empty_cache()
        result = {"video_count": len(rows), "trajectory_video_count": len(trajectory_rows),
            "videos": rows, "checkpoint_trajectory_videos": trajectory_rows,
            "protocol": str(protocol_path), "runtime_seconds": round(time.monotonic() - started, 3),
            "peak_gpu_vram_mib_observed": sampler.peak_mib}
    (eval_dir / "generation_summary.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


if __name__ == "__main__":
    print(json.dumps(run(), indent=2), flush=True)
