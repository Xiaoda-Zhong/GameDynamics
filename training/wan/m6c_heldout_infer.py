"""M6C inference-only held-out state/action generalization test (36 videos)."""

from __future__ import annotations

import json
import time
from pathlib import Path

import torch

from training.wan.current_frame_metrics import FRAMES, HEIGHT, WIDTH, generated_frames_rgb
from training.wan.finetrainers_smoke import GPUMemorySampler
from training.wan.m6c_heldout_select import OUT, sha256
from training.wan.vace_action_infer import CONDITIONS, _load_pipeline
from training.wan.vace_action_token_official_infer import _generate, _load_adapter
from training.wan.vace_action_token_prototype import ActionTokenAdapter
from training.wan.vace_action_token_official_train import OUTPUT as M6B4
from training.wan.vace_hard_first_frame_smoke import (
    CONDITIONING_SCALE, FLOW_SHIFT, GUIDANCE, INFERENCE_STEPS, MODEL_ID,
    MODEL_REVISION, PROMPT, _conditioning_images, _save_lossless_video,
    _verify_pipeline_mask,
)

SEEDS = (42, 43, 44)


def run() -> dict:
    selected_path = OUT / "selected_states.json"
    selected = json.loads(selected_path.read_text())
    states = selected["states"]
    if (len(states) != 4 or [row["state"] for row in states] != list("ABCD")
            or not selected["all_episodes_distinct"] or not selected["none_in_training_16"]
            or not selected["none_in_train_split"]):
        raise RuntimeError("Held-out state selection invalid")
    for row in states:
        if sha256(Path(row["source_o0"])) != row["source_o0_sha256"]:
            raise RuntimeError(f"Held-out O0 changed: {row['state']}")
    checkpoint = M6B4 / "checkpoints/step_0800/adapter.pt"
    checkpoint_sha = sha256(checkpoint)
    training = json.loads((M6B4 / "training_summary.json").read_text())
    if training["optimizer_steps"] != 800 or not training["pretrained_vace_unchanged"]:
        raise RuntimeError("M6B-4 official frozen adapter checkpoint is invalid")
    eval_dir = OUT / "eval"
    videos_dir = eval_dir / "videos"
    videos_dir.mkdir(parents=True, exist_ok=True)
    protocol = {
        "milestone": "M6C held-out action generalization", "training": False,
        "selected_states_path": str(selected_path), "selected_states_sha256": sha256(selected_path),
        "model_id": MODEL_ID, "model_revision": MODEL_REVISION,
        "adapter_checkpoint": str(checkpoint), "adapter_checkpoint_sha256": checkpoint_sha,
        "adapter_trainable_parameter_count": 3_684_864,
        "prompt": PROMPT, "negative_prompt": None,
        "conditions": {name: list(ids) for name, ids in CONDITIONS.items()},
        "seeds": list(SEEDS), "resolution": [WIDTH, HEIGHT], "frames": FRAMES,
        "inference_steps": INFERENCE_STEPS, "guidance_scale": GUIDANCE,
        "flow_shift": FLOW_SHIFT, "conditioning_scale": CONDITIONING_SCALE,
        "native_vace_o0_preprocessing_and_mask_unchanged": True,
        "matched_noise": "Fresh torch.Generator(device='cuda').manual_seed(seed) on each call; same O0 and seed across conditions",
        "first_frame_replacement": False,
        "thresholds": {"turn_direction_accuracy_min": 0.70,
            "median_noop_ratio_max": 0.50, "median_absolute_correlation_min": 0.50,
            "matched_pair_flip_min_of_12": 9},
    }
    protocol_path = eval_dir / "protocol.json"
    if protocol_path.exists() and json.loads(protocol_path.read_text()) != protocol:
        raise RuntimeError("Existing M6C protocol differs")
    protocol_path.write_text(json.dumps(protocol, indent=2) + "\n")
    adapter = ActionTokenAdapter().to(device="cuda", dtype=torch.bfloat16).eval().requires_grad_(False)
    _load_adapter(adapter, 800)
    started = time.monotonic()
    records = []
    with GPUMemorySampler() as sampler:
        pipe, audit = _load_pipeline()
        pipe.set_progress_bar_config(disable=True)
        for state in states:
            video, masks = _conditioning_images(state)
            mask_check = _verify_pipeline_mask(pipe, video, masks)
            for seed in SEEDS:
                for condition, actions in CONDITIONS.items():
                    path = videos_dir / f"state_{state['state']}_{condition}_seed_{seed}.mp4"
                    sidecar = path.with_suffix(".json")
                    expected = {"state": state["state"], "split": state["split"],
                        "sample_id": state["sample_id"], "episode_id": state["episode_id"],
                        "source_o0": state["source_o0"], "source_o0_sha256": state["source_o0_sha256"],
                        "condition": condition, "action_ids": list(actions), "seed": seed,
                        "prompt": PROMPT, "checkpoint_sha256": checkpoint_sha,
                        "resolution": [WIDTH, HEIGHT], "num_frames": FRAMES,
                        "inference_steps": INFERENCE_STEPS, "video_path": str(path)}
                    if path.exists():
                        if not sidecar.exists():
                            raise RuntimeError(f"Existing video lacks sidecar: {path}")
                        metadata = json.loads(sidecar.read_text())
                        if any(metadata.get(key) != value for key, value in expected.items()):
                            raise RuntimeError(f"Existing video settings differ: {path}")
                    else:
                        audit.clear()
                        frames, run_info = _generate(pipe, adapter, video, masks, actions, seed)
                        if not audit.get("native_pixel_mask_verified"):
                            raise RuntimeError(f"Inactive native O0 conditioning: {path}")
                        _save_lossless_video(frames, path)
                        metadata = {**expected, **run_info, "conditioning_audit": dict(audit),
                            "mask_preprocessing": mask_check, "frame_count_verified": FRAMES}
                        sidecar.write_text(json.dumps(metadata, indent=2) + "\n")
                    if len(generated_frames_rgb(path)) != FRAMES:
                        raise RuntimeError(f"Wrong decoded frame count: {path}")
                    records.append(metadata)
                    print("M6C_GENERATED " + json.dumps({"state": state["state"],
                        "condition": condition, "seed": seed,
                        "runtime_seconds": metadata["runtime_seconds"]}), flush=True)
                    torch.cuda.empty_cache()
    if len(records) != 36:
        raise RuntimeError(f"Expected 36 videos, found {len(records)}")
    result = {"video_count": len(records), "videos": records,
        "protocol": str(protocol_path), "runtime_seconds": round(time.monotonic() - started, 3),
        "peak_gpu_vram_mib_observed": sampler.peak_mib}
    (eval_dir / "generation_summary.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


if __name__ == "__main__":
    output = run()
    print(json.dumps({key: output[key] for key in ("video_count", "runtime_seconds", "peak_gpu_vram_mib_observed")}, indent=2))
