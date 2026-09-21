"""Evaluate only the predetermined M6F-2 step-6400 adapter on the M6C held-out protocol."""

from __future__ import annotations

import argparse
import json
import shutil
import time
from pathlib import Path

import torch

from training.wan.current_frame_metrics import FRAMES, generated_frames_rgb
from training.wan.finetrainers_smoke import GPUMemorySampler
from training.wan.m6c_heldout_eval import SEEDS, THRESHOLDS, run as score_heldout
from training.wan.m6c_heldout_select import OUT as M6C
from training.wan.m6d_select_128 import ROOT, digest
from training.wan.m6e_eval_6400 import OUT as M6E
from training.wan.m6f2_contrastive_train import OUT
from training.wan.vace_action_infer import CONDITIONS, _load_pipeline
from training.wan.vace_action_token_official_infer import _generate
from training.wan.vace_action_token_prototype import ActionTokenAdapter
from training.wan.vace_hard_first_frame_smoke import (
    _conditioning_images, _save_lossless_video, _verify_pipeline_mask,
)

CHECKPOINT = OUT / "checkpoints/step_6400/adapter.pt"


def _fixed_protocol() -> tuple[dict, str]:
    training = json.loads((OUT / "train/summary.json").read_text())
    sidecar = json.loads((CHECKPOINT.parent / "adapter.json").read_text())
    checkpoint_sha = digest(CHECKPOINT)
    if (training["status"] != "complete" or training["optimizer_steps"] != 6400
            or training["official_checkpoint_step"] != 6400
            or not training["pretrained_vace_unchanged"]
            or sidecar["step"] != 6400 or sidecar["sha256"] != checkpoint_sha
            or sidecar["adapter_parameters"] != 3_684_864):
        raise RuntimeError("Predetermined M6F-2 step-6400 checkpoint is incomplete")
    baseline = json.loads((M6E / "eval/protocol.json").read_text())
    m6c = json.loads((M6C / "eval/protocol.json").read_text())
    fixed = ("model_id", "model_revision", "prompt", "negative_prompt", "conditions",
             "seeds", "resolution", "frames", "inference_steps", "guidance_scale",
             "flow_shift", "conditioning_scale", "matched_noise",
             "first_frame_replacement", "thresholds")
    if (any(baseline[key] != m6c[key] for key in fixed)
            or baseline["thresholds"] != THRESHOLDS
            or baseline["seeds"] != list(SEEDS)
            or baseline["frames"] != FRAMES
            or baseline["conditions"] != {name: list(ids) for name, ids in CONDITIONS.items()}):
        raise RuntimeError("M6C/M6E frozen held-out protocol changed")
    return baseline, checkpoint_sha


def infer() -> dict:
    baseline, checkpoint_sha = _fixed_protocol()
    source_selection = M6C / "selected_states.json"
    selected_path = OUT / "selected_states.json"
    selected_path.parent.mkdir(parents=True, exist_ok=True)
    if selected_path.exists():
        if digest(selected_path) != digest(source_selection):
            raise RuntimeError("M6F-2 selected states differ from M6C")
    else:
        shutil.copyfile(source_selection, selected_path)
    selected = json.loads(selected_path.read_text())
    if len(selected["states"]) != 4 or not selected["none_in_train_split"]:
        raise RuntimeError("The four M6C held-out states changed")
    protocol = {**baseline, "milestone": "M6F-2 step-6400 contrastive held-out evaluation",
        "selected_states_path": str(selected_path),
        "selected_states_sha256": digest(selected_path),
        "adapter_checkpoint": str(CHECKPOINT),
        "adapter_checkpoint_sha256": checkpoint_sha,
        "official_checkpoint_step": 6400,
        "training_objective": "future-only denoising MSE plus locked margin-0.01 lambda-0.25 swapped-action ranking"}
    eval_dir = OUT / "eval"
    video_dir = eval_dir / "videos"
    video_dir.mkdir(parents=True, exist_ok=True)
    protocol_path = eval_dir / "protocol.json"
    if protocol_path.exists() and json.loads(protocol_path.read_text()) != protocol:
        raise RuntimeError("Existing M6F-2 evaluation protocol differs")
    protocol_path.write_text(json.dumps(protocol, indent=2) + "\n")
    adapter = ActionTokenAdapter()
    adapter.load_state_dict(torch.load(CHECKPOINT, map_location="cpu", weights_only=True), strict=True)
    if sum(p.numel() for p in adapter.parameters()) != 3_684_864:
        raise RuntimeError("Official contrastive adapter shape changed")
    adapter.to(device="cuda", dtype=torch.bfloat16).eval().requires_grad_(False)
    started = time.monotonic()
    records = []
    with GPUMemorySampler() as sampler:
        pipe, audit = _load_pipeline()
        pipe.set_progress_bar_config(disable=True)
        for state in selected["states"]:
            if digest(Path(state["source_o0"])) != state["source_o0_sha256"]:
                raise RuntimeError(f"Held-out O0 changed: {state['state']}")
            video, masks = _conditioning_images(state)
            mask_check = _verify_pipeline_mask(pipe, video, masks)
            for seed in SEEDS:
                for condition, actions in CONDITIONS.items():
                    path = video_dir / f"state_{state['state']}_{condition}_seed_{seed}.mp4"
                    sidecar_path = path.with_suffix(".json")
                    expected = {"state": state["state"], "split": state["split"],
                        "sample_id": state["sample_id"], "episode_id": state["episode_id"],
                        "source_o0": state["source_o0"], "source_o0_sha256": state["source_o0_sha256"],
                        "condition": condition, "action_ids": list(actions), "seed": seed,
                        "prompt": protocol["prompt"], "checkpoint_sha256": checkpoint_sha,
                        "resolution": protocol["resolution"], "num_frames": FRAMES,
                        "inference_steps": protocol["inference_steps"], "video_path": str(path)}
                    if path.exists():
                        if not sidecar_path.exists():
                            raise RuntimeError(f"Existing video lacks sidecar: {path}")
                        metadata = json.loads(sidecar_path.read_text())
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
                        sidecar_path.write_text(json.dumps(metadata, indent=2) + "\n")
                    if len(generated_frames_rgb(path)) != FRAMES:
                        raise RuntimeError(f"Wrong decoded frame count: {path}")
                    records.append(metadata)
                    print("M6F2_GENERATED " + json.dumps({"state": state["state"],
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


def score() -> dict:
    if not (OUT / "eval/generation_summary.json").exists():
        raise RuntimeError("All 36 official M6F-2 videos must be generated before scoring")
    score_heldout(root=OUT)
    path = OUT / "eval/metrics.json"
    metrics = json.loads(path.read_text())
    baseline = json.loads((M6E / "eval/metrics.json").read_text())
    if (metrics["thresholds"] != baseline["thresholds"]
            or [(row["sample_id"], row["source_o0_sha256"]) for row in metrics["selected_states"]]
               != [(row["sample_id"], row["source_o0_sha256"]) for row in baseline["selected_states"]]
            or len(metrics["videos"]) != 36 or len(baseline["videos"]) != 36):
        raise RuntimeError("M6F-2 states, thresholds, or population differ from M6E")
    keys = ("turn_direction_accuracy", "median_noop_motion_ratio",
            "median_absolute_action_motion_correlation", "matched_seed_flip_successes")
    metrics["milestone"] = "M6F-2 controlled action-contrastive objective experiment"
    metrics["predetermined_official_checkpoint_step"] = 6400
    metrics["baseline_m6e_metrics_path"] = str(M6E / "eval/metrics.json")
    metrics["comparison_to_m6e"] = {key: {"baseline_m6e": baseline["aggregate"][key],
        "contrastive_m6f2": metrics["aggregate"][key],
        "difference_contrastive_minus_baseline": metrics["aggregate"][key] - baseline["aggregate"][key]}
        for key in keys}
    metrics["passed_primary_criteria"] = sum(row["pass"] for row in metrics["criteria"].values())
    path.write_text(json.dumps(metrics, indent=2, allow_nan=False) + "\n")
    a, b = baseline["aggregate"], metrics["aggregate"]
    criteria, selected, per_state = metrics["criteria"], metrics["selected_states"], metrics["per_state"]
    lines = ["# M6F-2: controlled action-contrastive held-out test", "",
        f"Predetermined step-6400 contrastive checkpoint; {metrics['passed_primary_criteria']}/4 predeclared criteria pass.", "",
        "The 128-clip subset, initial adapter, sample/timestep/noise streams, model architecture, frozen VACE, "
        "inference settings, and four held-out O0 states match M6E. Only the TRAIN objective changed. "
        "Margin 0.01 and lambda 0.25 were locked from the TRAIN-only M6F-1 audit before this run.", "",
        "## Direct held-out comparison", "",
        "| Criterion | Threshold | M6E standard objective | M6F-2 contrastive | M6F-2 result |",
        "|---|---:|---:|---:|---|",
        f"| Turn direction accuracy | ≥0.70 | {a[keys[0]]:.3f} | {b[keys[0]]:.3f} | {'PASS' if criteria['turn_direction_accuracy']['pass'] else 'FAIL'} |",
        f"| Median NOOP motion ratio | ≤0.50 | {a[keys[1]]:.3f} | {b[keys[1]]:.3f} | {'PASS' if criteria['noop_motion_suppression']['pass'] else 'FAIL'} |",
        f"| Median absolute action-motion correlation | ≥0.50 | {a[keys[2]]:.3f} | {b[keys[2]]:.3f} | {'PASS' if criteria['signed_action_motion_correlation']['pass'] else 'FAIL'} |",
        f"| Matched-seed LEFT/RIGHT flip | ≥9/12 | {a[keys[3]]}/12 | {b[keys[3]]}/12 | {'PASS' if criteria['counterfactual_flip']['pass'] else 'FAIL'} |",
        "", "## Per held-out state", "",
        "| State | Split / episode / sample | Turn direction | Median NOOP ratio | Median abs(r) | Flip |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for state in selected:
        row = per_state[state["state"]]
        lines.append(f"| {state['state']} | {state['split']} / {state['episode_id']} / {state['sample_id']} | "
            f"{row['turn_correct']}/{row['turn_count']} = {row['turn_direction_accuracy']:.3f} | "
            f"{row['median_noop_motion_ratio']:.3f} | "
            f"{row['median_absolute_action_motion_correlation']:.3f} | "
            f"{row['matched_seed_flip_successes']}/3 |")
    turn_change = b[keys[0]] - a[keys[0]]
    corr_change = b[keys[2]] - a[keys[2]]
    flip_change = b[keys[3]] - a[keys[3]]
    lines += ["", "## Interpretation", "",
        f"Matched-seed flips changed by {flip_change:+d}/12. Turn accuracy changed by {turn_change:+.3f}; "
        f"absolute action-motion correlation changed by {corr_change:+.3f}. "
        "These are the predetermined step-6400 results; no checkpoint was selected from held-out performance.", "",
        "## Protocol and artifacts", "",
        "The unchanged M6C scorer uses generated O1→O2 through O15→O16 and excludes O0→O1. "
        "All 36 videos decode to 17 frames; native VACE O0 masking and 60 action-hook calls per selected block "
        "are validated by the scorer. Each matched O0/seed group resets the CUDA noise generator identically.", "",
        f"Checkpoint: `{CHECKPOINT}`. Training log: `{OUT / 'train/steps.jsonl'}`. "
        f"Videos: `{OUT / 'eval/videos'}`. Full metrics: `{path}`. "
        f"Contact sheet: `{OUT / 'eval/contact_sheet.png'}`. "
        f"Generation summary: `{OUT / 'eval/generation_summary.json'}`.", ""]
    (OUT / "eval/report.md").write_text("\n".join(lines))
    return {"criteria": metrics["criteria"], "aggregate": metrics["aggregate"],
        "comparison_to_m6e": metrics["comparison_to_m6e"], "metrics_path": str(path)}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("infer", "score"))
    result = infer() if parser.parse_args().mode == "infer" else score()
    print(json.dumps(result if "aggregate" in result else {key: result[key] for key in (
        "video_count", "runtime_seconds", "peak_gpu_vram_mib_observed")}, indent=2))
