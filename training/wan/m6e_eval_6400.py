"""Inference-only step-6400 evaluation on the unchanged M6C held-out protocol."""

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
from training.wan.m6d_select_128 import OUT as M6D, ROOT, digest
from training.wan.vace_action_infer import CONDITIONS, _load_pipeline
from training.wan.vace_action_token_official_infer import _generate
from training.wan.vace_action_token_prototype import ActionTokenAdapter
from training.wan.vace_hard_first_frame_smoke import (
    _conditioning_images, _save_lossless_video, _verify_pipeline_mask,
)

OUT = ROOT / "data/experiments/m6e_128clip_6400_heldout"
CHECKPOINT = M6D / "checkpoints/step_6400/adapter.pt"


def _source_protocol_and_checkpoint() -> tuple[dict, str]:
    training = json.loads((M6D / "train/continuation_summary_6400.json").read_text())
    sidecar = json.loads((CHECKPOINT.parent / "adapter.json").read_text())
    if (training["status"] != "complete" or training["optimizer_steps"] != 6400
            or not training["pretrained_vace_unchanged"]
            or sidecar["step"] != 6400 or sidecar["adapter_parameters"] != 3_684_864
            or sidecar["sha256"] != digest(CHECKPOINT)):
        raise RuntimeError("Official step-6400 checkpoint is incomplete or changed")
    prior = json.loads((M6D / "eval/protocol.json").read_text())
    m6c = json.loads((M6C / "eval/protocol.json").read_text())
    fixed = ("prompt", "negative_prompt", "conditions", "seeds", "resolution",
             "frames", "inference_steps", "guidance_scale", "flow_shift",
             "conditioning_scale", "matched_noise", "first_frame_replacement", "thresholds")
    if any(prior[key] != m6c[key] for key in fixed) or prior["thresholds"] != THRESHOLDS:
        raise RuntimeError("M6C/M6D held-out inference protocols differ")
    if (prior["seeds"] != list(SEEDS) or prior["frames"] != FRAMES
            or prior["conditions"] != {key: list(value) for key, value in CONDITIONS.items()}):
        raise RuntimeError("Held-out seed, frame, or action conditions changed")
    return prior, sidecar["sha256"]


def infer() -> dict:
    prior, checkpoint_sha = _source_protocol_and_checkpoint()
    source_selection = M6C / "selected_states.json"
    selected_path = OUT / "selected_states.json"
    selected_path.parent.mkdir(parents=True, exist_ok=True)
    if selected_path.exists():
        if digest(selected_path) != digest(source_selection):
            raise RuntimeError("M6E held-out state selection differs from M6C")
    else:
        shutil.copyfile(source_selection, selected_path)
    selected = json.loads(selected_path.read_text())
    if (len(selected["states"]) != 4 or not selected["none_in_train_split"]
            or digest(selected_path) != digest(M6D / "selected_states.json")):
        raise RuntimeError("The four fixed M6C/M6D states changed")
    protocol = {**prior, "milestone": "M6E step-6400 held-out action comparison",
        "selected_states_path": str(selected_path),
        "selected_states_sha256": digest(selected_path),
        "adapter_checkpoint": str(CHECKPOINT),
        "adapter_checkpoint_sha256": checkpoint_sha,
        "official_checkpoint_step": 6400}
    eval_dir = OUT / "eval"
    video_dir = eval_dir / "videos"
    video_dir.mkdir(parents=True, exist_ok=True)
    protocol_path = eval_dir / "protocol.json"
    if protocol_path.exists() and json.loads(protocol_path.read_text()) != protocol:
        raise RuntimeError("Existing M6E evaluation protocol differs")
    protocol_path.write_text(json.dumps(protocol, indent=2) + "\n")

    weights = torch.load(CHECKPOINT, map_location="cpu", weights_only=True)
    adapter = ActionTokenAdapter()
    adapter.load_state_dict(weights, strict=True)
    if sum(p.numel() for p in adapter.parameters()) != 3_684_864:
        raise RuntimeError("Step-6400 adapter shape changed")
    adapter.to(device="cuda", dtype=torch.bfloat16).eval().requires_grad_(False)
    records = []
    started = time.monotonic()
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
                    print("M6E_GENERATED " + json.dumps({"state": state["state"],
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
        raise RuntimeError("Generate all 36 step-6400 videos before scoring")
    score_heldout(root=OUT)
    path = OUT / "eval/metrics.json"
    metrics = json.loads(path.read_text())
    baseline = [json.loads((root / "eval/metrics.json").read_text()) for root in (M6C, M6D)]
    for prior in baseline:
        if (metrics["thresholds"] != prior["thresholds"]
                or [(row["sample_id"], row["source_o0_sha256"]) for row in metrics["selected_states"]]
                   != [(row["sample_id"], row["source_o0_sha256"]) for row in prior["selected_states"]]
                or len(prior["videos"]) != 36 or len(metrics["videos"]) != 36):
            raise RuntimeError("Held-out population or thresholds differ from earlier runs")
    metrics["milestone"] = "M6E step-6400 held-out budget test"
    metrics["predetermined_official_checkpoint_step"] = 6400
    metrics["baseline_metrics_paths"] = [str(root / "eval/metrics.json") for root in (M6C, M6D)]
    aggregate = [prior["aggregate"] for prior in baseline] + [metrics["aggregate"]]
    keys = ("turn_direction_accuracy", "median_noop_motion_ratio",
            "median_absolute_action_motion_correlation", "matched_seed_flip_successes")
    metrics["direct_comparison"] = {key: [row[key] for row in aggregate] for key in keys}
    passes = sum(row["pass"] for row in metrics["criteria"].values())
    # Apply the user-defined qualitative B/C distinction to the predetermined result.
    improved = sum((aggregate[2][key] > aggregate[1][key]) if key != "median_noop_motion_ratio"
                   else (aggregate[2][key] < aggregate[1][key]) for key in keys)
    classification = ("A. BUDGET-LIMITED" if passes == 4 else
        "B. PARTIALLY BUDGET-LIMITED" if passes >= 2 and improved >= 2 else
        "C. NOT EXPLAINED BY TRAINING BUDGET")
    metrics["classification"] = classification
    metrics["passed_primary_criteria"] = passes
    metrics["improved_vs_step_2000_criteria_count"] = improved
    path.write_text(json.dumps(metrics, indent=2, allow_nan=False) + "\n")
    selected = metrics["selected_states"]
    per_state = metrics["per_state"]
    lines = ["# M6E: step-6400 held-out action budget test", "",
        f"**Classification: {classification}** ({passes}/4 predeclared criteria pass).", "",
        "Only the predetermined step-6400 adapter was evaluated. The exact M6C/M6D held-out states, "
        "conditions, seeds, matched noise, prompt, inference settings, and future-only optical-flow scorer were reused. "
        "No training or model change was made for this evaluation.", "",
        "## Direct held-out comparison", "",
        "| Criterion | Threshold | 16 clips @ 800 | 128 clips @ 2000 | 128 clips @ 6400 | Step-6400 result |",
        "|---|---:|---:|---:|---:|---|",
        f"| Turn direction accuracy | ≥0.70 | {aggregate[0][keys[0]]:.3f} | {aggregate[1][keys[0]]:.3f} | {aggregate[2][keys[0]]:.3f} | {'PASS' if metrics['criteria']['turn_direction_accuracy']['pass'] else 'FAIL'} |",
        f"| Median NOOP motion ratio | ≤0.50 | {aggregate[0][keys[1]]:.3f} | {aggregate[1][keys[1]]:.3f} | {aggregate[2][keys[1]]:.3f} | {'PASS' if metrics['criteria']['noop_motion_suppression']['pass'] else 'FAIL'} |",
        f"| Median abs action-motion correlation | ≥0.50 | {aggregate[0][keys[2]]:.3f} | {aggregate[1][keys[2]]:.3f} | {aggregate[2][keys[2]]:.3f} | {'PASS' if metrics['criteria']['signed_action_motion_correlation']['pass'] else 'FAIL'} |",
        f"| Matched-seed flip | ≥9/12 | {aggregate[0][keys[3]]}/12 | {aggregate[1][keys[3]]}/12 | {aggregate[2][keys[3]]}/12 | {'PASS' if metrics['criteria']['counterfactual_flip']['pass'] else 'FAIL'} |",
        "", "## Per held-out O0 at step 6400", "",
        "| State | Split / episode / sample | Turn direction | Median NOOP ratio | Median abs(r) | Flip |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for state in selected:
        row = per_state[state["state"]]
        lines.append(f"| {state['state']} | {state['split']} / {state['episode_id']} / {state['sample_id']} | "
            f"{row['turn_correct']}/{row['turn_count']} = {row['turn_direction_accuracy']:.3f} | "
            f"{row['median_noop_motion_ratio']:.3f} | {row['median_absolute_action_motion_correlation']:.3f} | "
            f"{row['matched_seed_flip_successes']}/3 |")
    lines += ["", "## Protocol and artifacts", "",
        "The unchanged scorer uses generated O1→O2 through O15→O16 aligned to A1…A15. "
        "The observed O0→generated O1 transition is excluded. All videos decode to 17 frames; "
        "native VACE O0 masking and action-hook calls are checked by the existing evaluator. "
        "The same seed resets the CUDA generator for each action condition within an O0/seed group.", "",
        f"Official adapter: `{CHECKPOINT}`. Protocol: `{OUT / 'eval/protocol.json'}`. "
        f"36 videos: `{OUT / 'eval/videos'}`. Metrics: `{path}`. "
        f"Contact sheet: `{OUT / 'eval/contact_sheet.png'}`. "
        f"Runtime/VRAM: `{OUT / 'eval/generation_summary.json'}`.", ""]
    (OUT / "eval/report.md").write_text("\n".join(lines))
    return {"classification": classification, "criteria": metrics["criteria"],
        "aggregate": metrics["aggregate"], "metrics_path": str(path)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("infer", "score"))
    mode = parser.parse_args().mode
    result = infer() if mode == "infer" else score()
    print(json.dumps(result if mode == "score" else {
        key: result[key] for key in ("video_count", "runtime_seconds", "peak_gpu_vram_mib_observed")}, indent=2))


if __name__ == "__main__":
    main()
