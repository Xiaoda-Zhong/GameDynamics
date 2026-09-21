"""M6B-4: reuse exact M5C-5/M6B-1 future-only optical-flow scoring."""

from __future__ import annotations

import json
from pathlib import Path

from training.wan.action_dynamics_metrics import calibrate_source, measure_video
from training.wan.vace_action_eval import (
    ACTION_IDS, CONDITIONS, SEEDS, SOURCE_VIDEO, _conditions, _source,
    analyze, flip_future, score_future,
)
from training.wan.vace_action_token_official_train import OUTPUT


def run() -> dict:
    eval_dir = OUTPUT / "eval"
    generation = json.loads((eval_dir / "generation_summary.json").read_text())
    if generation["video_count"] != 15 or generation["trajectory_video_count"] != 8:
        raise RuntimeError("Required final or trajectory videos are missing")
    # analyze() is the unchanged M6B-1 implementation of the validated M5C-5 flow procedure.
    metrics = analyze(eval_dir)
    passes = sum(bool(value["pass"]) for value in metrics["criteria"].values())
    classification = (
        "A. CROSS-ATTENTION ACTION CONTROL PASS" if passes == 4 else
        "B. PARTIAL ACTION CONTROL" if passes >= 2 else
        "C. CROSS-ATTENTION ACTION CONTROL FAIL"
    )
    metrics["milestone"] = "M6B-4 cross-attention action-token pilot"
    metrics["classification"] = classification
    metrics["passed_primary_criteria"] = passes
    metrics["training_summary_path"] = str(OUTPUT / "training_summary.json")
    metrics["generation_summary_path"] = str(eval_dir / "generation_summary.json")
    metrics["checkpoint_trajectory_metrics_path"] = str(eval_dir / "checkpoint_trajectory/metrics.json")
    metrics["reuse_of_validated_evaluator"] = "training.wan.vace_action_eval.analyze unchanged"

    source = _source()
    calibration = calibrate_source(measure_video(SOURCE_VIDEO), source["raw_actions"])
    sign = calibration["left_positive_multiplier"]
    action_labels = _conditions()
    trajectory = []
    for step in (100, 200, 400, 800):
        step_records = {}
        for condition in ("original", "reversed"):
            path = eval_dir / "checkpoint_trajectory/videos" / f"step_{step:04d}_{condition}_seed_42.mp4"
            measured = measure_video(path)
            scored = score_future(measured, action_labels[condition], sign)
            step_records[condition] = {"step": step, "condition": condition, "seed": 42,
                "video_path": str(path), "action_ids": ACTION_IDS[condition], **scored}
        flip = flip_future(step_records["original"], step_records["reversed"])
        trajectory.append({"checkpoint_step": step,
            "original": step_records["original"], "reversed": step_records["reversed"],
            "combined_turn_direction_accuracy":
                (step_records["original"]["turn_correct"] + step_records["reversed"]["turn_correct"])
                / (step_records["original"]["turn_count"] + step_records["reversed"]["turn_count"]),
            "mean_absolute_action_motion_correlation":
                sum((step_records[name]["action_motion_r_absolute"] or 0) for name in ("original", "reversed")) / 2,
            "counterfactual_flip": flip})
    trajectory_result = {"purpose": "descriptive seed-42 ORIGINAL versus REVERSED action-control trajectory",
        "future_only_flow_procedure": "unchanged M5C-5/M6B-1 measure_video and score_future",
        "source_flow_sign": sign, "checkpoints": trajectory}
    trajectory_path = eval_dir / "checkpoint_trajectory/metrics.json"
    trajectory_path.write_text(json.dumps(trajectory_result, indent=2, allow_nan=False) + "\n")
    (eval_dir / "metrics.json").write_text(json.dumps(metrics, indent=2, allow_nan=False) + "\n")
    criteria = metrics["criteria"]
    lines = ["# M6B-4: cross-attention action-token pilot", "",
        f"**Classification: {classification}** ({passes}/4 primary criteria passed).", "",
        "The official run resumed exactly from the validated step-32 checkpoint and ended at step 800. "
        "O0 remained native VACE conditioning; all evaluation criteria use generated future transitions only. "
        "The action-token architecture, LR, data, text, mask behavior, and frozen pretrained backbone were unchanged.", "",
        "## Primary decision", "",
        "| Criterion | Threshold | Observed | Result |", "|---|---:|---:|---|",
        f"| Turn direction accuracy | ≥0.70 | {metrics['aggregate_turn_correct']}/{metrics['aggregate_turn_total']} = {metrics['aggregate_turn_direction_accuracy']:.3f} | {'PASS' if criteria['turn_direction_accuracy']['pass'] else 'FAIL'} |",
        f"| Median NOOP motion ratio | ≤0.50 | {metrics['median_r_noop_within_turn_containing_videos']:.3f} | {'PASS' if criteria['noop_motion_suppression']['pass'] else 'FAIL'} |",
        f"| Median absolute action-motion correlation | ≥0.50 | {metrics['median_absolute_action_motion_correlation']:.3f} | {'PASS' if criteria['signed_action_motion_correlation']['pass'] else 'FAIL'} |",
        f"| Matched-seed LEFT/RIGHT flip | ≥4/5 | {metrics['counterfactual_flip_successes']}/5 | {'PASS' if criteria['counterfactual_flip']['pass'] else 'FAIL'} |",
        "", "The primary flow procedure is exactly `training.wan.vace_action_eval.analyze`, which reuses the validated "
        "M5C-5 Farneback ROI, source sign calibration, and O1→O2 through O15→O16 action alignment. "
        "O0→O1 is descriptive only. No criterion was changed after observing outputs.", "",
        "## Seed-42 checkpoint trajectory", "",
        "| Step | Turn accuracy | Mean abs(r), two runs | Expected first/final flip |",
        "|---:|---:|---:|---|",
    ]
    for row in trajectory:
        lines.append(f"| {row['checkpoint_step']} | {row['combined_turn_direction_accuracy']:.3f} | "
            f"{row['mean_absolute_action_motion_correlation']:.3f} | "
            f"{'Yes' if row['counterfactual_flip']['success'] else 'No'} |")
    lines += ["", "## Artifacts", "",
        f"Training summary: `{OUTPUT / 'training_summary.json'}`; full 800-step log: `{OUTPUT / 'steps.jsonl'}`. "
        f"Final evaluation: `{eval_dir / 'metrics.json'}`; 15 videos: `{eval_dir / 'videos'}`. "
        f"Trajectory: `{trajectory_path}` and `{eval_dir / 'checkpoint_trajectory/videos'}`. "
        f"Contact sheet: `{eval_dir / 'contact_sheet.png'}`. "
        "Metrics JSON lists all final video paths and hashes.", ""]
    (eval_dir / "report.md").write_text("\n".join(lines))
    return {"classification": classification, "passes": passes,
        "criteria": criteria, "metrics_path": str(eval_dir / "metrics.json"),
        "trajectory_metrics_path": str(trajectory_path)}


if __name__ == "__main__":
    print(json.dumps(run(), indent=2))
