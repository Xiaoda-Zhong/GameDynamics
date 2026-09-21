"""Evaluate fixed M6D step 2000 with exact M6C metrics and direct 16-vs-128 comparison."""

from __future__ import annotations

import json

from training.wan.m6c_heldout_select import OUT as M6C
from training.wan.m6c_heldout_eval import run as evaluate_m6c_protocol
from training.wan.m6d_select_128 import OUT


def run() -> dict:
    m6c = json.loads((M6C / "eval/metrics.json").read_text())
    # Same validated flow implementation, source sign, masks, thresholds, and held-out states.
    evaluate_m6c_protocol(root=OUT)
    path = OUT / "eval/metrics.json"
    metrics = json.loads(path.read_text())
    if (metrics["thresholds"] != m6c["thresholds"]
            or [(row["sample_id"], row["source_o0_sha256"]) for row in metrics["selected_states"]]
               != [(row["sample_id"], row["source_o0_sha256"]) for row in m6c["selected_states"]]
            or len(metrics["videos"]) != 36 or len(m6c["videos"]) != 36):
        raise RuntimeError("M6D held-out states, population, or thresholds differ from M6C")
    criteria = metrics["criteria"]
    passes = sum(row["pass"] for row in criteria.values())
    classification = ("A. 128-CLIP GENERALIZATION PASS" if passes == 4 else
        "B. PARTIAL GENERALIZATION" if passes >= 2 else "C. GENERALIZATION FAIL")
    keys = ("turn_direction_accuracy", "median_noop_motion_ratio",
            "median_absolute_action_motion_correlation", "matched_seed_flip_successes")
    comparison = {key: {"m6c_16_clip": m6c["aggregate"][key],
        "m6d_128_clip": metrics["aggregate"][key],
        "difference_m6d_minus_m6c": metrics["aggregate"][key] - m6c["aggregate"][key]}
        for key in keys}
    metrics["milestone"] = "M6D 128-clip action generalization scale-up pilot"
    metrics["classification"] = classification
    metrics["passed_primary_criteria"] = passes
    metrics["comparison_to_m6c_16_clip"] = comparison
    metrics["baseline_m6c_metrics_path"] = str(M6C / "eval/metrics.json")
    metrics["training_subset_path"] = str(OUT / "subset_manifest.json")
    metrics["training_summary_path"] = str(OUT / "train/summary.json")
    metrics["predetermined_official_checkpoint_step"] = 2000
    path.write_text(json.dumps(metrics, indent=2, allow_nan=False) + "\n")
    aggregate = metrics["aggregate"]
    selected = metrics["selected_states"]
    state_rows = metrics["per_state"]
    lines = ["# M6D: 128-clip action generalization scale-up pilot", "",
        f"**Classification: {classification}** ({passes}/4 predeclared criteria passed at the predetermined step-2000 checkpoint).", "",
        "The unchanged M6B-4 cross-attention architecture was trained from a fresh seed-42 zero-initialized adapter "
        "on 128 real TRAIN clips. VACE, VAE, and text weights remained frozen; native O0 and mask behavior stayed unchanged. "
        "The official evaluation uses exactly the four M6C held-out O0s, three conditions, three seeds, "
        "generic prompt, inference settings, and future-only optical-flow implementation.", "",
        "## Direct held-out comparison", "",
        "| Criterion | Required | M6C: 16 clips | M6D: 128 clips | M6D result |",
        "|---|---:|---:|---:|---|",
        f"| Turn direction accuracy | ≥0.70 | {m6c['aggregate']['turn_direction_accuracy']:.3f} | {aggregate['turn_direction_accuracy']:.3f} | {'PASS' if criteria['turn_direction_accuracy']['pass'] else 'FAIL'} |",
        f"| Median NOOP motion ratio | ≤0.50 | {m6c['aggregate']['median_noop_motion_ratio']:.3f} | {aggregate['median_noop_motion_ratio']:.3f} | {'PASS' if criteria['noop_motion_suppression']['pass'] else 'FAIL'} |",
        f"| Median absolute action-motion correlation | ≥0.50 | {m6c['aggregate']['median_absolute_action_motion_correlation']:.3f} | {aggregate['median_absolute_action_motion_correlation']:.3f} | {'PASS' if criteria['signed_action_motion_correlation']['pass'] else 'FAIL'} |",
        f"| Matched-seed flip | ≥9/12 | {m6c['aggregate']['matched_seed_flip_successes']}/12 | {aggregate['matched_seed_flip_successes']}/12 | {'PASS' if criteria['counterfactual_flip']['pass'] else 'FAIL'} |",
        "", "## Per held-out O0", "",
        "| State | Split / episode / sample | Turn direction | Median NOOP ratio | Median abs(r) | Flip |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for state in selected:
        row = state_rows[state["state"]]
        lines.append(f"| {state['state']} | {state['split']} / {state['episode_id']} / {state['sample_id']} | "
            f"{row['turn_correct']}/{row['turn_count']} = {row['turn_direction_accuracy']:.3f} | "
            f"{row['median_noop_motion_ratio']:.3f} | {row['median_absolute_action_motion_correlation']:.3f} | "
            f"{row['matched_seed_flip_successes']}/3 |")
    lines += ["", "## Protocol and artifacts", "",
        "The M6C evaluator's `score_future` and `flip_future` functions were reused unchanged. "
        "Flow uses generated O1→O2 through O15→O16, aligned to A1…A15; O0→O1 is excluded. "
        "The fixed source sign calibration and OpenCV 4.12 Farneback settings are unchanged. "
        "The four held-out episodes remain absent from the 128-clip TRAIN subset. "
        "The step-2000 checkpoint was specified before evaluation; no checkpoint was selected using held-out scores.", "",
        f"Subset: `{OUT / 'subset_manifest.json'}`. Training log: `{OUT / 'train/steps.jsonl'}`. "
        f"Checkpoints: `{OUT / 'checkpoints'}`. Generation: `{OUT / 'eval/generation_summary.json'}`. "
        f"36 videos: `{OUT / 'eval/videos'}`. Full metrics: `{OUT / 'eval/metrics.json'}`. "
        f"Contact sheet: `{OUT / 'eval/contact_sheet.png'}`.", ""]
    (OUT / "eval/report.md").write_text("\n".join(lines))
    return {"classification": classification, "criteria": criteria,
        "comparison": comparison, "metrics_path": str(path)}


if __name__ == "__main__":
    print(json.dumps(run(), indent=2))
