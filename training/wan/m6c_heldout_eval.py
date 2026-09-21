"""M6C held-out future-only optical-flow evaluation with fixed M6B-4 formulas."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from decord import VideoReader, cpu
from PIL import Image, ImageDraw

from training.wan.action_dynamics_metrics import FLOW_PARAMETERS, calibrate_source, measure_video
from training.wan.m6c_heldout_select import OUT, sha256
from training.wan.vace_action_eval import SOURCE_VIDEO, _conditions, flip_future, score_future
from training.wan.vace_action_infer import CONDITIONS

SEEDS = (42, 43, 44)
THRESHOLDS = {"turn_direction_accuracy_min": 0.70, "median_noop_ratio_max": 0.50,
    "median_absolute_correlation_min": 0.50, "matched_pair_flip_min_of_12": 9}


def median(values: list[float | None]) -> float | None:
    present = [value for value in values if value is not None]
    return float(np.median(present)) if present else None


def summary(rows: list[dict], flips: list[dict]) -> dict:
    mixed = [row for row in rows if row["condition"] in ("original", "reversed")]
    correct = sum(row["turn_correct"] for row in mixed)
    total = sum(row["turn_count"] for row in mixed)
    return {"turn_correct": correct, "turn_count": total,
        "turn_direction_accuracy": correct / total,
        "median_noop_motion_ratio": median([row["r_noop_within_video"] for row in mixed]),
        "median_absolute_action_motion_correlation": median([
            row["action_motion_r_absolute"] if row["action_motion_r_absolute"] is not None else 0.0
            for row in mixed]),
        "matched_seed_flip_successes": sum(bool(row["success"]) for row in flips),
        "matched_seed_flip_total": len(flips),
        "median_all_noop_motion_over_matched_turn_motion": median([
            row.get("r_noop_matched_seed_turn_baseline") for row in rows if row["condition"] == "all_noop"])}


def contact_sheet(rows: list[dict], selected: dict, path: Path) -> None:
    # Four states × three seeds × three action conditions, one row per video.
    positions = (1, 4, 8, 16)
    thumb_w, thumb_h, label_w, label_h = 224, 128, 190, 23
    sheet = Image.new("RGB", (label_w + 5 * thumb_w, len(rows) * (thumb_h + label_h)), "#202020")
    draw = ImageDraw.Draw(sheet)
    sources = {}
    for state in selected["states"]:
        with Image.open(state["source_o0"]) as image:
            sources[state["state"]] = image.convert("RGB").resize((thumb_w, thumb_h), Image.Resampling.BICUBIC)
    for row_index, record in enumerate(rows):
        top = row_index * (thumb_h + label_h)
        draw.text((5, top + 5), f"{record['state']} {record['condition']} {record['seed']}", fill="white")
        sheet.paste(sources[record["state"]], (label_w, top + label_h))
        draw.text((label_w + 4, top + 4), "real O0", fill="white")
        reader = VideoReader(record["video_path"], ctx=cpu(0))
        if len(reader) != 17:
            raise RuntimeError(f"Wrong frame count: {record['video_path']}")
        for col, frame_index in enumerate(positions, start=1):
            image = Image.fromarray(reader[frame_index].asnumpy(), "RGB")
            if image.size != (448, 256):
                raise RuntimeError(f"Wrong video resolution: {record['video_path']}")
            image = image.resize((thumb_w, thumb_h), Image.Resampling.BICUBIC)
            sheet.paste(image, (label_w + col * thumb_w, top + label_h))
            draw.text((label_w + col * thumb_w + 4, top + 4), f"O{frame_index}", fill="white")
    sheet.save(path)


def run(root: Path = OUT) -> dict:
    import cv2

    if cv2.__version__ != "4.12.0":
        raise RuntimeError(f"Validated M5C-5 OpenCV 4.12.0 required, found {cv2.__version__}")
    selected = json.loads((root / "selected_states.json").read_text())
    protocol = json.loads((root / "eval/protocol.json").read_text())
    generation = json.loads((root / "eval/generation_summary.json").read_text())
    if (generation["video_count"] != 36 or protocol["thresholds"] != THRESHOLDS
            or protocol["selected_states_sha256"] != sha256(root / "selected_states.json")):
        raise RuntimeError("M6C fixed protocol or 36-video generation is incomplete")
    for state in selected["states"]:
        if sha256(Path(state["source_o0"])) != state["source_o0_sha256"]:
            raise RuntimeError(f"Source O0 changed: {state['state']}")
    # This is the exact source sign convention and implementation used in M6B-4.
    source_actions = ["MOVE_FORWARD_LEFT"] * 2 + ["NOOP"] * 8 + ["MOVE_FORWARD_RIGHT"] * 6
    calibration = calibrate_source(measure_video(SOURCE_VIDEO), source_actions)
    sign = calibration["left_positive_multiplier"]
    labels = _conditions()
    videos = []
    transitions = []
    by_key = {}
    sidecars = {(row["state"], row["condition"], row["seed"]): row for row in generation["videos"]}
    if len(sidecars) != 36:
        raise RuntimeError("Duplicate or missing video condition")
    for state in selected["states"]:
        letter = state["state"]
        for seed in SEEDS:
            for condition in CONDITIONS:
                metadata = sidecars[(letter, condition, seed)]
                path = Path(metadata["video_path"])
                if (metadata["frame_count_verified"] != 17
                        or metadata["source_o0_sha256"] != state["source_o0_sha256"]
                        or metadata["checkpoint_sha256"] != protocol["adapter_checkpoint_sha256"]
                        or metadata["action_ids"] != protocol["conditions"][condition]
                        or metadata["prompt"] != protocol["prompt"]
                        or metadata["mask_preprocessing"]["frame0_mask_min_max"] != [0.0, 0.0]
                        or metadata["mask_preprocessing"]["future_mask_min_max"] != [1.0, 1.0]
                        or not metadata["conditioning_audit"]["native_pixel_mask_verified"]
                        or any(count != 60 for count in metadata["action_hook_calls"].values())):
                    raise RuntimeError(f"Conditioning or fixed settings changed: {path}")
                flow = measure_video(path)
                scored = score_future(flow, labels[condition], sign)
                record = {"state": letter, "sample_id": state["sample_id"],
                    "episode_id": state["episode_id"], "split": state["split"],
                    "condition": condition, "seed": seed, "video_path": str(path),
                    "video_sha256": sha256(path), "actions": labels[condition], **scored}
                videos.append(record)
                by_key[(letter, condition, seed)] = record
                for transition, row in enumerate(flow):
                    transitions.append({"state": letter, "condition": condition, "seed": seed,
                        "transition": transition, "action": labels[condition][transition],
                        "primary_future_only": transition >= 1, "calibrated_mean_u": sign * row["mean_u"], **row})
    for state in selected["states"]:
        letter = state["state"]
        for seed in SEEDS:
            noop = by_key[(letter, "all_noop", seed)]
            turn_baseline = float(np.mean([
                by_key[(letter, condition, seed)]["mean_turn_magnitude"]
                for condition in ("original", "reversed")]))
            noop["r_noop_matched_seed_turn_baseline"] = (
                noop["mean_noop_magnitude"] / turn_baseline if turn_baseline > 0 else None)
    flips = []
    for state in selected["states"]:
        letter = state["state"]
        for seed in SEEDS:
            flips.append({"state": letter, "seed": seed,
                **flip_future(by_key[(letter, "original", seed)], by_key[(letter, "reversed", seed)])})
    aggregate = summary(videos, flips)
    per_state = {state["state"]: summary(
        [row for row in videos if row["state"] == state["state"]],
        [row for row in flips if row["state"] == state["state"]]) for state in selected["states"]}
    criteria = {
        "turn_direction_accuracy": {"observed": aggregate["turn_direction_accuracy"],
            "minimum": THRESHOLDS["turn_direction_accuracy_min"],
            "pass": aggregate["turn_direction_accuracy"] >= THRESHOLDS["turn_direction_accuracy_min"]},
        "noop_motion_suppression": {"observed": aggregate["median_noop_motion_ratio"],
            "maximum": THRESHOLDS["median_noop_ratio_max"],
            "pass": aggregate["median_noop_motion_ratio"] <= THRESHOLDS["median_noop_ratio_max"]},
        "signed_action_motion_correlation": {"observed": aggregate["median_absolute_action_motion_correlation"],
            "minimum": THRESHOLDS["median_absolute_correlation_min"],
            "pass": aggregate["median_absolute_action_motion_correlation"] >= THRESHOLDS["median_absolute_correlation_min"]},
        "counterfactual_flip": {"observed": aggregate["matched_seed_flip_successes"],
            "minimum": THRESHOLDS["matched_pair_flip_min_of_12"],
            "pass": aggregate["matched_seed_flip_successes"] >= THRESHOLDS["matched_pair_flip_min_of_12"]},
    }
    passes = sum(item["pass"] for item in criteria.values())
    classification = ("A. HELD-OUT ACTION GENERALIZATION PASS" if passes == 4 else
        "B. PARTIAL GENERALIZATION" if passes >= 2 else "C. HELD-OUT GENERALIZATION FAIL")
    result = {"milestone": "M6C", "classification": classification,
        "passed_primary_criteria": passes, "thresholds": THRESHOLDS,
        "criteria": criteria, "aggregate": aggregate, "per_state": per_state,
        "source_flow_calibration": calibration, "flow_parameters": FLOW_PARAMETERS,
        "flow_protocol": "Exactly M6B-4/M6B-1 score_future and flip_future: generated O1->O2 through O15->O16 aligned to A1..A15; O0->O1 excluded",
        "opencv_version": cv2.__version__, "selected_states": selected["states"],
        "pairwise_o0_ssim": selected["pairwise_o0_ssim"],
        "matched_pairs": flips, "videos": videos,
        "generation_summary_path": str(root / "eval/generation_summary.json"),
        "transitions_path": str(root / "eval/transitions.jsonl"),
        "contact_sheet_path": str(root / "eval/contact_sheet.png")}
    eval_dir = root / "eval"
    (eval_dir / "transitions.jsonl").write_text("".join(json.dumps(row) + "\n" for row in transitions))
    (eval_dir / "metrics.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    contact_sheet(videos, selected, eval_dir / "contact_sheet.png")
    lines = ["# M6C: held-out action generalization test", "",
        f"**Classification: {classification}** ({passes}/4 aggregate criteria passed).", "",
        "Frozen M6B-4 step-800 cross-attention adapter, native VACE O0 conditioning, fixed generic prompt, "
        "36 videos (four held-out O0 states × three action conditions × seeds 42–44). No training or model changes.", "",
        "## Aggregate decision", "", "| Criterion | Required | Observed | Result |",
        "|---|---:|---:|---|",
        f"| Turn direction accuracy | ≥0.70 | {aggregate['turn_correct']}/{aggregate['turn_count']} = {aggregate['turn_direction_accuracy']:.3f} | {'PASS' if criteria['turn_direction_accuracy']['pass'] else 'FAIL'} |",
        f"| Median NOOP motion ratio | ≤0.50 | {aggregate['median_noop_motion_ratio']:.3f} | {'PASS' if criteria['noop_motion_suppression']['pass'] else 'FAIL'} |",
        f"| Median abs action-motion correlation | ≥0.50 | {aggregate['median_absolute_action_motion_correlation']:.3f} | {'PASS' if criteria['signed_action_motion_correlation']['pass'] else 'FAIL'} |",
        f"| Matched-seed flip | ≥9/12 | {aggregate['matched_seed_flip_successes']}/12 | {'PASS' if criteria['counterfactual_flip']['pass'] else 'FAIL'} |",
        "", "## Per held-out O0", "",
        "| State | Split / episode / sample | Turn direction | Median NOOP ratio | Median abs(r) | Flip |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for state in selected["states"]:
        row = per_state[state["state"]]
        lines.append(f"| {state['state']} | {state['split']} / {state['episode_id']} / {state['sample_id']} | "
            f"{row['turn_correct']}/{row['turn_count']} = {row['turn_direction_accuracy']:.3f} | "
            f"{row['median_noop_motion_ratio']:.3f} | {row['median_absolute_action_motion_correlation']:.3f} | "
            f"{row['matched_seed_flip_successes']}/3 |")
    lines += ["", "All four episodes are absent from the official TRAIN split and the 16-sample M6B training set. "
        "The six pairwise source-O0 SSIM values range from "
        f"{min(selected['pairwise_o0_ssim'][i][j] for i in range(4) for j in range(i+1,4)):.3f} to "
        f"{max(selected['pairwise_o0_ssim'][i][j] for i in range(4) for j in range(i+1,4)):.3f}.", "",
        "## Protocol and artifacts", "",
        "The exact validated M5C-5 Farneback ROI and M6B-4 source sign calibration were reused. "
        "Primary flow uses only generated O1→O2 through O15→O16, aligned to A1…A15. "
        "The median NOOP ratio is across 24 mixed-action videos; ALL-NOOP matched-seed motion ratios are descriptive. "
        "Within each O0/seed group the generator is reset to the same seed, so only the action IDs change. "
        "All 36 sidecars verify 17 frames, native black O0/white future masks, nonzero O0 control, "
        "and 60 action-hook calls per selected block. No generated frame was replaced.", "",
        f"Selection: `{root / 'selected_states.json'}`. Videos: `{eval_dir / 'videos'}`. "
        f"Full metrics: `{eval_dir / 'metrics.json'}`. Per-transition flow: `{eval_dir / 'transitions.jsonl'}`. "
        f"Contact sheet: `{eval_dir / 'contact_sheet.png'}`. Generation runtime/VRAM: "
        f"`{eval_dir / 'generation_summary.json'}`.", ""]
    (eval_dir / "report.md").write_text("\n".join(lines))
    return {"classification": classification, "criteria": criteria,
        "aggregate": aggregate, "per_state": per_state,
        "metrics_path": str(eval_dir / "metrics.json")}


if __name__ == "__main__":
    result = run()
    print(json.dumps(result, indent=2))
