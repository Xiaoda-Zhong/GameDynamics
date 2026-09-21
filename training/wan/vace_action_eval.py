"""Score M6B-1 action-conditioned VACE videos with the M5C-5 flow method.

The primary population is generated-future-to-generated-future transitions:
O1->O2 through O15->O16, aligned to A1 through A15. O0->O1 is saved as
descriptive flow but excluded because O0 is an observed conditioning frame.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from decord import VideoReader, cpu
from PIL import Image, ImageDraw

from training.wan.action_dynamics_metrics import (
    ACTION_SIGNAL,
    FLOW_PARAMETERS,
    FRAMES,
    HEIGHT,
    LEFT,
    NOOP,
    RIGHT,
    SEEDS,
    WIDTH,
    calibrate_source,
    measure_video,
)
from training.wan.current_frame_metrics import (
    generated_frames_rgb,
    source_frame_rgb,
    ssim_rgb,
)


REPO = Path(__file__).resolve().parents[2]
DEFAULT_EVAL = REPO / "data/experiments/m6b1_vace_action_tiny/eval"
SELECTED_STATES = REPO / "data/experiments/m6a3_four_state_1600/selected_states.json"
SOURCE_VIDEO = Path(
    "/root/autodl-tmp/outputs/wan/overfit_0016/"
    "single_sample_scene_action_0800/eval/source_17_frames_lossless.mp4"
)
CONDITIONS = ("original", "reversed", "all_noop")
PROMPT = "A first-person gameplay view in a retro 3D stone maze."
THRESHOLDS = {
    "turn_direction_accuracy_minimum": 0.70,
    "median_r_noop_maximum": 0.50,
    "median_absolute_action_motion_correlation_minimum": 0.50,
    "counterfactual_flip_successes_minimum_of_5": 4,
}
ACTION_IDS = {
    "original": [4] * 2 + [3] * 8 + [5] * 6,
    "reversed": [5] * 2 + [3] * 8 + [4] * 6,
    "all_noop": [3] * 16,
}


def _source() -> dict:
    selected = json.loads(SELECTED_STATES.read_text(encoding="utf-8"))
    state = next(
        (item for item in selected["states"] if item["sample_id"] == "my_way_home_episode_0028_clip_000011"),
        None,
    )
    if state is None or len(state["source_rgb_frames"]) != FRAMES:
        raise ValueError("M6A-3 state A or its 17 real frames are missing")
    expected = [LEFT] * 2 + [NOOP] * 8 + [RIGHT] * 6
    if state["raw_actions"] != expected:
        raise ValueError("The selected state A actions differ from the predeclared sequence")
    return state


def _conditions() -> dict[str, list[str]]:
    return {
        "original": [LEFT] * 2 + [NOOP] * 8 + [RIGHT] * 6,
        "reversed": [RIGHT] * 2 + [NOOP] * 8 + [LEFT] * 6,
        "all_noop": [NOOP] * 16,
    }


def _median(values: list[float | None]) -> float | None:
    present = [value for value in values if value is not None]
    return float(np.median(present)) if present else None


def score_future(rows: list[dict], actions: list[str], sign: int) -> dict:
    """Apply the M5C-5 formulas to aligned A1..A15 and O1..O16 only."""
    if len(rows) != 16 or len(actions) != 16 or sign not in (-1, 1):
        raise ValueError("Expected 16 aligned source transitions and calibrated flow sign")
    if any(action not in ACTION_SIGNAL for action in actions):
        raise ValueError("Unexpected action label for the fixed counterfactual protocol")
    primary_rows = rows[1:]
    signal = [ACTION_SIGNAL[action] for action in actions[1:]]
    signed_u = [sign * row["mean_u"] for row in primary_rows]
    turn_indices = [i for i, value in enumerate(signal) if value != 0]
    noop_indices = [i for i, value in enumerate(signal) if value == 0]
    correct = sum(signal[i] * signed_u[i] > 0 for i in turn_indices)
    mean_turn = (
        float(np.mean([primary_rows[i]["mean_magnitude"] for i in turn_indices]))
        if turn_indices else None
    )
    mean_noop = (
        float(np.mean([primary_rows[i]["mean_magnitude"] for i in noop_indices]))
        if noop_indices else None
    )
    ratio = mean_noop / mean_turn if mean_turn is not None and mean_turn > 0 and mean_noop is not None else None
    correlation = (
        float(np.corrcoef(signal, signed_u)[0, 1])
        if len(set(signal)) > 1 and np.std(signed_u) >= 1e-12 else None
    )
    return {
        "primary_transition_indices": list(range(1, 16)),
        "turn_correct": correct,
        "turn_count": len(turn_indices),
        "turn_direction_accuracy": correct / len(turn_indices) if turn_indices else None,
        "mean_noop_magnitude": mean_noop,
        "mean_turn_magnitude": mean_turn,
        "r_noop_within_video": ratio,
        "action_motion_r_signed": correlation,
        "action_motion_r_absolute": abs(correlation) if correlation is not None else None,
        "segment_mean_signed_u": {
            "first_turn_A1": signed_u[0],
            "middle_noop_A2_to_A9": float(np.mean(signed_u[1:9])),
            "final_turn_A10_to_A15": float(np.mean(signed_u[9:])),
        },
        "excluded_observed_O0_to_generated_O1": {
            "mean_u": rows[0]["mean_u"],
            "mean_magnitude": rows[0]["mean_magnitude"],
        },
    }


def flip_future(original: dict, reversed_run: dict) -> dict:
    """Require the first and last future-only turns to follow both action signs."""
    original_segments = original["segment_mean_signed_u"]
    reversed_segments = reversed_run["segment_mean_signed_u"]
    first = (
        original_segments["first_turn_A1"] > 0
        and reversed_segments["first_turn_A1"] < 0
    )
    last = (
        original_segments["final_turn_A10_to_A15"] < 0
        and reversed_segments["final_turn_A10_to_A15"] > 0
    )
    return {
        "original_first_turn_A1_signed_u": original_segments["first_turn_A1"],
        "reversed_first_turn_A1_signed_u": reversed_segments["first_turn_A1"],
        "original_final_turn_A10_to_A15_signed_u": original_segments["final_turn_A10_to_A15"],
        "reversed_final_turn_A10_to_A15_signed_u": reversed_segments["final_turn_A10_to_A15"],
        "first_segment_expected_flip": first,
        "final_segment_expected_flip": last,
        "success": first and last,
    }


def _source_similarity(path: Path, real_future: list) -> dict:
    generated = generated_frames_rgb(path)
    scores = [ssim_rgb(generated[index], real_future[index - 1]) for index in range(1, FRAMES)]
    return {
        "future_frame_ssim": scores,
        "mean_future_ssim": float(np.mean(scores)),
        "median_future_ssim": float(np.median(scores)),
    }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _contact_sheet(videos: list[dict], path: Path) -> None:
    frame_indices = (1, 4, 8, 12, 16)
    thumb_w, thumb_h, label_w, label_h = 224, 128, 120, 22
    sheet = Image.new("RGB", (label_w + len(frame_indices) * thumb_w, len(videos) * (thumb_h + label_h)), "#202020")
    draw = ImageDraw.Draw(sheet)
    for row_index, record in enumerate(videos):
        top = row_index * (thumb_h + label_h)
        draw.text((5, top + 5), f"{record['condition']} {record['seed']}", fill="white")
        reader = VideoReader(record["video_path"], ctx=cpu(0))
        if len(reader) != FRAMES:
            raise ValueError(f"Expected {FRAMES} frames: {record['video_path']}")
        for column, frame_index in enumerate(frame_indices):
            image = Image.fromarray(reader[frame_index].asnumpy(), "RGB")
            if image.size != (WIDTH, HEIGHT):
                raise ValueError(f"Unexpected video resolution: {record['video_path']}")
            image = image.resize((thumb_w, thumb_h), Image.Resampling.BICUBIC)
            sheet.paste(image, (label_w + column * thumb_w, top + label_h))
            draw.text((label_w + column * thumb_w + 4, top + 4), f"O{frame_index}", fill="white")
    sheet.save(path)


def analyze(eval_dir: Path) -> dict:
    import cv2

    if cv2.__version__ != "4.12.0":
        raise RuntimeError(f"M5C-5 used OpenCV 4.12.0; found {cv2.__version__}")
    source = _source()
    if not SOURCE_VIDEO.is_file():
        raise FileNotFoundError(f"Validated M5C-5 source video missing: {SOURCE_VIDEO}")
    source_rows = measure_video(SOURCE_VIDEO)
    calibration = calibrate_source(source_rows, source["raw_actions"])
    sign = calibration["left_positive_multiplier"]
    source_future = score_future(source_rows, source["raw_actions"], sign)
    real_future = [source_frame_rgb(frame) for frame in source["source_rgb_frames"][1:]]
    actions = _conditions()
    records: list[dict] = []
    by_condition_seed: dict[tuple[str, int], dict] = {}
    transition_records: list[dict] = []
    for condition in CONDITIONS:
        for seed in SEEDS:
            path = eval_dir / "videos" / f"{condition}_seed_{seed}.mp4"
            if not path.is_file():
                raise FileNotFoundError(f"Missing required M6B-1 video: {path}")
            rows = measure_video(path)
            record = {
                "condition": condition,
                "seed": seed,
                "video_path": str(path.resolve()),
                "video_sha256": _sha256(path),
                "actions": actions[condition],
                **score_future(rows, actions[condition], sign),
            }
            if condition == "original":
                record["descriptive_source_similarity_O1_to_O16"] = _source_similarity(path, real_future)
            records.append(record)
            by_condition_seed[(condition, seed)] = record
            for index, row in enumerate(rows):
                transition_records.append({
                    "condition": condition,
                    "seed": seed,
                    "transition": index,
                    "action": actions[condition][index],
                    "primary_future_only": index >= 1,
                    "calibrated_mean_u": sign * row["mean_u"],
                    **row,
                })
    for seed in SEEDS:
        noop = by_condition_seed[("all_noop", seed)]
        turn_baseline = float(np.mean([
            by_condition_seed[(condition, seed)]["mean_turn_magnitude"]
            for condition in ("original", "reversed")
        ]))
        noop["r_noop_matched_seed_turn_baseline"] = (
            noop["mean_noop_magnitude"] / turn_baseline if turn_baseline > 0 else None
        )
    flips = [
        {"seed": seed, **flip_future(by_condition_seed[("original", seed)], by_condition_seed[("reversed", seed)])}
        for seed in SEEDS
    ]
    actionable = [record for record in records if record["condition"] in ("original", "reversed")]
    direction_correct = sum(record["turn_correct"] for record in actionable)
    direction_total = sum(record["turn_count"] for record in actionable)
    median_ratio = _median([record["r_noop_within_video"] for record in actionable])
    median_abs_r = float(np.median([
        record["action_motion_r_absolute"] if record["action_motion_r_absolute"] is not None else 0.0
        for record in actionable
    ]))
    successes = sum(record["success"] for record in flips)
    criteria = {
        "turn_direction_accuracy": {
            "observed": direction_correct / direction_total,
            "minimum": THRESHOLDS["turn_direction_accuracy_minimum"],
            "pass": direction_correct / direction_total >= THRESHOLDS["turn_direction_accuracy_minimum"],
        },
        "noop_motion_suppression": {
            "observed": median_ratio,
            "maximum": THRESHOLDS["median_r_noop_maximum"],
            "pass": median_ratio is not None and median_ratio <= THRESHOLDS["median_r_noop_maximum"],
        },
        "signed_action_motion_correlation": {
            "observed_median_absolute_r": median_abs_r,
            "minimum": THRESHOLDS["median_absolute_action_motion_correlation_minimum"],
            "pass": median_abs_r >= THRESHOLDS["median_absolute_action_motion_correlation_minimum"],
        },
        "counterfactual_flip": {
            "observed_successes_of_5": successes,
            "minimum_successes": THRESHOLDS["counterfactual_flip_successes_minimum_of_5"],
            "pass": successes >= THRESHOLDS["counterfactual_flip_successes_minimum_of_5"],
        },
    }
    passes = sum(criterion["pass"] for criterion in criteria.values())
    classification = (
        "A. EXPLICIT ACTION CONTROL PASS" if passes == 4 else
        "B. PARTIAL ACTION CONTROL" if passes >= 2 else
        "C. EXPLICIT ACTION CONTROL FAIL"
    )
    condition_aggregates = {}
    for condition in CONDITIONS:
        subset = [record for record in records if record["condition"] == condition]
        condition_aggregates[condition] = {
            "turn_correct": sum(record["turn_correct"] for record in subset),
            "turn_count": sum(record["turn_count"] for record in subset),
            "turn_direction_accuracy": (
                sum(record["turn_correct"] for record in subset) / sum(record["turn_count"] for record in subset)
                if condition != "all_noop" else None
            ),
            "median_r_noop_within_video": _median([record["r_noop_within_video"] for record in subset]),
            "median_action_motion_r_signed": _median([record["action_motion_r_signed"] for record in subset]),
            "median_action_motion_r_absolute": _median([record["action_motion_r_absolute"] for record in subset]),
            "median_matched_seed_noop_ratio": (
                _median([record["r_noop_matched_seed_turn_baseline"] for record in subset])
                if condition == "all_noop" else None
            ),
        }
    metrics = {
        "milestone": "M6B-1 explicit VACE action adapter tiny pilot",
        "classification": classification,
        "source_sample_id": source["sample_id"],
        "source_video": str(SOURCE_VIDEO),
        "source_video_sha256": _sha256(SOURCE_VIDEO),
        "generic_prompt": PROMPT,
        "protocol": {
            "flow_parameters": FLOW_PARAMETERS,
            "flow_procedure": "training.wan.action_dynamics_metrics.measure_video, unchanged from M5C-5",
            "sign_calibration": "M5C-5 calibrate_source on all 16 real-source transitions; positive signed mean_u denotes left-turn motion",
            "primary_transition_indices": list(range(1, 16)),
            "primary_transition_rule": "O1->O2 through O15->O16, aligned to A1..A15; O0->O1 is descriptive only",
            "turn_direction": "strict expected sign of calibrated mean_u on each turn transition",
            "r_noop": "per mixed-action video: mean NOOP flow magnitude divided by mean turn flow magnitude; median across 10 videos",
            "signed_correlation": "Pearson r of [+1 left, 0 NOOP, -1 right] and calibrated mean_u across A1..A15; undefined r counts as |r|=0 in criterion",
            "flip": "for each matched seed, ORIGINAL A1 >0 and A10..A15 <0; REVERSED A1 <0 and A10..A15 >0",
            "all_noop": "within-video turn ratio, turn accuracy, and r undefined; matched-seed NOOP/turn ratio descriptive",
            "source_similarity": "descriptive SSIM of decoded generated O1..O16 against exact real source PNGs resized with M6A bicubic source_frame_rgb",
            "source_frame_fidelity": "O0 excluded from success criteria and source similarity",
            "opencv_version": cv2.__version__,
        },
        "source_calibration": calibration,
        "source_future_only_metrics": source_future,
        "thresholds": THRESHOLDS,
        "criteria": criteria,
        "passed_primary_criteria": passes,
        "aggregate_turn_correct": direction_correct,
        "aggregate_turn_total": direction_total,
        "aggregate_turn_direction_accuracy": direction_correct / direction_total,
        "median_r_noop_within_turn_containing_videos": median_ratio,
        "median_absolute_action_motion_correlation": median_abs_r,
        "counterfactual_flip_successes": successes,
        "condition_aggregates": condition_aggregates,
        "counterfactual_flip_tests": flips,
        "videos": records,
        "video_paths": [record["video_path"] for record in records],
        "transition_metrics_path": str((eval_dir / "transitions.jsonl").resolve()),
        "contact_sheet_path": str((eval_dir / "contact_sheet.png").resolve()),
    }
    eval_dir.mkdir(parents=True, exist_ok=True)
    (eval_dir / "transitions.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in transition_records), encoding="utf-8"
    )
    (eval_dir / "metrics.json").write_text(json.dumps(metrics, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    _contact_sheet(records, eval_dir / "contact_sheet.png")
    (eval_dir / "report.md").write_text(_report(metrics), encoding="utf-8")
    return metrics


def _fmt(value: float | None, digits: int = 3) -> str:
    return "—" if value is None else f"{value:.{digits}f}"


def _report(metrics: dict) -> str:
    criteria = metrics["criteria"]
    lines = [
        "# M6B-1: explicit VACE action adapter tiny pilot",
        "",
        f"**Classification: {metrics['classification']}**",
        "",
        "The official M6A-4 classification remains C under its predeclared O0 hard-anchor rule. "
        "M6B-1 treats O0 as observed conditioning and evaluates future dynamics only.",
        "",
        "## Fixed flow protocol",
        "",
        "The M5C-5 OpenCV Farneback implementation and ROI are reused unchanged. The real source clip "
        "calibrates flow sign from its two left and six right turns. Primary M6B-1 scoring uses only "
        "the 15 generated-future pairs O1→O2 through O15→O16, aligned to A1…A15. "
        "O0→O1 is retained as descriptive flow and is excluded from every criterion. "
        "Consequently the first flip segment is A1 alone and the final segment is A10…A15. "
        "The NOOP ratio is computed within each mixed-action video and the median of 10 ratios is primary, as in M5C-5.",
        "",
        f"Source sign: {metrics['source_calibration']['convention']}; "
        f"source left mean raw u={metrics['source_calibration']['left_turn_mean_raw_u']:+.3f}, "
        f"right mean raw u={metrics['source_calibration']['right_turn_mean_raw_u']:+.3f} px.",
        "",
        "## Primary decision",
        "",
        "| Criterion | Required | Observed | Result |",
        "|---|---:|---:|---|",
        f"| Turn direction | ≥0.70 | {metrics['aggregate_turn_correct']}/{metrics['aggregate_turn_total']} = {metrics['aggregate_turn_direction_accuracy']:.3f} | {'PASS' if criteria['turn_direction_accuracy']['pass'] else 'FAIL'} |",
        f"| Median R_noop | ≤0.50 | {_fmt(metrics['median_r_noop_within_turn_containing_videos'])} | {'PASS' if criteria['noop_motion_suppression']['pass'] else 'FAIL'} |",
        f"| Median abs(r) | ≥0.50 | {metrics['median_absolute_action_motion_correlation']:.3f} | {'PASS' if criteria['signed_action_motion_correlation']['pass'] else 'FAIL'} |",
        f"| Matched-seed flip | ≥4/5 | {metrics['counterfactual_flip_successes']}/5 | {'PASS' if criteria['counterfactual_flip']['pass'] else 'FAIL'} |",
        "",
        "## Per-video future-only flow",
        "",
        "| Condition | Seed | Turns correct | R_noop | Signed r | abs(r) | Matched all-NOOP ratio |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for record in metrics["videos"]:
        turns = "—" if record["turn_count"] == 0 else f"{record['turn_correct']}/{record['turn_count']}"
        lines.append(
            f"| {record['condition']} | {record['seed']} | {turns} | "
            f"{_fmt(record['r_noop_within_video'])} | {_fmt(record['action_motion_r_signed'])} | "
            f"{_fmt(record['action_motion_r_absolute'])} | "
            f"{_fmt(record.get('r_noop_matched_seed_turn_baseline'))} |"
        )
    lines.extend([
        "",
        "## Counterfactual segments",
        "",
        "| Seed | ORIGINAL A1 | REVERSED A1 | ORIGINAL A10…A15 | REVERSED A10…A15 | Flip |",
        "|---:|---:|---:|---:|---:|---|",
    ])
    for flip in metrics["counterfactual_flip_tests"]:
        lines.append(
            f"| {flip['seed']} | {_fmt(flip['original_first_turn_A1_signed_u'])} | "
            f"{_fmt(flip['reversed_first_turn_A1_signed_u'])} | "
            f"{_fmt(flip['original_final_turn_A10_to_A15_signed_u'])} | "
            f"{_fmt(flip['reversed_final_turn_A10_to_A15_signed_u'])} | "
            f"{'Yes' if flip['success'] else 'No'} |"
        )
    original = [row for row in metrics["videos"] if row["condition"] == "original"]
    mean_ssim = float(np.mean([row["descriptive_source_similarity_O1_to_O16"]["mean_future_ssim"] for row in original]))
    lines.extend([
        "",
        "## Supporting future-frame similarity",
        "",
        f"ORIGINAL runs mean O1…O16 SSIM to the real source: {mean_ssim:.3f}; descriptive only, no threshold. "
        "Decoded O0 fidelity is not evaluated as a success criterion.",
        "",
        "## Artifacts",
        "",
        f"Metrics: `{metrics['transition_metrics_path'].rsplit('/', 1)[0]}/metrics.json`. "
        f"Flow transitions: `{metrics['transition_metrics_path']}`. "
        f"Contact sheet: `{metrics['contact_sheet_path']}`. "
        "The metrics JSON lists all 15 video paths and SHA-256 hashes.",
        "",
    ])
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--eval-dir", type=Path, default=DEFAULT_EVAL)
    args = parser.parse_args()
    metrics = analyze(args.eval_dir)
    print(json.dumps({
        "classification": metrics["classification"],
        "criteria": metrics["criteria"],
        "metrics_path": str((args.eval_dir / "metrics.json").resolve()),
    }, indent=2))


if __name__ == "__main__":
    main()
