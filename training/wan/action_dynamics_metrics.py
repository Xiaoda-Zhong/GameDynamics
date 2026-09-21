"""Deterministic dense-flow measurements for the M5C-5 action diagnostic."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
from decord import VideoReader, cpu


WIDTH = 448
HEIGHT = 256
FRAMES = 17
LEFT = "MOVE_FORWARD_LEFT"
RIGHT = "MOVE_FORWARD_RIGHT"
NOOP = "NOOP"
ACTION_SIGNAL = {LEFT: 1, NOOP: 0, RIGHT: -1}
SEEDS = (42, 43, 44, 45, 46)
FLOW_PARAMETERS = {
    "method": "OpenCV Farneback dense optical flow",
    "pyr_scale": 0.5,
    "levels": 3,
    "winsize": 15,
    "iterations": 3,
    "poly_n": 5,
    "poly_sigma": 1.2,
    "flags": 0,
    "roi_xyxy": [45, 26, 403, 230],
    "roi_description": "central 80% of width and height; no texture mask",
    "source_resize": "cv2.INTER_CUBIC to 448x256 before flow",
    "flow_unit": "pixels per transition at 448x256",
    "direction_statistic": "mean_u; median_u is reported for diagnostic context",
}


def measure_video(video_path: Path) -> list[dict]:
    """Return 16 dense-flow transition measurements, preserving pixel flow sign."""
    import cv2

    cv2.setNumThreads(1)
    video = VideoReader(str(video_path), ctx=cpu(0))
    if len(video) != FRAMES:
        raise ValueError(f"Expected {FRAMES} frames in {video_path}, found {len(video)}")
    grays = []
    for index in range(FRAMES):
        rgb = video[index].asnumpy()
        if tuple(rgb.shape[:2]) != (HEIGHT, WIDTH):
            rgb = cv2.resize(rgb, (WIDTH, HEIGHT), interpolation=cv2.INTER_CUBIC)
        grays.append(cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY))
    x0, y0, x1, y1 = FLOW_PARAMETERS["roi_xyxy"]
    rows = []
    for index in range(FRAMES - 1):
        flow = cv2.calcOpticalFlowFarneback(
            grays[index], grays[index + 1], None,
            FLOW_PARAMETERS["pyr_scale"], FLOW_PARAMETERS["levels"],
            FLOW_PARAMETERS["winsize"], FLOW_PARAMETERS["iterations"],
            FLOW_PARAMETERS["poly_n"], FLOW_PARAMETERS["poly_sigma"],
            FLOW_PARAMETERS["flags"],
        )[y0:y1, x0:x1]
        u = flow[..., 0]
        v = flow[..., 1]
        row = {
            "transition": index,
            "mean_u": float(np.mean(u, dtype=np.float64)),
            "median_u": float(np.median(u)),
            "mean_v": float(np.mean(v, dtype=np.float64)),
            "mean_magnitude": float(np.mean(np.hypot(u, v), dtype=np.float64)),
        }
        if not all(math.isfinite(value) for key, value in row.items() if key != "transition"):
            raise FloatingPointError(f"Nonfinite flow in {video_path}, transition {index}")
        rows.append(row)
    return rows


def calibrate_source(rows: list[dict], actions: list[str]) -> dict:
    """Calibrate signs from actual left and right turns in the source video."""
    if len(rows) != 16 or len(actions) != 16:
        raise ValueError("Source needs 16 aligned transitions and actions")
    left = [row["mean_u"] for row, action in zip(rows, actions) if action == LEFT]
    right = [row["mean_u"] for row, action in zip(rows, actions) if action == RIGHT]
    if len(left) != 2 or len(right) != 6:
        raise ValueError("Source must contain 2 left and 6 right turning transitions")
    left_mean = float(np.mean(left))
    right_mean = float(np.mean(right))
    if left_mean == 0 or right_mean == 0 or left_mean * right_mean >= 0:
        raise ValueError("Source turns do not establish opposite horizontal-flow signs")
    sign = 1 if left_mean > 0 else -1
    return {
        "left_turn_mean_raw_u": left_mean,
        "right_turn_mean_raw_u": right_mean,
        "left_positive_multiplier": sign,
        "convention": (
            "positive raw u means left-turn camera motion; negative means right-turn"
            if sign == 1 else
            "negative raw u means left-turn camera motion; positive means right-turn"
        ),
    }


def _correlation(action_values: list[int], signed_flow: list[float]) -> float | None:
    if len(action_values) != len(signed_flow):
        raise ValueError("Action and flow sequences differ in length")
    if len(set(action_values)) == 1 or np.std(signed_flow) < 1e-12:
        return None
    return float(np.corrcoef(action_values, signed_flow)[0, 1])


def evaluate_video(rows: list[dict], actions: list[str], left_positive_multiplier: int) -> dict:
    """Score directions, NOOP suppression, and signed temporal alignment."""
    if len(rows) != 16 or len(actions) != 16 or left_positive_multiplier not in (-1, 1):
        raise ValueError("Expected 16 transitions/actions and a calibrated flow sign")
    if any(action not in ACTION_SIGNAL for action in actions):
        raise ValueError("Encountered an unsupported action label")
    signal = [ACTION_SIGNAL[action] for action in actions]
    signed_flow = [left_positive_multiplier * row["mean_u"] for row in rows]
    turn_indices = [index for index, value in enumerate(signal) if value != 0]
    noop_indices = [index for index, value in enumerate(signal) if value == 0]
    correct = sum(signal[index] * signed_flow[index] > 0 for index in turn_indices)
    turn_motion = float(np.mean([rows[index]["mean_magnitude"] for index in turn_indices])) if turn_indices else None
    noop_motion = float(np.mean([rows[index]["mean_magnitude"] for index in noop_indices])) if noop_indices else None
    ratio = noop_motion / turn_motion if turn_motion is not None and turn_motion > 0 and noop_motion is not None else None
    r = _correlation(signal, signed_flow)
    return {
        "turn_correct": correct,
        "turn_count": len(turn_indices),
        "turn_direction_accuracy": correct / len(turn_indices) if turn_indices else None,
        "mean_noop_magnitude": noop_motion,
        "mean_turn_magnitude": turn_motion,
        "r_noop_within_video": ratio,
        "action_motion_r_signed": r,
        "action_motion_r_absolute": abs(r) if r is not None else None,
        "segment_mean_signed_u": {
            "first_2": float(np.mean(signed_flow[:2])),
            "middle_8": float(np.mean(signed_flow[2:10])),
            "last_6": float(np.mean(signed_flow[10:])),
        },
    }


def flip_test(original: dict, reversed_video: dict) -> dict:
    """Require both segment pairs to reverse in the expected directions."""
    o = original["segment_mean_signed_u"]
    r = reversed_video["segment_mean_signed_u"]
    first = o["first_2"] > 0 and r["first_2"] < 0
    final = o["last_6"] < 0 and r["last_6"] > 0
    return {
        "original_first_2_signed_u": o["first_2"],
        "reversed_first_2_signed_u": r["first_2"],
        "original_last_6_signed_u": o["last_6"],
        "reversed_last_6_signed_u": r["last_6"],
        "first_segment_expected_flip": first,
        "final_segment_expected_flip": final,
        "success": first and final,
    }
