#!/usr/bin/env python3
"""Summarize validated Milestone 3A motion scores."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

try:
    from .validate_motion_scores import validate_motion_scores
except ImportError:
    from validate_motion_scores import validate_motion_scores


PERCENTILES = (10, 25, 50, 75, 90)
HISTOGRAM_BIN_COUNT = 5


def summarize_motion_scores(clip_index_path: Path) -> dict:
    clip_index_path = _normalize_index_path(clip_index_path)
    errors = validate_motion_scores(clip_index_path)
    if errors:
        details = "\n".join(f"  - {error}" for error in errors)
        raise ValueError(f"invalid motion scores {clip_index_path}:\n{details}")

    clip_index = json.loads(clip_index_path.read_text(encoding="utf-8"))
    clips = clip_index["clips"]
    clip_means = np.asarray(
        [clip["motion_mean"] for clip in clips],
        dtype=np.float64,
    )
    if not clips:
        return {
            "number_of_clips": 0,
            "global_mean_motion_score": None,
            "min_clip_motion_mean": None,
            "max_clip_motion_mean": None,
            "std_across_clip_motion_means": None,
            "percentiles": {f"P{value}": None for value in PERCENTILES},
            "lowest_motion_clip_ids": [],
            "highest_motion_clip_ids": [],
            "histogram": [],
        }

    percentile_values = np.percentile(clip_means, PERCENTILES)
    lowest = sorted(
        clips,
        key=lambda clip: (clip["motion_mean"], clip["clip_id"]),
    )[:5]
    highest = sorted(
        clips,
        key=lambda clip: (-clip["motion_mean"], clip["clip_id"]),
    )[:5]
    return {
        "number_of_clips": len(clips),
        "global_mean_motion_score": float(np.mean(clip_means)),
        "min_clip_motion_mean": float(np.min(clip_means)),
        "max_clip_motion_mean": float(np.max(clip_means)),
        "std_across_clip_motion_means": float(np.std(clip_means)),
        "percentiles": {
            f"P{percentile}": float(value)
            for percentile, value in zip(PERCENTILES, percentile_values)
        },
        "lowest_motion_clip_ids": [clip["clip_id"] for clip in lowest],
        "highest_motion_clip_ids": [clip["clip_id"] for clip in highest],
        "histogram": _text_histogram(clip_means),
    }


def _text_histogram(values: np.ndarray) -> list[str]:
    minimum = float(np.min(values))
    maximum = float(np.max(values))
    if minimum == maximum:
        return [f"{minimum:.6f}-{maximum:.6f}: {len(values)} clips"]

    counts, edges = np.histogram(
        values,
        bins=HISTOGRAM_BIN_COUNT,
        range=(minimum, maximum),
    )
    return [
        f"{lower:.6f}-{upper:.6f}: {int(count)} clips"
        for lower, upper, count in zip(edges[:-1], edges[1:], counts)
    ]


def _normalize_index_path(path: Path) -> Path:
    return path / "clips.json" if path.is_dir() else path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("clip_index", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    try:
        summary = summarize_motion_scores(args.clip_index)
    except ValueError as error:
        raise SystemExit(str(error)) from error
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
