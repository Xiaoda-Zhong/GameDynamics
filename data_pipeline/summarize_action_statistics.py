#!/usr/bin/env python3
"""Summarize validated Milestone 3B action statistics."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np

try:
    from .validate_action_statistics import validate_action_statistics
except ImportError:
    from validate_action_statistics import validate_action_statistics


PERCENTILES = (10, 25, 50, 75, 90)
OVERLAP_NOTE = (
    "Transition-level counts are sample-level statistics across overlapping clips; "
    "they are not the unique source-episode transition distribution."
)


def summarize_action_statistics(clip_index_path: Path) -> dict:
    clip_index_path = _normalize_index_path(clip_index_path)
    errors = validate_action_statistics(clip_index_path)
    if errors:
        details = "\n".join(f"  - {error}" for error in errors)
        raise ValueError(f"invalid action statistics {clip_index_path}:\n{details}")

    clip_index = json.loads(clip_index_path.read_text(encoding="utf-8"))
    clips = clip_index["clips"]
    source_episode = Path(clip_index["source_episode"])
    source_metadata = json.loads(
        (source_episode / "metadata.json").read_text(encoding="utf-8")
    )
    action_labels = [entry["label"] for entry in source_metadata["action_space"]]

    overall_counts = Counter(
        action
        for clip in clips
        for action in clip["actions"]
    )
    total_actions = sum(overall_counts.values())
    ordered_counts = {
        action: overall_counts.get(action, 0)
        for action in action_labels
    }
    ordered_fractions = {
        action: count / total_actions if total_actions else 0.0
        for action, count in ordered_counts.items()
    }
    dominated_counts = Counter(
        clip["action_statistics"]["dominant_action"]
        for clip in clips
    )
    clips_dominated_by_action = {
        action: dominated_counts.get(action, 0)
        for action in action_labels
    }
    unique_action_distribution = Counter(
        clip["action_statistics"]["unique_action_count"]
        for clip in clips
    )

    dominant_fractions = np.asarray(
        [clip["action_statistics"]["dominant_action_fraction"] for clip in clips],
        dtype=np.float64,
    )
    noop_fractions = np.asarray(
        [clip["action_statistics"]["noop_fraction"] for clip in clips],
        dtype=np.float64,
    )
    entropies = np.asarray(
        [clip["action_statistics"]["action_entropy"] for clip in clips],
        dtype=np.float64,
    )

    return {
        "total_clips": len(clips),
        "total_action_transitions_across_clips": total_actions,
        "overall_action_frequencies": ordered_counts,
        "overall_action_fractions": ordered_fractions,
        "clips_dominated_by_action": clips_dominated_by_action,
        "mean_dominant_action_fraction": _mean_or_none(dominant_fractions),
        "median_dominant_action_fraction": _median_or_none(dominant_fractions),
        "mean_noop_fraction": _mean_or_none(noop_fractions),
        "median_noop_fraction": _median_or_none(noop_fractions),
        "unique_action_count_distribution": {
            str(count): frequency
            for count, frequency in sorted(unique_action_distribution.items())
        },
        "action_entropy_mean": _mean_or_none(entropies),
        "action_entropy_std": _std_or_none(entropies),
        "action_entropy_percentiles": _percentiles_or_none(entropies),
        "lowest_entropy_clip_ids": _rank_clip_ids(
            clips,
            "action_entropy",
            reverse=False,
        ),
        "highest_entropy_clip_ids": _rank_clip_ids(
            clips,
            "action_entropy",
            reverse=True,
        ),
        "highest_noop_clip_ids": _rank_clip_ids(
            clips,
            "noop_fraction",
            reverse=True,
        ),
        "overlap_note": OVERLAP_NOTE,
    }


def _rank_clip_ids(clips: list[dict], field: str, reverse: bool) -> list[int]:
    direction = -1 if reverse else 1
    ranked = sorted(
        clips,
        key=lambda clip: (
            direction * clip["action_statistics"][field],
            clip["clip_id"],
        ),
    )
    return [clip["clip_id"] for clip in ranked[:5]]


def _mean_or_none(values: np.ndarray) -> float | None:
    return float(np.mean(values)) if len(values) else None


def _median_or_none(values: np.ndarray) -> float | None:
    return float(np.median(values)) if len(values) else None


def _std_or_none(values: np.ndarray) -> float | None:
    return float(np.std(values)) if len(values) else None


def _percentiles_or_none(values: np.ndarray) -> dict[str, float | None]:
    if not len(values):
        return {f"P{percentile}": None for percentile in PERCENTILES}
    results = np.percentile(values, PERCENTILES)
    return {
        f"P{percentile}": float(value)
        for percentile, value in zip(PERCENTILES, results)
    }


def _normalize_index_path(path: Path) -> Path:
    return path / "clips.json" if path.is_dir() else path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("clip_index", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    try:
        summary = summarize_action_statistics(args.clip_index)
    except ValueError as error:
        raise SystemExit(str(error)) from error
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
