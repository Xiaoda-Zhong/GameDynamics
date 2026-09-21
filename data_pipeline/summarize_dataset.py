#!/usr/bin/env python3
"""Summarize a validated global clip dataset and its episode-level splits."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np

try:
    from .split_dataset import SPLIT_NAMES, validate_dataset_splits
except ImportError:
    from split_dataset import SPLIT_NAMES, validate_dataset_splits


MOTION_PERCENTILES = (10, 25, 50, 75, 90)
TERMINATION_REASONS = (
    "task_terminal",
    "environment_timeout",
    "collector_truncation",
)
OVERLAP_NOTE = (
    "Aggregate actions count clip samples. Overlapping clips may count the same "
    "source transition more than once."
)


def summarize_dataset(
    dataset_index_path: Path,
    split_dir: Path | None = None,
) -> dict:
    """Return dataset, split, storage, motion, action, and episode statistics."""
    dataset_index_path = _normalize_dataset_index_path(dataset_index_path)
    destination = (split_dir or dataset_index_path.parent).resolve()
    errors = validate_dataset_splits(dataset_index_path, destination)
    if errors:
        details = "\n".join(f"  - {error}" for error in errors)
        raise ValueError(f"invalid dataset or splits:\n{details}")

    dataset = _read_json(dataset_index_path)
    splits = {
        split_name: _read_json(destination / f"{split_name}.json")
        for split_name in SPLIT_NAMES
    }
    clips = dataset["clips"]
    action_labels = dataset["declared_action_space"]

    rgb_paths = {Path(clip["rgb_video"]) for clip in clips}
    depth_paths = {
        Path(path)
        for clip in clips
        for path in (clip["depth_frames"] or [])
    }
    motion_means = np.asarray(
        [clip["motion_statistics"]["mean"] for clip in clips],
        dtype=np.float64,
    )
    action_counts = Counter(
        action
        for clip in clips
        for action in clip["actions"]
    )
    ordered_action_counts = {
        action: action_counts.get(action, 0) for action in action_labels
    }
    total_actions = sum(ordered_action_counts.values())
    action_fractions = {
        action: count / total_actions if total_actions else 0.0
        for action, count in ordered_action_counts.items()
    }
    termination_counts = Counter(
        episode["termination_reason"] for episode in dataset["episodes"]
    )

    return {
        "scenario_name": dataset["scenario_name"],
        "dataset_index": str(dataset_index_path),
        "number_of_episodes": dataset["num_episodes"],
        "number_of_clips": dataset["num_clips"],
        "split_counts": {
            split_name: {
                "episodes": splits[split_name]["num_episodes"],
                "clips": splits[split_name]["num_clips"],
            }
            for split_name in SPLIT_NAMES
        },
        "storage": {
            "rgb_video_bytes": _sum_file_sizes(rgb_paths),
            "depth_bytes": _sum_file_sizes(depth_paths),
            "rgb_video_files": len(rgb_paths),
            "depth_files": len(depth_paths),
        },
        "motion_distribution": _summarize_motion(motion_means),
        "aggregate_action_distribution": {
            "num_clip_sample_transitions": total_actions,
            "counts": ordered_action_counts,
            "fractions": action_fractions,
            "overlap_note": OVERLAP_NOTE,
        },
        "termination_reason_distribution": {
            reason: termination_counts.get(reason, 0)
            for reason in TERMINATION_REASONS
        },
    }


def _summarize_motion(values: np.ndarray) -> dict:
    if not len(values):
        return {
            "count": 0,
            "mean": None,
            "std": None,
            "min": None,
            "max": None,
            "percentiles": {
                f"P{percentile}": None for percentile in MOTION_PERCENTILES
            },
        }
    percentiles = np.percentile(values, MOTION_PERCENTILES)
    return {
        "count": len(values),
        "mean": float(np.mean(values)),
        "std": float(np.std(values)),
        "min": float(np.min(values)),
        "max": float(np.max(values)),
        "percentiles": {
            f"P{percentile}": float(value)
            for percentile, value in zip(MOTION_PERCENTILES, percentiles)
        },
    }


def _sum_file_sizes(paths: set[Path]) -> int:
    return sum(path.stat().st_size for path in paths)


def _read_json(path: Path) -> dict:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"could not read {path}: {error}") from error
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object in {path}")
    return value


def _normalize_dataset_index_path(path: Path) -> Path:
    path = path.expanduser().resolve()
    return path / "all_clips.json" if path.is_dir() else path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Summarize a global clip dataset and validated splits."
    )
    parser.add_argument(
        "dataset_index",
        type=Path,
        help="all_clips.json or its containing dataset directory",
    )
    parser.add_argument(
        "--splits",
        type=Path,
        help="directory containing train.json, val.json, and test.json",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    try:
        summary = summarize_dataset(args.dataset_index, args.splits)
    except ValueError as error:
        raise SystemExit(str(error)) from error
    print(json.dumps(summary, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
