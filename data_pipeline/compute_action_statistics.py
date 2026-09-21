#!/usr/bin/env python3
"""Compute categorical action statistics for indexed gameplay clips."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from collections import Counter
from pathlib import Path

try:
    from .validate_clip_index import validate_clip_index
except ImportError:
    from validate_clip_index import validate_clip_index


ACTION_STATISTICS_KEY = "action_statistics"
ACTION_METADATA_KEY = "action_statistics_metadata"
METRIC_NAME = "categorical_action_distribution"
METRIC_VERSION = 1
METRIC_SOURCE = "clip_actions"
ENTROPY_LOG_BASE = "e"
DOMINANT_ACTION_TIE_BREAK = "lexicographically_smallest_action_label"


def compute_action_statistics(clip_index_path: Path) -> Path:
    clip_index_path = _normalize_index_path(clip_index_path)
    errors = validate_clip_index(
        clip_index_path,
        validate_source_depth_files=False,
    )
    if errors:
        details = "\n".join(f"  - {error}" for error in errors)
        raise ValueError(f"invalid clip index {clip_index_path}:\n{details}")

    clip_index = json.loads(clip_index_path.read_text(encoding="utf-8"))
    preserved_metadata_sha256 = compute_preserved_metadata_sha256(clip_index)

    for clip in clip_index["clips"]:
        statistics = calculate_action_statistics(clip["actions"])
        if sum(statistics["action_counts"].values()) != clip["num_actions"]:
            raise ValueError(
                f"clip {clip['clip_id']} action count does not equal num_actions"
            )
        clip[ACTION_STATISTICS_KEY] = statistics

    clip_index[ACTION_METADATA_KEY] = {
        "metric_name": METRIC_NAME,
        "metric_version": METRIC_VERSION,
        "source": METRIC_SOURCE,
        "entropy_log_base": ENTROPY_LOG_BASE,
        "dominant_action_tie_break": DOMINANT_ACTION_TIE_BREAK,
        "preserved_metadata_sha256": preserved_metadata_sha256,
    }

    if compute_preserved_metadata_sha256(clip_index) != preserved_metadata_sha256:
        raise RuntimeError("non-action metadata changed during action computation")

    _write_json_atomic(clip_index_path, clip_index)
    return clip_index_path


def calculate_action_statistics(actions: list[str]) -> dict:
    if not actions:
        raise ValueError("action statistics require at least one action")

    counts = Counter(actions)
    ordered_counts = {
        action: counts[action]
        for action in sorted(counts)
    }
    num_actions = len(actions)
    fractions = {
        action: count / num_actions
        for action, count in ordered_counts.items()
    }
    maximum_count = max(ordered_counts.values())
    dominant_action = min(
        action
        for action, count in ordered_counts.items()
        if count == maximum_count
    )
    entropy = -sum(
        probability * math.log(probability)
        for probability in fractions.values()
        if probability > 0
    )
    if entropy == 0.0:
        entropy = 0.0

    return {
        "action_counts": ordered_counts,
        "action_fractions": fractions,
        "dominant_action": dominant_action,
        "dominant_action_fraction": fractions[dominant_action],
        "noop_fraction": fractions.get("NOOP", 0.0),
        "unique_action_count": len(ordered_counts),
        "action_entropy": entropy,
    }


def compute_preserved_metadata_sha256(clip_index: dict) -> str:
    preserved = {
        key: value
        for key, value in clip_index.items()
        if key not in {"clips", ACTION_METADATA_KEY}
    }
    preserved["clips"] = [
        {
            key: value
            for key, value in clip.items()
            if key != ACTION_STATISTICS_KEY
        }
        for clip in clip_index.get("clips", [])
    ]
    payload = json.dumps(
        preserved,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _write_json_atomic(path: Path, value: dict) -> None:
    temporary_path = path.with_name(f".{path.name}.tmp")
    try:
        temporary_path.write_text(
            json.dumps(value, indent=2, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary_path, path)
    finally:
        if temporary_path.exists():
            temporary_path.unlink()


def _normalize_index_path(path: Path) -> Path:
    return path / "clips.json" if path.is_dir() else path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("clip_index", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    try:
        output_path = compute_action_statistics(args.clip_index)
    except (ValueError, RuntimeError) as error:
        raise SystemExit(str(error)) from error

    clip_index = json.loads(output_path.read_text(encoding="utf-8"))
    print(
        f"COMPUTED action statistics for {clip_index['num_clips']} clips in "
        f"{output_path}"
    )


if __name__ == "__main__":
    main()
