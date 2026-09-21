#!/usr/bin/env python3
"""Summarize a validated Milestone 2A clip index."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

try:
    from .validate_clip_index import validate_clip_index
except ImportError:
    from validate_clip_index import validate_clip_index


def summarize_clip_index(clip_index_path: Path) -> dict:
    if clip_index_path.is_dir():
        clip_index_path = clip_index_path / "clips.json"
    errors = validate_clip_index(clip_index_path)
    if errors:
        details = "\n".join(f"  - {error}" for error in errors)
        raise ValueError(f"invalid clip index {clip_index_path}:\n{details}")

    clip_index = json.loads(clip_index_path.read_text(encoding="utf-8"))
    clips = clip_index["clips"]
    action_counts = Counter(
        action for clip in clips for action in clip["actions"]
    )

    return {
        "scenario_name": clip_index["scenario_name"],
        "source_episode": clip_index["source_episode"],
        "num_clips": len(clips),
        "clip_length": clip_index["clip_length"],
        "stride": clip_index["stride"],
        "first_clip_range": _clip_range(clips[0]) if clips else None,
        "last_clip_range": _clip_range(clips[-1]) if clips else None,
        "action_frequency_across_clips": dict(sorted(action_counts.items())),
    }


def _clip_range(clip: dict) -> list[int]:
    return [
        clip["start_observation_index"],
        clip["end_observation_index"],
    ]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("clip_index", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    try:
        summary = summarize_clip_index(args.clip_index)
    except ValueError as error:
        raise SystemExit(str(error)) from error
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
