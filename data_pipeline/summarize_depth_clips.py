#!/usr/bin/env python3
"""Summarize validated Milestone 2C depth frame sequences."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from PIL import Image

try:
    from .validate_depth_clips import validate_depth_clips
except ImportError:
    from validate_depth_clips import validate_depth_clips


def summarize_depth_clips(clip_index_path: Path) -> dict:
    clip_index_path = _normalize_index_path(clip_index_path)
    errors = validate_depth_clips(clip_index_path)
    if errors:
        details = "\n".join(f"  - {error}" for error in errors)
        raise ValueError(f"invalid depth clips {clip_index_path}:\n{details}")

    clip_index = json.loads(clip_index_path.read_text(encoding="utf-8"))
    clips = clip_index["clips"]
    complete_clips = [
        clip for clip in clips if clip.get("depth_frames") is not None
    ]
    depth_paths = [
        clip_index_path.parent / relative_path
        for clip in complete_clips
        for relative_path in clip["depth_frames"]
    ]
    resolutions = []
    for depth_path in depth_paths:
        with Image.open(depth_path) as image:
            resolutions.append(list(image.size))

    total_clips = len(clips)
    return {
        "number_of_total_clips": total_clips,
        "number_of_clips_with_complete_depth": len(complete_clips),
        "percentage_with_complete_depth": (
            100.0 * len(complete_clips) / total_clips
            if total_clips
            else 0.0
        ),
        "number_of_depth_frames": len(depth_paths),
        "resolution": _common_value(resolutions),
        "storage_size_bytes": sum(path.stat().st_size for path in depth_paths),
        "all_validated_depth_frames_pixel_identical_to_sources": True,
    }


def _common_value(values: list):
    if not values:
        return None
    first = values[0]
    return first if all(value == first for value in values) else values


def _normalize_index_path(path: Path) -> Path:
    return path / "clips.json" if path.is_dir() else path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("clip_index", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    try:
        summary = summarize_depth_clips(args.clip_index)
    except ValueError as error:
        raise SystemExit(str(error)) from error
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
