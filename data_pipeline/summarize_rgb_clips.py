#!/usr/bin/env python3
"""Summarize validated Milestone 2B RGB MP4 clips."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

try:
    from .validate_rgb_clips import probe_video, validate_rgb_clips
except ImportError:
    from validate_rgb_clips import probe_video, validate_rgb_clips


def summarize_rgb_clips(clip_index_path: Path) -> dict:
    clip_index_path = _normalize_index_path(clip_index_path)
    errors = validate_rgb_clips(clip_index_path)
    if errors:
        details = "\n".join(f"  - {error}" for error in errors)
        raise ValueError(f"invalid RGB clips {clip_index_path}:\n{details}")

    ffprobe = shutil.which("ffprobe")
    if ffprobe is None:
        raise ValueError("ffprobe was not found on PATH")

    clip_index = json.loads(clip_index_path.read_text(encoding="utf-8"))
    video_paths = [
        clip_index_path.parent / clip["rgb_video"]
        for clip in clip_index["clips"]
    ]
    video_metadata = [probe_video(path, ffprobe) for path in video_paths]
    sizes = [path.stat().st_size for path in video_paths]

    return {
        "number_of_rgb_videos": len(video_paths),
        "total_size_bytes": sum(sizes),
        "average_file_size_bytes": (
            sum(sizes) / len(sizes) if sizes else None
        ),
        "fps": _common_value([item["fps"] for item in video_metadata]),
        "frame_count_per_video": _common_value(
            [item["frame_count"] for item in video_metadata]
        ),
        "resolution": _common_value(
            [[item["width"], item["height"]] for item in video_metadata]
        ),
        "total_video_duration_seconds": sum(
            item["duration"] for item in video_metadata
        ),
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
        summary = summarize_rgb_clips(args.clip_index)
    except ValueError as error:
        raise SystemExit(str(error)) from error
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
