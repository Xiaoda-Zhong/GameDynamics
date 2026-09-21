#!/usr/bin/env python3
"""Validate encoded Milestone 2B RGB MP4 clips."""

from __future__ import annotations

import argparse
import json
import math
import shutil
import subprocess
from fractions import Fraction
from pathlib import Path

from PIL import Image

try:
    from .validate_clip_index import validate_clip_index
except ImportError:
    from validate_clip_index import validate_clip_index


def validate_rgb_clips(clip_index_path: Path) -> list[str]:
    clip_index_path = _normalize_index_path(clip_index_path)
    errors = validate_clip_index(clip_index_path)
    if errors:
        return errors

    clip_index = json.loads(clip_index_path.read_text(encoding="utf-8"))
    source_episode = Path(clip_index["source_episode"])
    source_metadata = json.loads(
        (source_episode / "metadata.json").read_text(encoding="utf-8")
    )
    observations = {
        item["observation_index"]: item
        for item in source_metadata["observations"]
    }

    ffprobe = shutil.which("ffprobe")
    if ffprobe is None:
        errors.append(
            "ffprobe was not found on PATH; install ffmpeg and verify with "
            "`ffprobe -version`"
        )

    expected_paths: set[str] = set()
    declared_fps_values: set[float] = set()
    dimension_cache: dict[Path, tuple[int, int]] = {}

    for clip in clip_index["clips"]:
        clip_id = clip["clip_id"]
        expected_relative_path = f"rgb/clip_{clip_id:06d}.mp4"
        expected_paths.add(expected_relative_path)

        if clip.get("rgb_video") != expected_relative_path:
            errors.append(
                f"clip {clip_id} rgb_video must be {expected_relative_path}"
            )

        declared_fps = clip.get("fps")
        if (
            isinstance(declared_fps, bool)
            or not isinstance(declared_fps, (int, float))
            or not math.isfinite(declared_fps)
            or declared_fps <= 0
        ):
            errors.append(f"clip {clip_id} fps must be a positive finite number")
            declared_fps = None
        else:
            declared_fps = float(declared_fps)
            declared_fps_values.add(declared_fps)

        video_path = clip_index_path.parent / expected_relative_path
        if not video_path.is_file():
            errors.append(f"clip {clip_id} RGB video is missing: {video_path}")
            continue

        source_sizes = _source_dimensions(
            source_episode,
            observations,
            clip["observation_indices"],
            dimension_cache,
            clip_id,
            errors,
        )
        if ffprobe is None:
            continue

        try:
            video = probe_video(video_path, ffprobe)
        except ValueError as error:
            errors.append(f"clip {clip_id}: {error}")
            continue

        if video["frame_count"] != clip["num_frames"]:
            errors.append(
                f"clip {clip_id} has {video['frame_count']} encoded frames, "
                f"expected {clip['num_frames']}"
            )
        if declared_fps is not None:
            tolerance = max(0.001, declared_fps * 0.0001)
            if abs(video["fps"] - declared_fps) > tolerance:
                errors.append(
                    f"clip {clip_id} encoded fps {video['fps']:.6g} does not "
                    f"match declared fps {declared_fps:.6g}"
                )
        if source_sizes and (video["width"], video["height"]) not in source_sizes:
            errors.append(
                f"clip {clip_id} resolution {video['width']}x{video['height']} "
                f"does not match source RGB frames"
            )
        if video["codec_name"] != "h264":
            errors.append(f"clip {clip_id} codec is not H.264")
        if video["pixel_format"] != "yuv420p":
            errors.append(f"clip {clip_id} pixel format is not yuv420p")

    if len(declared_fps_values) > 1:
        errors.append("clips declare inconsistent fps values")

    rgb_dir = clip_index_path.parent / "rgb"
    actual_paths = (
        {
            path.relative_to(clip_index_path.parent).as_posix()
            for path in rgb_dir.rglob("*")
            if path.is_file() and path.suffix.lower() == ".mp4"
        }
        if rgb_dir.is_dir()
        else set()
    )
    stale_paths = sorted(actual_paths - expected_paths)
    if stale_paths:
        errors.append("stale RGB videos: " + ", ".join(stale_paths))
    return errors


def probe_video(video_path: Path, ffprobe: str | None = None) -> dict:
    ffprobe = ffprobe or shutil.which("ffprobe")
    if ffprobe is None:
        raise ValueError("ffprobe was not found on PATH")

    command = [
        ffprobe,
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-count_frames",
        "-show_entries",
        "stream=codec_name,pix_fmt,width,height,avg_frame_rate,nb_read_frames,duration",
        "-of",
        "json",
        str(video_path),
    ]
    result = subprocess.run(command, capture_output=True, text=True)
    if result.returncode != 0:
        message = result.stderr.strip() or "ffprobe returned no error details"
        raise ValueError(f"ffprobe failed for {video_path}: {message}")

    try:
        payload = json.loads(result.stdout)
        streams = payload["streams"]
    except (json.JSONDecodeError, KeyError, TypeError) as error:
        raise ValueError(f"ffprobe returned invalid JSON for {video_path}") from error
    if len(streams) != 1:
        raise ValueError(
            f"expected one video stream in {video_path}, got {len(streams)}"
        )

    stream = streams[0]
    try:
        frame_count = int(stream["nb_read_frames"])
        fps = float(Fraction(stream["avg_frame_rate"]))
        width = int(stream["width"])
        height = int(stream["height"])
        duration = float(stream["duration"])
    except (KeyError, TypeError, ValueError, ZeroDivisionError) as error:
        raise ValueError(
            f"ffprobe returned incomplete video metadata for {video_path}"
        ) from error
    if frame_count < 0 or not math.isfinite(fps) or fps <= 0:
        raise ValueError(f"ffprobe returned invalid frame timing for {video_path}")

    return {
        "frame_count": frame_count,
        "fps": fps,
        "width": width,
        "height": height,
        "duration": duration,
        "codec_name": stream.get("codec_name"),
        "pixel_format": stream.get("pix_fmt"),
    }


def _source_dimensions(
    source_episode: Path,
    observations: dict[int, dict],
    observation_indices: list[int],
    dimension_cache: dict[Path, tuple[int, int]],
    clip_id: int,
    errors: list[str],
) -> set[tuple[int, int]]:
    sizes = set()
    for observation_index in observation_indices:
        rgb_value = observations[observation_index]["rgb"]
        frame_path = source_episode / rgb_value
        if frame_path not in dimension_cache:
            try:
                with Image.open(frame_path) as image:
                    dimension_cache[frame_path] = image.size
            except (OSError, ValueError) as error:
                errors.append(
                    f"clip {clip_id} could not read source RGB frame "
                    f"{frame_path}: {error}"
                )
                continue
        sizes.add(dimension_cache[frame_path])
    if len(sizes) > 1:
        errors.append(f"clip {clip_id} source RGB dimensions are inconsistent")
    return sizes


def _normalize_index_path(path: Path) -> Path:
    return path / "clips.json" if path.is_dir() else path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("clip_indices", nargs="+", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    invalid = False
    for clip_index_path in args.clip_indices:
        normalized_path = _normalize_index_path(clip_index_path)
        errors = validate_rgb_clips(normalized_path)
        if errors:
            invalid = True
            print(f"INVALID {normalized_path}")
            for error in errors:
                print(f"  - {error}")
        else:
            print(f"VALID {normalized_path}")
    if invalid:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
