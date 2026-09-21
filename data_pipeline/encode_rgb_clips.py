#!/usr/bin/env python3
"""Encode indexed RGB observation windows as H.264 MP4 clips."""

from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import subprocess
import tempfile
from pathlib import Path

from PIL import Image

try:
    from .validate_clip_index import validate_clip_index
except ImportError:
    from validate_clip_index import validate_clip_index


DEFAULT_FPS = 8.0


def encode_rgb_clips(clip_index_path: Path, fps: float = DEFAULT_FPS) -> Path:
    clip_index_path = _normalize_index_path(clip_index_path)
    if (
        isinstance(fps, bool)
        or not isinstance(fps, (int, float))
        or not math.isfinite(fps)
        or fps <= 0
    ):
        raise ValueError("fps must be a positive finite number")
    fps = float(fps)

    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise ValueError(
            "ffmpeg was not found on PATH; install it and verify with "
            "`ffmpeg -version`"
        )

    errors = validate_clip_index(clip_index_path)
    if errors:
        details = "\n".join(f"  - {error}" for error in errors)
        raise ValueError(f"invalid clip index {clip_index_path}:\n{details}")

    clip_index = json.loads(clip_index_path.read_text(encoding="utf-8"))
    source_episode = Path(clip_index["source_episode"])
    source_metadata = json.loads(
        (source_episode / "metadata.json").read_text(encoding="utf-8")
    )
    observations = {
        item["observation_index"]: item
        for item in source_metadata["observations"]
    }

    rgb_dir = clip_index_path.parent / "rgb"
    rgb_dir.mkdir(parents=True, exist_ok=True)
    old_videos = [
        path
        for path in rgb_dir.rglob("*")
        if (path.is_file() or path.is_symlink())
        and path.suffix.lower() == ".mp4"
    ]

    serialized_fps: int | float = int(fps) if fps.is_integer() else fps
    staged_outputs: list[tuple[Path, Path]] = []
    dimension_cache: dict[Path, tuple[int, int]] = {}

    with tempfile.TemporaryDirectory(
        prefix=".rgb_staging_",
        dir=rgb_dir,
    ) as staging_directory:
        staging_dir = Path(staging_directory)
        for clip in clip_index["clips"]:
            clip_id = clip["clip_id"]
            frame_paths = _resolve_frame_paths(
                source_episode,
                observations,
                clip["observation_indices"],
            )
            _check_frame_dimensions(frame_paths, dimension_cache, clip_id)

            filename = f"clip_{clip_id:06d}.mp4"
            staged_video = staging_dir / filename
            with tempfile.TemporaryDirectory(
                prefix=f"clip_{clip_id:06d}_frames_",
                dir=staging_dir,
            ) as frame_directory:
                frame_dir = Path(frame_directory)
                for frame_number, source_path in enumerate(frame_paths):
                    link_path = frame_dir / f"{frame_number:06d}.png"
                    link_path.symlink_to(source_path)

                _encode_sequence(
                    ffmpeg=ffmpeg,
                    frame_pattern=frame_dir / "%06d.png",
                    output_path=staged_video,
                    fps=fps,
                    num_frames=clip["num_frames"],
                )

            final_video = rgb_dir / filename
            staged_outputs.append((staged_video, final_video))
            clip["rgb_video"] = f"rgb/{filename}"
            clip["fps"] = serialized_fps

        for old_video in old_videos:
            old_video.unlink()
        for staged_video, final_video in staged_outputs:
            os.replace(staged_video, final_video)

    _write_json_atomic(clip_index_path, clip_index)
    return clip_index_path


def _resolve_frame_paths(
    source_episode: Path,
    observations: dict[int, dict],
    observation_indices: list[int],
) -> list[Path]:
    frame_paths = []
    for observation_index in observation_indices:
        observation = observations.get(observation_index)
        if observation is None:
            raise ValueError(f"missing source observation {observation_index}")
        rgb_value = observation.get("rgb")
        if not isinstance(rgb_value, str):
            raise ValueError(
                f"source observation {observation_index} has no RGB path"
            )
        relative_path = Path(rgb_value)
        if relative_path.is_absolute() or ".." in relative_path.parts:
            raise ValueError(
                f"source observation {observation_index} has an unsafe RGB path"
            )
        frame_path = (source_episode / relative_path).resolve()
        if not frame_path.is_file():
            raise ValueError(
                f"source observation {observation_index} RGB file is missing: "
                f"{frame_path}"
            )
        frame_paths.append(frame_path)
    return frame_paths


def _check_frame_dimensions(
    frame_paths: list[Path],
    dimension_cache: dict[Path, tuple[int, int]],
    clip_id: int,
) -> None:
    sizes = set()
    for frame_path in frame_paths:
        if frame_path not in dimension_cache:
            try:
                with Image.open(frame_path) as image:
                    dimension_cache[frame_path] = image.size
            except (OSError, ValueError) as error:
                raise ValueError(f"could not read RGB frame {frame_path}: {error}") from error
        sizes.add(dimension_cache[frame_path])

    if len(sizes) != 1:
        raise ValueError(f"clip {clip_id} source RGB dimensions are inconsistent")
    width, height = next(iter(sizes))
    if width % 2 or height % 2:
        raise ValueError(
            f"clip {clip_id} has {width}x{height} RGB frames; yuv420p requires "
            "even width and height"
        )


def _encode_sequence(
    ffmpeg: str,
    frame_pattern: Path,
    output_path: Path,
    fps: float,
    num_frames: int,
) -> None:
    command = [
        ffmpeg,
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-framerate",
        f"{fps:g}",
        "-start_number",
        "0",
        "-i",
        str(frame_pattern),
        "-frames:v",
        str(num_frames),
        "-an",
        "-c:v",
        "libx264",
        "-preset",
        "medium",
        "-crf",
        "18",
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        str(output_path),
    ]
    result = subprocess.run(command, capture_output=True, text=True)
    if result.returncode != 0:
        message = result.stderr.strip() or "ffmpeg returned no error details"
        raise ValueError(f"ffmpeg failed for {output_path.name}: {message}")


def _write_json_atomic(path: Path, value: dict) -> None:
    temporary_path = path.with_name(f".{path.name}.tmp")
    try:
        temporary_path.write_text(
            json.dumps(value, indent=2) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary_path, path)
    finally:
        if temporary_path.exists():
            temporary_path.unlink()


def _normalize_index_path(path: Path) -> Path:
    return path / "clips.json" if path.is_dir() else path


def _positive_fps(value: str) -> float:
    try:
        fps = float(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("fps must be a number") from error
    if not math.isfinite(fps) or fps <= 0:
        raise argparse.ArgumentTypeError("fps must be a positive finite number")
    return fps


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("clip_index", type=Path)
    parser.add_argument("--fps", type=_positive_fps, default=DEFAULT_FPS)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    try:
        output_path = encode_rgb_clips(args.clip_index, args.fps)
    except ValueError as error:
        raise SystemExit(str(error)) from error

    clip_index = json.loads(output_path.read_text(encoding="utf-8"))
    print(
        f"ENCODED {clip_index['num_clips']} RGB clips at {args.fps:g} fps "
        f"under {output_path.parent / 'rgb'}"
    )


if __name__ == "__main__":
    main()
