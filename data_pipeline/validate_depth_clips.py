#!/usr/bin/env python3
"""Validate lossless Milestone 2C depth frame sequences."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image

try:
    from .validate_clip_index import validate_clip_index
except ImportError:
    from validate_clip_index import validate_clip_index


def validate_depth_clips(clip_index_path: Path) -> list[str]:
    clip_index_path = _normalize_index_path(clip_index_path)
    errors = validate_clip_index(
        clip_index_path,
        validate_source_depth_files=False,
    )
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

    expected_files: set[str] = set()
    expected_directories: set[str] = set()
    source_cache: dict[Path, tuple[tuple[int, int], str, np.ndarray]] = {}

    for clip in clip_index["clips"]:
        clip_id = clip["clip_id"]
        source_paths = _resolve_source_depth_paths(
            source_episode,
            observations,
            clip["observation_indices"],
        )
        depth_frames = clip.get("depth_frames")
        num_depth_frames = clip.get("num_depth_frames")

        if depth_frames is None:
            if num_depth_frames != 0:
                errors.append(
                    f"clip {clip_id} with null depth_frames must have "
                    "num_depth_frames=0"
                )
            if source_paths is not None:
                errors.append(
                    f"clip {clip_id} has complete source depth but null "
                    "depth_frames"
                )
            continue

        if not isinstance(depth_frames, list):
            errors.append(f"clip {clip_id} depth_frames must be a list or null")
            continue

        for frame_number, declared_path in enumerate(depth_frames):
            if not isinstance(declared_path, str):
                errors.append(
                    f"clip {clip_id} depth frame entry {frame_number} "
                    "must be a path string"
                )
                continue
            relative_path = Path(declared_path)
            if relative_path.is_absolute() or ".." in relative_path.parts:
                errors.append(
                    f"clip {clip_id} depth frame entry {frame_number} "
                    "has an unsafe path"
                )
            elif not (clip_index_path.parent / relative_path).is_file():
                errors.append(
                    f"clip {clip_id} declared depth file is missing: "
                    f"{declared_path}"
                )

        expected_relative_paths = [
            f"depth/clip_{clip_id:06d}/{frame_number:03d}.png"
            for frame_number in range(clip["num_frames"])
        ]
        expected_directory = f"depth/clip_{clip_id:06d}"
        expected_directories.add(expected_directory)
        expected_files.update(expected_relative_paths)

        if depth_frames != expected_relative_paths:
            errors.append(
                f"clip {clip_id} depth filenames are missing, extra, or out of order"
            )
        if len(depth_frames) != clip["num_frames"]:
            errors.append(
                f"clip {clip_id} has {len(depth_frames)} declared depth frames, "
                f"expected {clip['num_frames']}"
            )
        if (
            type(num_depth_frames) is not int
            or num_depth_frames != clip["num_frames"]
        ):
            errors.append(
                f"clip {clip_id} num_depth_frames must equal num_frames"
            )
        if source_paths is None:
            errors.append(
                f"clip {clip_id} declares depth frames but source depth is incomplete"
            )
            continue

        for frame_number, (relative_path, source_path) in enumerate(
            zip(expected_relative_paths, source_paths)
        ):
            depth_path = clip_index_path.parent / relative_path
            if not depth_path.is_file():
                errors.append(
                    f"clip {clip_id} depth frame {frame_number:03d} is missing: "
                    f"{depth_path}"
                )
                continue
            _compare_depth_images(
                source_path,
                depth_path,
                source_cache,
                clip_id,
                frame_number,
                errors,
            )

    _check_for_stale_depth_entries(
        clip_index_path.parent / "depth",
        clip_index_path.parent,
        expected_files,
        expected_directories,
        errors,
    )
    return errors


def _resolve_source_depth_paths(
    source_episode: Path,
    observations: dict[int, dict],
    observation_indices: list[int],
) -> list[Path] | None:
    source_paths = []
    for observation_index in observation_indices:
        depth_value = observations[observation_index].get("depth")
        if depth_value is None:
            return None
        if not isinstance(depth_value, str):
            return None
        relative_path = Path(depth_value)
        if relative_path.is_absolute() or ".." in relative_path.parts:
            return None
        source_path = source_episode / relative_path
        if not source_path.is_file():
            return None
        source_paths.append(source_path)
    return source_paths


def _compare_depth_images(
    source_path: Path,
    depth_path: Path,
    source_cache: dict[Path, tuple[tuple[int, int], str, np.ndarray]],
    clip_id: int,
    frame_number: int,
    errors: list[str],
) -> None:
    try:
        if source_path not in source_cache:
            with Image.open(source_path) as source_image:
                source_cache[source_path] = (
                    source_image.size,
                    source_image.mode,
                    np.asarray(source_image).copy(),
                )
        source_size, source_mode, source_array = source_cache[source_path]
        with Image.open(depth_path) as depth_image:
            depth_size = depth_image.size
            depth_mode = depth_image.mode
            depth_array = np.asarray(depth_image).copy()
    except (OSError, ValueError) as error:
        errors.append(
            f"clip {clip_id} could not read depth frame {frame_number:03d}: {error}"
        )
        return

    if depth_size != source_size:
        errors.append(
            f"clip {clip_id} depth frame {frame_number:03d} dimensions "
            "do not match its source"
        )
    if (
        depth_mode != source_mode
        or depth_array.dtype != source_array.dtype
        or not np.array_equal(depth_array, source_array)
    ):
        errors.append(
            f"clip {clip_id} depth frame {frame_number:03d} pixels do not "
            "match its source observation"
        )


def _check_for_stale_depth_entries(
    depth_root: Path,
    clip_index_directory: Path,
    expected_files: set[str],
    expected_directories: set[str],
    errors: list[str],
) -> None:
    if not depth_root.is_dir():
        if expected_directories:
            errors.append(f"missing depth output directory: {depth_root}")
        return

    actual_files = {
        path.relative_to(clip_index_directory).as_posix()
        for path in depth_root.rglob("*")
        if path.is_file() or path.is_symlink()
    }
    actual_directories = {
        path.relative_to(clip_index_directory).as_posix()
        for path in depth_root.rglob("*")
        if path.is_dir() and not path.is_symlink()
    }

    stale_files = sorted(actual_files - expected_files)
    if stale_files:
        errors.append("stale depth files: " + ", ".join(stale_files))
    stale_directories = sorted(actual_directories - expected_directories)
    if stale_directories:
        errors.append(
            "stale depth directories: " + ", ".join(stale_directories)
        )
    missing_directories = sorted(expected_directories - actual_directories)
    if missing_directories:
        errors.append(
            "missing depth directories: " + ", ".join(missing_directories)
        )


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
        errors = validate_depth_clips(normalized_path)
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
