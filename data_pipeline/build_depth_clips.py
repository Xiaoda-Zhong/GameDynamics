#!/usr/bin/env python3
"""Build lossless depth PNG sequences for indexed gameplay clips."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import tempfile
from pathlib import Path

try:
    from .validate_clip_index import validate_clip_index
except ImportError:
    from validate_clip_index import validate_clip_index


def build_depth_clips(clip_index_path: Path) -> Path:
    clip_index_path = _normalize_index_path(clip_index_path)
    errors = validate_clip_index(
        clip_index_path,
        validate_source_depth_files=False,
    )
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

    depth_root = clip_index_path.parent / "depth"
    with tempfile.TemporaryDirectory(
        prefix=".depth_staging_",
        dir=clip_index_path.parent,
    ) as staging_directory:
        staged_depth_root = Path(staging_directory) / "depth"
        staged_depth_root.mkdir()

        for clip in clip_index["clips"]:
            clip_id = clip["clip_id"]
            source_frames = _resolve_depth_frames(
                source_episode,
                observations,
                clip["observation_indices"],
            )
            if source_frames is None:
                clip["depth_frames"] = None
                clip["num_depth_frames"] = 0
                continue

            clip_directory_name = f"clip_{clip_id:06d}"
            clip_directory = staged_depth_root / clip_directory_name
            clip_directory.mkdir()
            depth_frames = []
            for frame_number, source_path in enumerate(source_frames):
                filename = f"{frame_number:03d}.png"
                destination = clip_directory / filename
                shutil.copy2(source_path, destination)
                depth_frames.append(
                    f"depth/{clip_directory_name}/{filename}"
                )

            clip["depth_frames"] = depth_frames
            clip["num_depth_frames"] = len(depth_frames)

        _remove_existing_depth_root(depth_root)
        os.replace(staged_depth_root, depth_root)

    _write_json_atomic(clip_index_path, clip_index)
    return clip_index_path


def _resolve_depth_frames(
    source_episode: Path,
    observations: dict[int, dict],
    observation_indices: list[int],
) -> list[Path] | None:
    source_frames = []
    for observation_index in observation_indices:
        observation = observations.get(observation_index)
        if observation is None:
            raise ValueError(f"missing source observation {observation_index}")

        depth_value = observation.get("depth")
        if depth_value is None:
            return None
        if not isinstance(depth_value, str):
            raise ValueError(
                f"source observation {observation_index} has an invalid depth path"
            )

        relative_path = Path(depth_value)
        if relative_path.is_absolute() or ".." in relative_path.parts:
            raise ValueError(
                f"source observation {observation_index} has an unsafe depth path"
            )
        source_path = (source_episode / relative_path).resolve()
        if not source_path.is_file():
            return None
        source_frames.append(source_path)
    return source_frames


def _remove_existing_depth_root(depth_root: Path) -> None:
    if depth_root.is_symlink() or depth_root.is_file():
        depth_root.unlink()
    elif depth_root.is_dir():
        shutil.rmtree(depth_root)


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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("clip_index", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    try:
        output_path = build_depth_clips(args.clip_index)
    except ValueError as error:
        raise SystemExit(str(error)) from error

    clip_index = json.loads(output_path.read_text(encoding="utf-8"))
    complete = sum(
        clip.get("depth_frames") is not None
        for clip in clip_index["clips"]
    )
    print(
        f"BUILT depth sequences for {complete}/{clip_index['num_clips']} clips "
        f"under {output_path.parent / 'depth'}"
    )


if __name__ == "__main__":
    main()
