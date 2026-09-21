#!/usr/bin/env python3
"""Compute source-PNG frame-difference motion statistics for indexed clips."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np
from PIL import Image

try:
    from .validate_clip_index import validate_clip_index
except ImportError:
    from validate_clip_index import validate_clip_index


MOTION_FIELDS = (
    "motion_mean",
    "motion_std",
    "motion_max",
    "motion_min",
    "motion_pairwise",
)
MOTION_METADATA_KEY = "motion_statistics"
METRIC_NAME = "mean_absolute_rgb_frame_difference"
METRIC_VERSION = 1
METRIC_SOURCE = "original_rgb_png"
METRIC_NORMALIZATION = [0.0, 1.0]
MOTION_HASH_EXCLUDED_INDEX_FIELDS = {
    MOTION_METADATA_KEY,
    "action_statistics_metadata",
}
MOTION_HASH_EXCLUDED_CLIP_FIELDS = set(MOTION_FIELDS) | {
    "action_statistics",
}


def compute_motion_scores(clip_index_path: Path) -> Path:
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
    source_episode = Path(clip_index["source_episode"])
    source_metadata = json.loads(
        (source_episode / "metadata.json").read_text(encoding="utf-8")
    )
    observations = {
        item["observation_index"]: item
        for item in source_metadata["observations"]
    }

    for clip in clip_index["clips"]:
        pairwise = compute_pairwise_motion(
            source_episode,
            observations,
            clip["observation_indices"],
        )
        if len(pairwise) != clip["num_frames"] - 1:
            raise ValueError(
                f"clip {clip['clip_id']} produced {len(pairwise)} motion pairs, "
                f"expected {clip['num_frames'] - 1}"
            )
        statistics = summarize_pairwise_motion(pairwise)
        clip.update(statistics)
        clip["motion_pairwise"] = pairwise

    clip_index[MOTION_METADATA_KEY] = {
        "metric_name": METRIC_NAME,
        "metric_version": METRIC_VERSION,
        "source": METRIC_SOURCE,
        "normalization": METRIC_NORMALIZATION,
        "preserved_metadata_sha256": preserved_metadata_sha256,
    }

    if compute_preserved_metadata_sha256(clip_index) != preserved_metadata_sha256:
        raise RuntimeError("non-motion metadata changed during motion computation")

    _write_json_atomic(clip_index_path, clip_index)
    return clip_index_path


def compute_pairwise_motion(
    source_episode: Path,
    observations: dict[int, dict],
    observation_indices: list[int],
) -> list[float]:
    if len(observation_indices) < 2:
        return []

    previous = _load_rgb_frame(
        _resolve_rgb_path(
            source_episode,
            observations,
            observation_indices[0],
        )
    )
    pairwise = []
    for observation_index in observation_indices[1:]:
        current = _load_rgb_frame(
            _resolve_rgb_path(source_episode, observations, observation_index)
        )
        if current.shape != previous.shape:
            raise ValueError(
                "source RGB dimensions change within an indexed clip"
            )
        score = float(
            np.mean(np.abs(current - previous), dtype=np.float64)
        )
        pairwise.append(score)
        previous = current
    return pairwise


def summarize_pairwise_motion(pairwise: list[float]) -> dict[str, float]:
    if not pairwise:
        raise ValueError("motion statistics require at least one frame pair")
    values = np.asarray(pairwise, dtype=np.float64)
    return {
        "motion_mean": float(np.mean(values)),
        "motion_std": float(np.std(values)),
        "motion_max": float(np.max(values)),
        "motion_min": float(np.min(values)),
    }


def compute_preserved_metadata_sha256(clip_index: dict) -> str:
    preserved = {
        key: value
        for key, value in clip_index.items()
        if key not in {"clips", *MOTION_HASH_EXCLUDED_INDEX_FIELDS}
    }
    preserved["clips"] = [
        {
            key: value
            for key, value in clip.items()
            if key not in MOTION_HASH_EXCLUDED_CLIP_FIELDS
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


def _resolve_rgb_path(
    source_episode: Path,
    observations: dict[int, dict],
    observation_index: int,
) -> Path:
    observation = observations.get(observation_index)
    if observation is None:
        raise ValueError(f"missing source observation {observation_index}")
    rgb_value = observation.get("rgb")
    if not isinstance(rgb_value, str):
        raise ValueError(f"source observation {observation_index} has no RGB path")
    relative_path = Path(rgb_value)
    if relative_path.is_absolute() or ".." in relative_path.parts:
        raise ValueError(
            f"source observation {observation_index} has an unsafe RGB path"
        )
    rgb_path = source_episode / relative_path
    if not rgb_path.is_file():
        raise ValueError(
            f"source observation {observation_index} RGB file is missing: "
            f"{rgb_path}"
        )
    return rgb_path


def _load_rgb_frame(rgb_path: Path) -> np.ndarray:
    try:
        with Image.open(rgb_path) as image:
            if image.mode != "RGB":
                raise ValueError(
                    f"source RGB frame must use RGB mode, got {image.mode}"
                )
            frame = np.asarray(image, dtype=np.float32).copy()
    except OSError as error:
        raise ValueError(f"could not read source RGB frame {rgb_path}: {error}") from error
    frame /= np.float32(255.0)
    if not np.all(np.isfinite(frame)) or frame.min() < 0 or frame.max() > 1:
        raise ValueError(f"source RGB normalization failed for {rgb_path}")
    return frame


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
        output_path = compute_motion_scores(args.clip_index)
    except (ValueError, RuntimeError) as error:
        raise SystemExit(str(error)) from error

    clip_index = json.loads(output_path.read_text(encoding="utf-8"))
    print(
        f"COMPUTED motion statistics for {clip_index['num_clips']} clips in "
        f"{output_path}"
    )


if __name__ == "__main__":
    main()
