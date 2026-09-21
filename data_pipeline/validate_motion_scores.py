#!/usr/bin/env python3
"""Validate Milestone 3A source-PNG motion statistics."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

try:
    from .compute_motion_scores import (
        METRIC_NAME,
        METRIC_NORMALIZATION,
        METRIC_SOURCE,
        METRIC_VERSION,
        MOTION_METADATA_KEY,
        compute_pairwise_motion,
        compute_preserved_metadata_sha256,
        summarize_pairwise_motion,
    )
    from .validate_clip_index import validate_clip_index
except ImportError:
    from compute_motion_scores import (
        METRIC_NAME,
        METRIC_NORMALIZATION,
        METRIC_SOURCE,
        METRIC_VERSION,
        MOTION_METADATA_KEY,
        compute_pairwise_motion,
        compute_preserved_metadata_sha256,
        summarize_pairwise_motion,
    )
    from validate_clip_index import validate_clip_index


MOTION_TOLERANCE_RELATIVE = 1e-7
MOTION_TOLERANCE_ABSOLUTE = 1e-9


def validate_motion_scores(clip_index_path: Path) -> list[str]:
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

    _validate_motion_metadata(clip_index, errors)
    for clip in clip_index["clips"]:
        _validate_clip_motion(
            clip,
            source_episode,
            observations,
            errors,
        )
    return errors


def _validate_motion_metadata(clip_index: dict, errors: list[str]) -> None:
    metadata = clip_index.get(MOTION_METADATA_KEY)
    if not isinstance(metadata, dict):
        errors.append(f"{MOTION_METADATA_KEY} must be an object")
        return

    expected_values = {
        "metric_name": METRIC_NAME,
        "metric_version": METRIC_VERSION,
        "source": METRIC_SOURCE,
        "normalization": METRIC_NORMALIZATION,
    }
    for field, expected in expected_values.items():
        if metadata.get(field) != expected:
            errors.append(
                f"{MOTION_METADATA_KEY}.{field} must equal {expected!r}"
            )

    expected_keys = set(expected_values) | {"preserved_metadata_sha256"}
    if set(metadata) != expected_keys:
        errors.append(f"{MOTION_METADATA_KEY} has missing or unexpected fields")

    stored_digest = metadata.get("preserved_metadata_sha256")
    actual_digest = compute_preserved_metadata_sha256(clip_index)
    if stored_digest != actual_digest:
        errors.append("non-motion clip/index metadata changed after scoring")


def _validate_clip_motion(
    clip: dict,
    source_episode: Path,
    observations: dict[int, dict],
    errors: list[str],
) -> None:
    clip_id = clip["clip_id"]
    pairwise = clip.get("motion_pairwise")
    expected_count = clip["num_frames"] - 1
    if not isinstance(pairwise, list):
        errors.append(f"clip {clip_id} motion_pairwise must be a list")
        return
    if len(pairwise) != expected_count:
        errors.append(
            f"clip {clip_id} has {len(pairwise)} pairwise scores, "
            f"expected {expected_count}"
        )

    pairwise_values = []
    for position, value in enumerate(pairwise):
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
            or value < 0
            or value > 1
        ):
            errors.append(
                f"clip {clip_id} motion pair {position} must be finite in [0, 1]"
            )
        else:
            pairwise_values.append(float(value))

    aggregate_fields = (
        "motion_mean",
        "motion_std",
        "motion_min",
        "motion_max",
    )
    aggregates_valid = True
    for field in aggregate_fields:
        value = clip.get(field)
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
            or value < 0
        ):
            errors.append(f"clip {clip_id} {field} must be finite and non-negative")
            aggregates_valid = False

    if len(pairwise_values) == len(pairwise) and pairwise_values:
        expected_statistics = summarize_pairwise_motion(pairwise_values)
        if aggregates_valid:
            for field in aggregate_fields:
                if not _motion_close(clip[field], expected_statistics[field]):
                    errors.append(
                        f"clip {clip_id} {field} disagrees with motion_pairwise"
                    )

    try:
        source_pairwise = compute_pairwise_motion(
            source_episode,
            observations,
            clip["observation_indices"],
        )
    except ValueError as error:
        errors.append(f"clip {clip_id} source RGB error: {error}")
        return

    if len(pairwise) == len(source_pairwise) and all(
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(value)
        and _motion_close(float(value), expected)
        for value, expected in zip(pairwise, source_pairwise)
    ):
        return
    errors.append(f"clip {clip_id} motion_pairwise disagrees with source RGB")


def _motion_close(actual: float, expected: float) -> bool:
    return math.isclose(
        actual,
        expected,
        rel_tol=MOTION_TOLERANCE_RELATIVE,
        abs_tol=MOTION_TOLERANCE_ABSOLUTE,
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
        errors = validate_motion_scores(normalized_path)
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
