#!/usr/bin/env python3
"""Validate Milestone 3B per-clip action statistics."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

try:
    from .compute_action_statistics import (
        ACTION_METADATA_KEY,
        ACTION_STATISTICS_KEY,
        DOMINANT_ACTION_TIE_BREAK,
        ENTROPY_LOG_BASE,
        METRIC_NAME,
        METRIC_SOURCE,
        METRIC_VERSION,
        calculate_action_statistics,
        compute_preserved_metadata_sha256,
    )
    from .validate_clip_index import validate_clip_index
except ImportError:
    from compute_action_statistics import (
        ACTION_METADATA_KEY,
        ACTION_STATISTICS_KEY,
        DOMINANT_ACTION_TIE_BREAK,
        ENTROPY_LOG_BASE,
        METRIC_NAME,
        METRIC_SOURCE,
        METRIC_VERSION,
        calculate_action_statistics,
        compute_preserved_metadata_sha256,
    )
    from validate_clip_index import validate_clip_index


STATISTIC_FIELDS = {
    "action_counts",
    "action_fractions",
    "dominant_action",
    "dominant_action_fraction",
    "noop_fraction",
    "unique_action_count",
    "action_entropy",
}
NUMERIC_TOLERANCE_RELATIVE = 1e-9
NUMERIC_TOLERANCE_ABSOLUTE = 1e-12


def validate_action_statistics(clip_index_path: Path) -> list[str]:
    clip_index_path = _normalize_index_path(clip_index_path)
    errors = validate_clip_index(
        clip_index_path,
        validate_source_depth_files=False,
    )
    if errors:
        return errors

    clip_index = json.loads(clip_index_path.read_text(encoding="utf-8"))
    _validate_action_metadata(clip_index, errors)
    for clip in clip_index["clips"]:
        _validate_clip_statistics(clip, errors)
    return errors


def _validate_action_metadata(clip_index: dict, errors: list[str]) -> None:
    metadata = clip_index.get(ACTION_METADATA_KEY)
    if not isinstance(metadata, dict):
        errors.append(f"{ACTION_METADATA_KEY} must be an object")
        return

    expected_values = {
        "metric_name": METRIC_NAME,
        "metric_version": METRIC_VERSION,
        "source": METRIC_SOURCE,
        "entropy_log_base": ENTROPY_LOG_BASE,
        "dominant_action_tie_break": DOMINANT_ACTION_TIE_BREAK,
    }
    for field, expected in expected_values.items():
        if metadata.get(field) != expected:
            errors.append(
                f"{ACTION_METADATA_KEY}.{field} must equal {expected!r}"
            )

    expected_keys = set(expected_values) | {"preserved_metadata_sha256"}
    if set(metadata) != expected_keys:
        errors.append(f"{ACTION_METADATA_KEY} has missing or unexpected fields")

    stored_digest = metadata.get("preserved_metadata_sha256")
    actual_digest = compute_preserved_metadata_sha256(clip_index)
    if stored_digest != actual_digest:
        errors.append("non-action clip/index metadata changed after scoring")


def _validate_clip_statistics(clip: dict, errors: list[str]) -> None:
    clip_id = clip["clip_id"]
    statistics = clip.get(ACTION_STATISTICS_KEY)
    if not isinstance(statistics, dict):
        errors.append(f"clip {clip_id} action_statistics must be an object")
        return
    if set(statistics) != STATISTIC_FIELDS:
        errors.append(
            f"clip {clip_id} action_statistics has missing or unexpected fields"
        )

    expected = calculate_action_statistics(clip["actions"])
    counts = statistics.get("action_counts")
    if not isinstance(counts, dict):
        errors.append(f"clip {clip_id} action_counts must be an object")
    else:
        if any(
            not isinstance(action, str)
            or type(count) is not int
            or count <= 0
            for action, count in counts.items()
        ):
            errors.append(
                f"clip {clip_id} action_counts must contain positive integers"
            )
        if counts != expected["action_counts"]:
            errors.append(f"clip {clip_id} action_counts do not match raw actions")
        if sum(count for count in counts.values() if type(count) is int) != clip["num_actions"]:
            errors.append(f"clip {clip_id} action_counts do not sum to num_actions")

    fractions = statistics.get("action_fractions")
    fractions_valid = isinstance(fractions, dict)
    if not fractions_valid:
        errors.append(f"clip {clip_id} action_fractions must be an object")
    else:
        if set(fractions) != set(expected["action_fractions"]):
            errors.append(f"clip {clip_id} action_fractions have invalid actions")
            fractions_valid = False
        for action, value in fractions.items():
            if not _valid_probability(value):
                errors.append(
                    f"clip {clip_id} action fraction for {action!r} is invalid"
                )
                fractions_valid = False
        if fractions_valid:
            for action, expected_value in expected["action_fractions"].items():
                if not _numeric_close(fractions[action], expected_value):
                    errors.append(
                        f"clip {clip_id} action fraction for {action!r} "
                        "does not match its count"
                    )
            if not _numeric_close(sum(fractions.values()), 1.0):
                errors.append(f"clip {clip_id} action_fractions do not sum to 1")

    dominant_action = statistics.get("dominant_action")
    if dominant_action not in expected["action_counts"]:
        errors.append(f"clip {clip_id} dominant_action is not a clip action")
    elif dominant_action != expected["dominant_action"]:
        errors.append(
            f"clip {clip_id} dominant_action violates the documented tie-break"
        )

    _check_numeric_field(
        clip_id,
        statistics,
        expected,
        "dominant_action_fraction",
        errors,
    )
    _check_numeric_field(
        clip_id,
        statistics,
        expected,
        "noop_fraction",
        errors,
    )

    unique_action_count = statistics.get("unique_action_count")
    if (
        type(unique_action_count) is not int
        or unique_action_count != expected["unique_action_count"]
    ):
        errors.append(f"clip {clip_id} unique_action_count is incorrect")

    _check_numeric_field(
        clip_id,
        statistics,
        expected,
        "action_entropy",
        errors,
    )


def _check_numeric_field(
    clip_id: int,
    statistics: dict,
    expected: dict,
    field: str,
    errors: list[str],
) -> None:
    value = statistics.get(field)
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        or value < 0
    ):
        errors.append(f"clip {clip_id} {field} must be finite and non-negative")
    elif not _numeric_close(value, expected[field]):
        errors.append(f"clip {clip_id} {field} is incorrect")


def _valid_probability(value: object) -> bool:
    return (
        not isinstance(value, bool)
        and isinstance(value, (int, float))
        and math.isfinite(value)
        and 0 <= value <= 1
    )


def _numeric_close(actual: float, expected: float) -> bool:
    return math.isclose(
        actual,
        expected,
        rel_tol=NUMERIC_TOLERANCE_RELATIVE,
        abs_tol=NUMERIC_TOLERANCE_ABSOLUTE,
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
        errors = validate_action_statistics(normalized_path)
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
