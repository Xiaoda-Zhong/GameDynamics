#!/usr/bin/env python3
"""Validate a fixed-budget global or legacy subset manifest."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

try:
    from .select_dataset_subsets import (
        GLOBAL_MANIFEST_VERSION,
        MANIFEST_VERSION,
        SOURCE_KIND_GLOBAL_TRAIN,
        STRATEGIES,
        _file_sha256,
        _global_train_source_errors,
        _load_action_labels,
        _load_json_object,
        build_subset_manifest,
        select_clip_ids,
    )
    from .validate_action_statistics import validate_action_statistics
    from .validate_motion_scores import validate_motion_scores
except ImportError:
    from select_dataset_subsets import (
        GLOBAL_MANIFEST_VERSION,
        MANIFEST_VERSION,
        SOURCE_KIND_GLOBAL_TRAIN,
        STRATEGIES,
        _file_sha256,
        _global_train_source_errors,
        _load_action_labels,
        _load_json_object,
        build_subset_manifest,
        select_clip_ids,
    )
    from validate_action_statistics import validate_action_statistics
    from validate_motion_scores import validate_motion_scores


NUMERIC_TOLERANCE_RELATIVE = 1e-9
NUMERIC_TOLERANCE_ABSOLUTE = 1e-12


def validate_dataset_subset(manifest_path: Path) -> list[str]:
    errors: list[str] = []
    if not manifest_path.is_file():
        return [f"missing subset manifest: {manifest_path}"]
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        return [f"could not read subset manifest: {error}"]
    if not isinstance(manifest, dict):
        return ["subset manifest must be an object"]

    if manifest.get("source_kind") == SOURCE_KIND_GLOBAL_TRAIN or (
        "source_train_manifest" in manifest
    ):
        return _validate_global_dataset_subset(manifest, manifest_path)

    source_value = manifest.get("source_clip_index")
    if not isinstance(source_value, str) or not source_value:
        return ["source_clip_index must be a non-empty path string"]
    source_path = Path(source_value)
    if not source_path.is_file():
        return [f"source clip index is missing: {source_path}"]

    motion_errors = validate_motion_scores(source_path)
    action_errors = validate_action_statistics(source_path)
    if motion_errors or action_errors:
        errors.append("source clip index has invalid statistics")
        errors.extend(f"source motion: {error}" for error in motion_errors)
        errors.extend(f"source action: {error}" for error in action_errors)
        return errors

    source = json.loads(source_path.read_text(encoding="utf-8"))
    clips = source["clips"]
    clips_by_id = {clip["clip_id"]: clip for clip in clips}
    action_labels = _load_action_labels(source)

    manifest_version = manifest.get("manifest_version")
    if type(manifest_version) is not int or manifest_version != MANIFEST_VERSION:
        errors.append(f"manifest_version must be {MANIFEST_VERSION}")
    if source_value != str(source_path.resolve()):
        errors.append("source_clip_index must be an absolute resolved path")
    if manifest.get("source_clip_index_sha256") != _file_sha256(source_path):
        errors.append("source_clip_index_sha256 does not match the source index")
    if manifest.get("source_episode") != source["source_episode"]:
        errors.append("source_episode does not match the source index")
    if manifest.get("scenario_name") != source["scenario_name"]:
        errors.append("scenario_name does not match the source index")
    available_count = manifest.get("num_available_clips")
    if type(available_count) is not int or available_count != len(clips):
        errors.append("num_available_clips does not match the source index")
    if manifest.get("declared_action_space") != action_labels:
        errors.append("declared_action_space does not match the source episode")

    strategy = manifest.get("strategy")
    if strategy not in STRATEGIES:
        errors.append(f"strategy must be one of: {', '.join(STRATEGIES)}")
        return errors
    budget = manifest.get("budget")
    if type(budget) is not int or budget < 1:
        errors.append("budget must be a positive integer")
        return errors
    seed = manifest.get("seed")
    if type(seed) is not int:
        errors.append("seed must be an integer")
        return errors

    selected_ids = manifest.get("selected_clip_ids")
    if not isinstance(selected_ids, list):
        errors.append("selected_clip_ids must be a list")
        return errors
    if any(type(clip_id) is not int for clip_id in selected_ids):
        errors.append("selected_clip_ids must contain integers")
        return errors
    if len(selected_ids) != len(set(selected_ids)):
        errors.append("selected_clip_ids contains duplicates")
    missing_ids = sorted(set(selected_ids) - set(clips_by_id))
    if missing_ids:
        errors.append(
            "selected clip IDs are absent from the source: "
            + ", ".join(str(clip_id) for clip_id in missing_ids)
        )

    expected_count = min(budget, len(clips))
    if len(selected_ids) != expected_count:
        errors.append(
            f"selected {len(selected_ids)} clips, expected {expected_count} "
            "for the declared budget and source size"
        )
    selected_count = manifest.get("num_selected_clips")
    if type(selected_count) is not int or selected_count != len(selected_ids):
        errors.append("num_selected_clips does not match selected_clip_ids")
    if errors:
        return errors

    expected_ids, expected_quotas = select_clip_ids(
        clips,
        expected_count,
        seed,
        strategy,
        action_labels,
    )
    if selected_ids != expected_ids:
        if strategy == "random":
            errors.append(
                "random selection is not reproducible for the declared seed"
            )
        else:
            errors.append("selected_clip_ids do not match the declared strategy")

    expected_manifest = build_subset_manifest(
        source_path,
        source,
        strategy,
        budget,
        seed,
        selected_ids,
        expected_quotas,
        action_labels,
    )
    exact_fields = (
        "motion_bin_quotas",
        "selection_metadata",
        "selected_clip_references",
        "motion_bin_counts",
        "aggregate_action_counts",
        "overlap_note",
    )
    for field in exact_fields:
        if not _values_match(manifest.get(field), expected_manifest[field]):
            errors.append(f"{field} does not match the source selection")

    approximate_fields = (
        "motion_binning",
        "aggregate_action_fractions",
        "summary_motion_statistics",
        "summary_action_balance_statistics",
    )
    for field in approximate_fields:
        if not _values_match(manifest.get(field), expected_manifest[field]):
            errors.append(f"{field} does not match the source selection")
    return errors


def _validate_global_dataset_subset(
    manifest: dict,
    manifest_path: Path,
) -> list[str]:
    errors: list[str] = []
    source_value = manifest.get("source_train_manifest")
    if not isinstance(source_value, str) or not source_value:
        return ["source_train_manifest must be a non-empty path string"]
    source_path = Path(source_value)
    if not source_path.is_file():
        return [f"source train manifest is missing: {source_path}"]
    if source_value != str(source_path.resolve()):
        errors.append("source_train_manifest must be an absolute resolved path")

    try:
        source = _load_json_object(source_path, "source train manifest")
    except ValueError as error:
        return [str(error)]
    errors.extend(_global_train_source_errors(source_path, source))
    if errors:
        return errors

    if manifest.get("manifest_version") != GLOBAL_MANIFEST_VERSION:
        errors.append(f"manifest_version must be {GLOBAL_MANIFEST_VERSION}")
    if manifest.get("source_kind") != SOURCE_KIND_GLOBAL_TRAIN:
        errors.append(f"source_kind must be {SOURCE_KIND_GLOBAL_TRAIN!r}")
    if manifest.get("source_train_manifest_sha256") != _file_sha256(source_path):
        errors.append("source_train_manifest_sha256 does not match")
    for field in (
        "source_dataset_index",
        "source_dataset_index_sha256",
        "scenario_name",
    ):
        if manifest.get(field) != source.get(field):
            errors.append(f"{field} does not match the train manifest")
    if manifest.get("source_split_name") != "train":
        errors.append("source_split_name must be 'train'")

    action_labels = _load_action_labels(source)
    clips = source["clips"]
    clips_by_id = {clip["global_clip_id"]: clip for clip in clips}
    if manifest.get("declared_action_space") != action_labels:
        errors.append("declared_action_space does not match the global dataset")
    if manifest.get("num_available_clips") != len(clips):
        errors.append("num_available_clips does not match the train manifest")

    strategy = manifest.get("strategy")
    if strategy not in STRATEGIES:
        errors.append(f"strategy must be one of: {', '.join(STRATEGIES)}")
        return errors
    budget = manifest.get("budget")
    if type(budget) is not int or budget < 1:
        errors.append("budget must be a positive integer")
        return errors
    seed = manifest.get("seed")
    if type(seed) is not int:
        errors.append("seed must be an integer")
        return errors

    selected_ids = manifest.get("selected_global_clip_ids")
    if not isinstance(selected_ids, list):
        errors.append("selected_global_clip_ids must be a list")
        return errors
    if any(not isinstance(clip_id, str) or not clip_id for clip_id in selected_ids):
        errors.append("selected_global_clip_ids must contain non-empty strings")
        return errors
    if len(selected_ids) != len(set(selected_ids)):
        errors.append("selected_global_clip_ids contains duplicates")
    missing_ids = sorted(set(selected_ids) - set(clips_by_id))
    if missing_ids:
        errors.append(
            "selected global clip IDs are absent from train.json: "
            + ", ".join(missing_ids)
        )
    expected_count = min(budget, len(clips))
    if len(selected_ids) != expected_count:
        errors.append(
            f"selected {len(selected_ids)} clips, expected {expected_count} "
            "for the declared budget and training split size"
        )
    if manifest.get("num_selected_clips") != len(selected_ids):
        errors.append("num_selected_clips does not match selected_global_clip_ids")
    errors.extend(_validate_holdout_exclusion(source_path, source, selected_ids))
    if errors:
        return errors

    expected_ids, expected_quotas = select_clip_ids(
        clips,
        expected_count,
        seed,
        strategy,
        action_labels,
    )
    if selected_ids != expected_ids:
        if strategy == "random":
            errors.append(
                "random selection is not reproducible for the declared seed"
            )
        else:
            errors.append(
                "selected_global_clip_ids do not match the declared strategy"
            )

    expected_manifest = build_subset_manifest(
        source_path,
        source,
        strategy,
        budget,
        seed,
        selected_ids,
        expected_quotas,
        action_labels,
    )
    for field, expected_value in expected_manifest.items():
        if not _values_match(manifest.get(field), expected_value):
            errors.append(f"{field} does not match the source selection")
    return errors


def _validate_holdout_exclusion(
    train_path: Path,
    train: dict,
    selected_ids: list[str],
) -> list[str]:
    """Read holdout ID lists only to prove selected IDs do not leak."""
    errors: list[str] = []
    selected_set = set(selected_ids)
    for split_name in ("val", "test"):
        path = train_path.parent / f"{split_name}.json"
        if not path.is_file():
            errors.append(f"missing {split_name} manifest required for leakage check")
            continue
        try:
            holdout = _load_json_object(path, f"{split_name} manifest")
        except ValueError as error:
            errors.append(str(error))
            continue
        if holdout.get("split_name") != split_name:
            errors.append(f"{split_name}.json has the wrong split_name")
        if holdout.get("source_dataset_index") != train.get("source_dataset_index"):
            errors.append(f"{split_name}.json references a different dataset index")
        holdout_ids = holdout.get("global_clip_ids")
        if not isinstance(holdout_ids, list) or any(
            not isinstance(clip_id, str) for clip_id in holdout_ids
        ):
            errors.append(f"{split_name}.json global_clip_ids is invalid")
            continue
        leaked = sorted(selected_set & set(holdout_ids))
        if leaked:
            errors.append(
                f"selected global clip IDs also occur in {split_name}.json: "
                + ", ".join(leaked)
            )
    return errors


def _values_match(actual, expected) -> bool:
    if isinstance(expected, bool):
        return type(actual) is bool and actual == expected
    if type(expected) is int:
        return type(actual) is int and actual == expected
    if isinstance(expected, dict):
        return (
            isinstance(actual, dict)
            and set(actual) == set(expected)
            and all(
                _values_match(actual[key], value)
                for key, value in expected.items()
            )
        )
    if isinstance(expected, list):
        return (
            isinstance(actual, list)
            and len(actual) == len(expected)
            and all(
                _values_match(actual_value, expected_value)
                for actual_value, expected_value in zip(actual, expected)
            )
        )
    if isinstance(expected, float):
        return (
            not isinstance(actual, bool)
            and isinstance(actual, (int, float))
            and math.isfinite(actual)
            and math.isclose(
                actual,
                expected,
                rel_tol=NUMERIC_TOLERANCE_RELATIVE,
                abs_tol=NUMERIC_TOLERANCE_ABSOLUTE,
            )
        )
    return actual == expected


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate fixed-budget global or legacy subset manifests."
    )
    parser.add_argument("manifests", nargs="+", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    invalid = False
    for manifest_path in args.manifests:
        errors = validate_dataset_subset(manifest_path)
        if errors:
            invalid = True
            print(f"INVALID {manifest_path}")
            for error in errors:
                print(f"  - {error}")
        else:
            print(f"VALID {manifest_path}")
    if invalid:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
