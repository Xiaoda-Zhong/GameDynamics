#!/usr/bin/env python3
"""Validate deterministic action-caption JSONL manifests."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

try:
    from .build_action_captions import (
        ACTION_PHRASES,
        CAPTION_MANIFEST_VERSION,
        CAPTION_METHOD,
        generate_action_prompt,
        load_caption_source,
        run_length_encode_actions,
    )
except ImportError:
    from build_action_captions import (
        ACTION_PHRASES,
        CAPTION_MANIFEST_VERSION,
        CAPTION_METHOD,
        generate_action_prompt,
        load_caption_source,
        run_length_encode_actions,
    )


def validate_action_caption_manifest(manifest_path: Path) -> list[str]:
    """Validate one JSONL against its declared training subset or eval split."""
    manifest_path = manifest_path.expanduser().resolve()
    try:
        records = read_jsonl_records(manifest_path)
    except ValueError as error:
        return [str(error)]
    if not records:
        return ["action-caption manifest must contain at least one record"]

    source_values = [record.get("source_manifest") for record in records]
    if any(not isinstance(value, str) or not value for value in source_values):
        return ["source_manifest must be a non-empty path string"]
    unique_source_values = set(source_values)
    if len(unique_source_values) != 1:
        return ["all records must reference the same source_manifest"]
    source_value = next(iter(unique_source_values))
    source_path = Path(source_value)
    if not source_path.is_absolute() or source_value != str(source_path.resolve()):
        return ["source_manifest must be an absolute resolved path"]

    try:
        context = load_caption_source(source_path)
    except ValueError as error:
        return [str(error)]
    expected_clips = context["clips"]
    expected_ids = context["global_clip_ids"]
    errors: list[str] = []
    actual_ids = [record.get("global_clip_id") for record in records]
    valid_actual_ids = [clip_id for clip_id in actual_ids if isinstance(clip_id, str)]
    if len(valid_actual_ids) != len(actual_ids):
        errors.append("global_clip_id values must be strings")
    if actual_ids != expected_ids:
        errors.append(
            "global clip IDs do not exactly match the corresponding source manifest"
        )
    if len(valid_actual_ids) != len(set(valid_actual_ids)):
        errors.append("global_clip_id values contain duplicates")
    if len(records) != len(expected_clips):
        errors.append("record count does not match the corresponding source manifest")

    expected_by_id = {
        clip["global_clip_id"]: clip for clip in expected_clips
    }
    for position, record in enumerate(records):
        record_id = record.get("global_clip_id")
        _validate_record(
            record,
            position,
            expected_by_id.get(record_id) if isinstance(record_id, str) else None,
            context,
            errors,
        )
    errors.extend(_validate_split_leakage(context, set(valid_actual_ids)))
    return errors


def _validate_record(
    record: dict,
    position: int,
    source_clip: dict | None,
    context: dict,
    errors: list[str],
) -> None:
    label = f"record {position}"
    if source_clip is None:
        errors.append(f"{label} does not reference a source clip")
        return
    if record.get("caption_manifest_version") != CAPTION_MANIFEST_VERSION:
        errors.append(f"{label} has the wrong caption_manifest_version")
    if record.get("caption_method") != CAPTION_METHOD:
        errors.append(f"{label} has the wrong caption_method")
    if record.get("dataset_split") != context["dataset_split"]:
        errors.append(f"{label} has the wrong dataset_split")
    if record.get("selection_strategy") != context["selection_strategy"]:
        errors.append(f"{label} has the wrong selection_strategy")
    if record.get("source_manifest_sha256") != context["source_manifest_sha256"]:
        errors.append(f"{label} source_manifest_sha256 does not match")

    exact_source_fields = {
        "source_episode": "source_episode",
        "local_clip_id": "local_clip_id",
        "video_path": "rgb_video",
        "depth_frames": "depth_frames",
        "num_depth_frames": "num_depth_frames",
        "num_frames": "num_frames",
        "num_actions": "num_actions",
        "observation_indices": "observation_indices",
        "motion_statistics": "motion_statistics",
        "action_statistics": "action_statistics",
    }
    for record_field, source_field in exact_source_fields.items():
        if record.get(record_field) != source_clip.get(source_field):
            errors.append(f"{label} {record_field} differs from the source clip")
    if record.get("episode_name") != source_clip.get("episode_name"):
        errors.append(f"{label} episode_name differs from the source clip")

    raw_actions = record.get("raw_actions")
    if raw_actions != source_clip.get("actions"):
        errors.append(f"{label} raw_actions differ from the source clip")
        return
    if not isinstance(raw_actions, list):
        errors.append(f"{label} raw_actions must be a list")
        return
    if record.get("num_actions") != len(raw_actions):
        errors.append(f"{label} num_actions does not match raw_actions length")
    if record.get("num_frames") == 16 and len(raw_actions) != 15:
        errors.append(f"{label} must have 15 raw actions for 16 frames")

    action_runs = record.get("action_runs")
    if not isinstance(action_runs, list):
        errors.append(f"{label} action_runs must be a list")
        return
    expanded_actions = []
    run_length_total = 0
    runs_are_well_formed = True
    for run_index, run in enumerate(action_runs):
        if not isinstance(run, dict) or set(run) != {"action", "length"}:
            errors.append(f"{label} action run {run_index} is malformed")
            runs_are_well_formed = False
            continue
        action = run.get("action")
        length = run.get("length")
        if action not in ACTION_PHRASES:
            errors.append(f"{label} action run {run_index} has no fixed phrase")
            runs_are_well_formed = False
        if type(length) is not int or length < 1:
            errors.append(f"{label} action run {run_index} has invalid length")
            runs_are_well_formed = False
            continue
        run_length_total += length
        if action in ACTION_PHRASES:
            expanded_actions.extend([action] * length)
    if run_length_total != record.get("num_actions"):
        errors.append(f"{label} action run lengths do not sum to num_actions")
    if expanded_actions != raw_actions:
        errors.append(f"{label} action runs do not reconstruct raw_actions")

    try:
        expected_runs = run_length_encode_actions(raw_actions)
    except ValueError as error:
        errors.append(f"{label} {error}")
        return
    if action_runs != expected_runs:
        errors.append(f"{label} action_runs are not the canonical encoding")
    if runs_are_well_formed:
        try:
            expected_prompt = generate_action_prompt(action_runs)
        except ValueError as error:
            errors.append(f"{label} could not regenerate prompt: {error}")
        else:
            if record.get("prompt") != expected_prompt:
                errors.append(
                    f"{label} prompt does not match deterministic regeneration"
                )

    video_value = record.get("video_path")
    if not isinstance(video_value, str) or not Path(video_value).is_file():
        errors.append(f"{label} referenced RGB video is missing")


def _validate_split_leakage(context: dict, actual_ids: set[object]) -> list[str]:
    errors: list[str] = []
    split_ids: dict[str, set[str]] = {}
    dataset_directory = context["dataset_directory"]
    for split_name in ("train", "val", "test"):
        path = dataset_directory / f"{split_name}.json"
        try:
            split = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            errors.append(
                f"could not read {split_name}.json for leakage check: {error}"
            )
            continue
        ids = split.get("global_clip_ids") if isinstance(split, dict) else None
        if not isinstance(ids, list) or any(not isinstance(item, str) for item in ids):
            errors.append(f"{split_name}.json has invalid global_clip_ids")
            continue
        split_ids[split_name] = set(ids)
    if len(split_ids) != 3:
        return errors
    for first, second in (("train", "val"), ("train", "test"), ("val", "test")):
        if split_ids[first] & split_ids[second]:
            errors.append(f"{first}.json and {second}.json contain overlapping clips")
    declared_split = context["dataset_split"]
    if actual_ids - split_ids[declared_split]:
        errors.append(
            f"caption records contain clips outside the {declared_split} split"
        )
    if declared_split == "train":
        holdout_ids = split_ids["val"] | split_ids["test"]
        if actual_ids & holdout_ids:
            errors.append("training captions contain validation or test clips")
    return errors


def read_jsonl_records(path: Path) -> list[dict]:
    if not path.is_file():
        raise ValueError(f"missing action-caption manifest: {path}")
    records = []
    try:
        with path.open("r", encoding="utf-8") as input_file:
            for line_number, line in enumerate(input_file, start=1):
                if not line.strip():
                    raise ValueError(f"blank JSONL line at {line_number}")
                value = json.loads(line)
                if not isinstance(value, dict):
                    raise ValueError(f"JSONL line {line_number} must be an object")
                records.append(value)
    except OSError as error:
        raise ValueError(f"could not read action-caption manifest: {error}") from error
    except json.JSONDecodeError as error:
        raise ValueError(
            f"invalid JSON on JSONL line {line_number}: {error}"
        ) from error
    return records


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate deterministic action-caption JSONL manifests."
    )
    parser.add_argument("manifests", nargs="+", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    invalid = False
    for path in args.manifests:
        errors = validate_action_caption_manifest(path)
        if errors:
            invalid = True
            print(f"INVALID {path}")
            for error in errors:
                print(f"  - {error}")
        else:
            print(f"VALID {path}")
    if invalid:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
