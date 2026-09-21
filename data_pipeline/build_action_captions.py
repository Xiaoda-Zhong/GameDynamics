#!/usr/bin/env python3
"""Build deterministic action-to-text JSONL manifests for video diffusion."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

try:
    from .select_dataset_subsets import SOURCE_KIND_GLOBAL_TRAIN
    from .validate_dataset_subset import validate_dataset_subset
except ImportError:
    from select_dataset_subsets import SOURCE_KIND_GLOBAL_TRAIN
    from validate_dataset_subset import validate_dataset_subset


CAPTION_MANIFEST_VERSION = 1
CAPTION_METHOD = "deterministic_action_run_length_text_v1"
ACTION_PHRASES = {
    "MOVE_FORWARD": "moves forward",
    "TURN_LEFT": "turns left",
    "TURN_RIGHT": "turns right",
    "NOOP": "stays still",
    "MOVE_FORWARD_LEFT": "moves forward while turning left",
    "MOVE_FORWARD_RIGHT": "moves forward while turning right",
}


def run_length_encode_actions(actions: list[str]) -> list[dict]:
    """Run-length encode consecutive action labels without changing order."""
    if not isinstance(actions, list) or not actions:
        raise ValueError("actions must be a non-empty list")
    runs: list[dict] = []
    for action in actions:
        if action not in ACTION_PHRASES:
            raise ValueError(f"no fixed phrase is declared for action {action!r}")
        if runs and runs[-1]["action"] == action:
            runs[-1]["length"] += 1
        else:
            runs.append({"action": action, "length": 1})
    return runs


def generate_action_prompt(action_runs: list[dict]) -> str:
    """Render one deterministic sentence from validated action runs."""
    if not isinstance(action_runs, list) or not action_runs:
        raise ValueError("action_runs must be a non-empty list")
    clauses = []
    for run in action_runs:
        if not isinstance(run, dict):
            raise ValueError("each action run must be an object")
        action = run.get("action")
        length = run.get("length")
        if action not in ACTION_PHRASES:
            raise ValueError(f"no fixed phrase is declared for action {action!r}")
        if type(length) is not int or length < 1:
            raise ValueError("action run length must be a positive integer")
        unit = "step" if length == 1 else "steps"
        clauses.append(f"{ACTION_PHRASES[action]} for {length} {unit}")

    if len(clauses) == 1:
        body = clauses[0]
    elif len(clauses) == 2:
        body = f"{clauses[0]}, then {clauses[1]}"
    else:
        body = f"{', '.join(clauses[:-1])}, then {clauses[-1]}"
    return f"The player {body}."


def build_action_caption_manifest(
    source_manifest_path: Path,
    output_path: Path,
) -> Path:
    """Build one JSONL manifest from a training subset or val/test split."""
    source_manifest_path = source_manifest_path.expanduser().resolve()
    output_path = output_path.expanduser().resolve()
    if output_path.suffix.lower() != ".jsonl":
        raise ValueError("output path must end in .jsonl")
    if output_path == source_manifest_path:
        raise ValueError("output must not overwrite its source manifest")

    context = load_caption_source(source_manifest_path)
    records = [
        build_caption_record(
            clip,
            source_manifest_path=source_manifest_path,
            source_manifest_sha256=context["source_manifest_sha256"],
            dataset_split=context["dataset_split"],
            selection_strategy=context["selection_strategy"],
        )
        for clip in context["clips"]
    ]
    _write_jsonl_atomic(output_path, records)
    return output_path


def build_caption_record(
    clip: dict,
    *,
    source_manifest_path: Path,
    source_manifest_sha256: str,
    dataset_split: str,
    selection_strategy: str | None,
) -> dict:
    raw_actions = list(clip["actions"])
    action_runs = run_length_encode_actions(raw_actions)
    video_path = Path(clip["rgb_video"])
    if not video_path.is_file():
        raise ValueError(f"referenced RGB video is missing: {video_path}")
    return {
        "caption_manifest_version": CAPTION_MANIFEST_VERSION,
        "caption_method": CAPTION_METHOD,
        "dataset_split": dataset_split,
        "selection_strategy": selection_strategy,
        "source_manifest": str(source_manifest_path),
        "source_manifest_sha256": source_manifest_sha256,
        "global_clip_id": clip["global_clip_id"],
        "source_episode": clip["source_episode"],
        "episode_name": clip.get("episode_name"),
        "local_clip_id": clip["local_clip_id"],
        "video_path": str(video_path),
        "depth_frames": clip.get("depth_frames"),
        "num_depth_frames": clip.get("num_depth_frames"),
        "prompt": generate_action_prompt(action_runs),
        "raw_actions": raw_actions,
        "action_runs": action_runs,
        "num_frames": clip["num_frames"],
        "num_actions": clip["num_actions"],
        "observation_indices": clip.get("observation_indices"),
        "motion_statistics": clip.get("motion_statistics"),
        "action_statistics": clip.get("action_statistics"),
    }


def load_caption_source(source_manifest_path: Path) -> dict:
    """Load and validate a formal training subset or complete eval split."""
    source = _load_json_object(source_manifest_path, "caption source manifest")
    if source.get("source_kind") == SOURCE_KIND_GLOBAL_TRAIN:
        errors = validate_dataset_subset(source_manifest_path)
        if errors:
            details = "\n".join(f"  - {error}" for error in errors)
            raise ValueError(f"invalid training subset manifest:\n{details}")
        clips = source.get("selected_clips")
        ids = source.get("selected_global_clip_ids")
        dataset_split = "train"
        selection_strategy = source.get("strategy")
        dataset_path = Path(source["source_dataset_index"])
    else:
        errors = _evaluation_split_errors(source_manifest_path, source)
        if errors:
            details = "\n".join(f"  - {error}" for error in errors)
            raise ValueError(f"invalid evaluation split manifest:\n{details}")
        clips = source["clips"]
        ids = source["global_clip_ids"]
        dataset_split = source["split_name"]
        selection_strategy = None
        dataset_path = Path(source["source_dataset_index"])

    dataset = _load_json_object(dataset_path, "global dataset index")
    action_labels = dataset.get("declared_action_space")
    if action_labels != list(ACTION_PHRASES):
        raise ValueError(
            "declared action space must exactly match the fixed caption mapping"
        )
    if [clip.get("global_clip_id") for clip in clips] != ids:
        raise ValueError("source clip records do not match their global ID list")
    return {
        "source": source,
        "clips": clips,
        "global_clip_ids": ids,
        "dataset_split": dataset_split,
        "selection_strategy": selection_strategy,
        "dataset_path": dataset_path.resolve(),
        "dataset_directory": dataset_path.resolve().parent,
        "source_manifest_sha256": _file_sha256(source_manifest_path),
        "declared_action_space": action_labels,
    }


def _evaluation_split_errors(source_path: Path, source: dict) -> list[str]:
    errors: list[str] = []
    split_name = source.get("split_name")
    if split_name not in {"val", "test"}:
        errors.append("evaluation source must be val.json or test.json")
    elif source_path.name != f"{split_name}.json":
        errors.append("evaluation source filename does not match split_name")

    clips = source.get("clips")
    ids = source.get("global_clip_ids")
    if not isinstance(clips, list):
        errors.append("clips must be a list")
        clips = []
    if not isinstance(ids, list) or any(not isinstance(item, str) for item in ids):
        errors.append("global_clip_ids must be a list of strings")
        ids = []
    if len(ids) != len(set(ids)):
        errors.append("global_clip_ids contains duplicates")
    if source.get("num_clips") != len(clips):
        errors.append("num_clips does not match clips length")
    if [clip.get("global_clip_id") for clip in clips if isinstance(clip, dict)] != ids:
        errors.append("global_clip_ids do not exactly match clips in order")

    dataset_value = source.get("source_dataset_index")
    dataset_path = Path(dataset_value) if isinstance(dataset_value, str) else None
    if dataset_path is None or not dataset_path.is_absolute():
        errors.append("source_dataset_index must be an absolute path")
        return errors
    if not dataset_path.is_file():
        errors.append(f"source_dataset_index is missing: {dataset_path}")
        return errors
    if dataset_value != str(dataset_path.resolve()):
        errors.append("source_dataset_index must be a resolved path")
    if source.get("source_dataset_index_sha256") != _file_sha256(dataset_path):
        errors.append("source_dataset_index_sha256 does not match")
    try:
        dataset = _load_json_object(dataset_path, "global dataset index")
    except ValueError as error:
        errors.append(str(error))
        return errors
    if source.get("scenario_name") != dataset.get("scenario_name"):
        errors.append("scenario_name differs from the global dataset index")
    dataset_clips = dataset.get("clips")
    if not isinstance(dataset_clips, list):
        errors.append("global dataset clips must be a list")
        return errors
    clips_by_id = {
        clip.get("global_clip_id"): clip
        for clip in dataset_clips
        if isinstance(clip, dict)
    }
    for position, clip in enumerate(clips):
        if not isinstance(clip, dict):
            errors.append(f"clips[{position}] must be an object")
            continue
        if clips_by_id.get(clip.get("global_clip_id")) != clip:
            errors.append(f"clips[{position}] differs from the global dataset")
    return errors


def _load_json_object(path: Path, label: str) -> dict:
    if not path.is_file():
        raise ValueError(f"missing {label}: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"could not read {label} {path}: {error}") from error
    if not isinstance(value, dict):
        raise ValueError(f"{label} must contain a JSON object: {path}")
    return value


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_jsonl_atomic(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_name(f".{path.name}.tmp")
    try:
        with temporary_path.open("w", encoding="utf-8") as output_file:
            for record in records:
                output_file.write(
                    json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n"
                )
        os.replace(temporary_path, path)
    finally:
        if temporary_path.exists():
            temporary_path.unlink()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build deterministic action-caption JSONL from a formal training "
            "subset or complete validation/test split."
        )
    )
    parser.add_argument("source_manifest", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    try:
        output_path = build_action_caption_manifest(
            args.source_manifest,
            args.output,
        )
    except ValueError as error:
        raise SystemExit(str(error)) from error
    count = sum(1 for _ in output_path.open("r", encoding="utf-8"))
    print(f"WROTE {count} action-caption records to {output_path}")


if __name__ == "__main__":
    main()
