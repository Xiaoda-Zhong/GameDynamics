#!/usr/bin/env python3
"""Validate a Milestone 2A clip index against its source episode."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

try:
    from .validate_episode import validate_episode
except ImportError:
    from validate_episode import validate_episode


def validate_clip_index(
    clip_index_path: Path,
    validate_source_depth_files: bool = True,
) -> list[str]:
    errors: list[str] = []
    clip_index_path = _normalize_index_path(clip_index_path)
    if not clip_index_path.is_file():
        return [f"missing clip index: {clip_index_path}"]

    try:
        clip_index = json.loads(clip_index_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        return [f"could not read clip index: {error}"]

    source_episode_value = clip_index.get("source_episode")
    if not isinstance(source_episode_value, str) or not source_episode_value:
        return ["source_episode must be a non-empty path string"]
    source_episode = Path(source_episode_value)

    episode_errors = validate_episode(
        source_episode,
        check_depth_files=validate_source_depth_files,
    )
    if episode_errors:
        errors.append(f"source episode is invalid: {source_episode}")
        errors.extend(f"source: {error}" for error in episode_errors)
        return errors

    try:
        source = json.loads(
            (source_episode / "metadata.json").read_text(encoding="utf-8")
        )
    except (OSError, json.JSONDecodeError) as error:
        return [f"could not read source metadata: {error}"]

    clip_length = clip_index.get("clip_length")
    stride = clip_index.get("stride")
    if type(clip_length) is not int or clip_length < 2:
        errors.append("clip_length must be an integer of at least 2")
    if type(stride) is not int or stride < 1:
        errors.append("stride must be a positive integer")

    observations = source["observations"]
    transitions = source["transitions"]
    transitions_by_step = {item["step"]: item for item in transitions}

    if clip_index.get("scenario_name") != source["provenance"]["scenario_name"]:
        errors.append("scenario_name does not match the source episode")
    if clip_index.get("source_num_observations") != len(observations):
        errors.append("source_num_observations does not match the source episode")
    if clip_index.get("source_num_transitions") != len(transitions):
        errors.append("source_num_transitions does not match the source episode")

    clips = clip_index.get("clips")
    if not isinstance(clips, list):
        errors.append("clips must be a list")
        return errors
    if clip_index.get("num_clips") != len(clips):
        errors.append("num_clips does not match clips length")

    if type(clip_length) is int and clip_length >= 2 and type(stride) is int and stride >= 1:
        expected_starts = _expected_valid_starts(
            observations,
            transitions_by_step,
            clip_length,
            stride,
        )
        actual_starts = [
            clip.get("start_observation_index")
            for clip in clips
            if isinstance(clip, dict)
        ]
        if actual_starts != expected_starts:
            errors.append(
                "clip windows do not match the valid windows at the declared stride"
            )

    for index, clip in enumerate(clips):
        if not isinstance(clip, dict):
            errors.append(f"clip {index} must be an object")
            continue
        if clip.get("clip_id") != index:
            errors.append(f"clip ID {index} is not continuous")
        if type(clip_length) is not int or clip_length < 2:
            continue
        _validate_clip(
            index,
            clip,
            clip_length,
            observations,
            transitions_by_step,
            errors,
        )
    return errors


def _validate_clip(
    clip_id: int,
    clip: dict,
    clip_length: int,
    observations: list[dict],
    transitions_by_step: dict[int, dict],
    errors: list[str],
) -> None:
    start = clip.get("start_observation_index")
    end = clip.get("end_observation_index")
    if type(start) is not int:
        errors.append(f"clip {clip_id} start index must be an integer")
        return

    expected_observations = list(range(start, start + clip_length))
    expected_steps = list(range(start, start + clip_length - 1))
    if end != expected_observations[-1]:
        errors.append(f"clip {clip_id} has an inconsistent end index")
    if clip.get("observation_indices") != expected_observations:
        errors.append(f"clip {clip_id} observation indices are not continuous")
    if clip.get("transition_steps") != expected_steps:
        errors.append(f"clip {clip_id} transition indices are not continuous")
    if clip.get("num_frames") != clip_length:
        errors.append(f"clip {clip_id} num_frames does not equal clip_length")
    if clip.get("num_actions") != clip_length - 1:
        errors.append(f"clip {clip_id} num_actions does not equal clip_length - 1")

    if any(index < 0 or index >= len(observations) for index in expected_observations):
        errors.append(f"clip {clip_id} references observations outside the episode")
        return

    source_transitions = []
    for step in expected_steps:
        transition = transitions_by_step.get(step)
        if transition is None:
            errors.append(f"clip {clip_id} references missing transition {step}")
            return
        if transition.get("observation_index") != step:
            errors.append(f"clip {clip_id} transition {step} has the wrong source")
        if transition.get("next_observation_index") is None:
            errors.append(f"clip {clip_id} crosses null-next transition {step}")
        elif transition.get("next_observation_index") != step + 1:
            errors.append(f"clip {clip_id} transition {step} has the wrong destination")
        if transition.get("done") is True:
            errors.append(f"clip {clip_id} crosses terminal transition {step}")
        source_transitions.append(transition)

    expected_actions = [item["action"] for item in source_transitions]
    expected_vectors = [item["action_vector"] for item in source_transitions]
    expected_chunk_ids = [item["chunk_id"] for item in source_transitions]
    if clip.get("actions") != expected_actions:
        errors.append(f"clip {clip_id} actions do not match source transitions")
    if clip.get("action_vectors") != expected_vectors:
        errors.append(f"clip {clip_id} action_vectors do not match source transitions")
    if clip.get("chunk_ids") != expected_chunk_ids:
        errors.append(f"clip {clip_id} chunk_ids do not match source transitions")


def _expected_valid_starts(
    observations: list[dict],
    transitions_by_step: dict[int, dict],
    clip_length: int,
    stride: int,
) -> list[int]:
    starts = []
    for start in range(0, len(observations) - clip_length + 1, stride):
        observation_indices = range(start, start + clip_length)
        transition_steps = range(start, start + clip_length - 1)
        observations_valid = all(
            observations[index].get("observation_index") == index
            for index in observation_indices
        )
        transitions_valid = all(
            step in transitions_by_step
            and transitions_by_step[step].get("observation_index") == step
            and transitions_by_step[step].get("next_observation_index") == step + 1
            for step in transition_steps
        )
        if observations_valid and transitions_valid:
            starts.append(start)
    return starts


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
        errors = validate_clip_index(normalized_path)
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
