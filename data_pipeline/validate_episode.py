#!/usr/bin/env python3
"""Validate a Milestone 1 ViZDoom episode directory."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


TERMINATION_REASONS = {
    "collector_truncation",
    "environment_timeout",
    "task_terminal",
}


def validate_episode(
    episode_dir: Path,
    check_depth_files: bool = True,
) -> list[str]:
    errors: list[str] = []
    metadata_path = episode_dir / "metadata.json"
    if not metadata_path.is_file():
        return [f"missing metadata file: {metadata_path}"]

    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        return [f"could not read metadata: {error}"]

    observations = metadata.get("observations")
    transitions = metadata.get("transitions")
    if not isinstance(observations, list):
        errors.append("observations must be a list")
        observations = []
    if not isinstance(transitions, list):
        errors.append("transitions must be a list")
        transitions = []

    terminal_without_next = bool(
        transitions
        and isinstance(transitions[-1], dict)
        and transitions[-1].get("done") is True
        and transitions[-1].get("next_observation_index") is None
    )
    expected_observations = len(transitions) + (0 if terminal_without_next else 1)
    if transitions and not observations:
        errors.append("transitions exist but the initial observation is missing")
    elif observations and len(observations) != expected_observations:
        errors.append(
            f"expected {expected_observations} observations for this termination "
            f"mode, got {len(observations)}"
        )

    initial_state_available = metadata.get("initial_state_available")
    if not isinstance(initial_state_available, bool):
        errors.append("initial_state_available must be a boolean")
    elif initial_state_available != bool(observations):
        errors.append(
            "initial_state_available is inconsistent with the observation list"
        )

    if metadata.get("num_observations") != len(observations):
        errors.append("num_observations does not match observations length")
    if metadata.get("num_transitions") != len(transitions):
        errors.append("num_transitions does not match transitions length")

    referenced_rgb: set[Path] = set()
    referenced_depth: set[Path] = set()
    for index, observation in enumerate(observations):
        if not isinstance(observation, dict):
            errors.append(f"observation {index} must be an object")
            continue
        if observation.get("observation_index") != index:
            errors.append(f"observation index {index} is not continuous")

        observation_type = observation.get("observation_type")
        if observation_type not in (None, "vizdoom_state"):
            errors.append(
                f"observation {index} is synthetic or absorbing: {observation_type}"
            )
        if observation.get("synthetic") is True:
            errors.append(f"observation {index} is marked synthetic")

        rgb = observation.get("rgb")
        if not isinstance(rgb, str):
            errors.append(f"observation {index} has no valid RGB path")
        else:
            _check_referenced_file(
                episode_dir, rgb, "RGB", index, errors, referenced_rgb
            )

        depth = observation.get("depth")
        if depth is not None and not isinstance(depth, str):
            errors.append(f"observation {index} depth must be a path or null")
        elif isinstance(depth, str):
            _check_referenced_file(
                episode_dir,
                depth,
                "depth",
                index,
                errors,
                referenced_depth,
                require_exists=check_depth_files,
            )

    for index, transition in enumerate(transitions):
        if not isinstance(transition, dict):
            errors.append(f"transition {index} must be an object")
            continue
        if transition.get("step") != index:
            errors.append(f"transition step {index} is not continuous")
        if transition.get("observation_index") != index:
            errors.append(f"transition {index} does not start at observation {index}")
        done = transition.get("done")
        truncated = transition.get("truncated")
        if not isinstance(done, bool) or not isinstance(truncated, bool):
            errors.append(f"transition {index} done/truncated must be booleans")
            continue

        next_observation_index = transition.get("next_observation_index")
        if next_observation_index is None:
            if not done:
                errors.append(
                    f"non-terminal transition {index} has no next observation"
                )
        else:
            if next_observation_index != index + 1:
                errors.append(
                    f"transition {index} does not end at observation {index + 1}"
                )
            if not isinstance(next_observation_index, int) or not (
                0 <= next_observation_index < len(observations)
            ):
                errors.append(
                    f"transition {index} references a missing next observation"
                )

        if done and truncated:
            errors.append(f"transition {index} cannot be both done and truncated")
        if index < len(transitions) - 1 and (done or truncated):
            errors.append(f"only the final transition may terminate the episode")

    if transitions:
        final = transitions[-1]
        if isinstance(final, dict):
            done = final.get("done")
            truncated = final.get("truncated")
            if done is False and truncated is False:
                errors.append("final transition must be done or truncated")

    _validate_termination_reason(metadata, transitions, errors)
    _validate_scenario_provenance(metadata, errors)
    _validate_action_space(metadata, transitions, errors)
    _validate_action_chunks(transitions, metadata, errors)

    _check_for_stale_files(episode_dir, "rgb", referenced_rgb, errors)
    if check_depth_files:
        _check_for_stale_files(episode_dir, "depth", referenced_depth, errors)
    return errors


def _validate_scenario_provenance(metadata: dict, errors: list[str]) -> None:
    provenance = metadata.get("provenance")
    if not isinstance(provenance, dict):
        errors.append("provenance must be an object")
        return

    for field in ("scenario_name", "config_path", "map"):
        if not isinstance(provenance.get(field), str) or not provenance[field]:
            errors.append(f"provenance {field} must be a non-empty string")
    wad_path = provenance.get("wad_path")
    if wad_path is not None and not isinstance(wad_path, str):
        errors.append("provenance wad_path must be a path or null")


def _validate_action_space(
    metadata: dict,
    transitions: list,
    errors: list[str],
) -> None:
    provenance = metadata.get("provenance", {})
    available_buttons = provenance.get("available_buttons")
    if (
        not isinstance(available_buttons, list)
        or any(not isinstance(button, str) for button in available_buttons)
        or len(set(available_buttons)) != len(available_buttons)
    ):
        errors.append("provenance available_buttons must be a list of unique names")
        return

    action_space = metadata.get("action_space")
    if not isinstance(action_space, list) or not action_space:
        errors.append("action_space must be a non-empty list")
        return

    declared_actions: dict[str, list[bool]] = {}
    for index, action in enumerate(action_space):
        if not isinstance(action, dict):
            errors.append(f"action_space entry {index} must be an object")
            continue

        label = action.get("label")
        buttons = action.get("buttons")
        vector = action.get("vector")
        if not isinstance(label, str) or not label:
            errors.append(f"action_space entry {index} has no valid label")
            continue
        if label in declared_actions:
            errors.append(f"action_space contains duplicate label {label}")
            continue
        if (
            not isinstance(buttons, list)
            or any(not isinstance(button, str) for button in buttons)
            or len(set(buttons)) != len(buttons)
            or any(button not in available_buttons for button in buttons)
        ):
            errors.append(f"action {label} has invalid button names")
            continue
        if (
            not isinstance(vector, list)
            or len(vector) != len(available_buttons)
            or any(type(value) is not bool for value in vector)
        ):
            errors.append(
                f"action {label} vector must contain one boolean per available button"
            )
            continue

        expected_vector = [button in buttons for button in available_buttons]
        if vector != expected_vector:
            errors.append(
                f"action {label} vector does not match available-button ordering"
            )
        declared_actions[label] = vector

    for index, transition in enumerate(transitions):
        if not isinstance(transition, dict):
            continue
        label = transition.get("action")
        if label not in declared_actions:
            errors.append(
                f"transition {index} uses undeclared action {label!r}"
            )
        elif transition.get("action_vector") != declared_actions[label]:
            errors.append(
                f"transition {index} action vector does not match declared action {label}"
            )


def _validate_termination_reason(
    metadata: dict,
    transitions: list,
    errors: list[str],
) -> None:
    reason = metadata.get("termination_reason")
    if reason not in TERMINATION_REASONS:
        errors.append(
            "termination_reason must be collector_truncation, "
            "environment_timeout, or task_terminal"
        )
        return

    if not transitions:
        if reason == "collector_truncation":
            errors.append("collector_truncation requires a final transition")
        return

    final = transitions[-1]
    if not isinstance(final, dict):
        return

    done = final.get("done")
    truncated = final.get("truncated")
    next_observation_index = final.get("next_observation_index")
    provenance = metadata.get("provenance", {})

    if reason == "collector_truncation":
        if done is not False or truncated is not True:
            errors.append(
                "collector_truncation requires final done=false and truncated=true"
            )
        if next_observation_index is None:
            errors.append("collector_truncation requires a real next observation")
        if len(transitions) != provenance.get("max_steps"):
            errors.append(
                "collector_truncation transition count must equal max_steps"
            )
        return

    if done is not True or truncated is not False:
        errors.append(
            f"{reason} requires final done=true and truncated=false"
        )
    if next_observation_index is not None:
        errors.append(f"{reason} must not reference an invented next observation")

    if reason == "environment_timeout":
        episode_timeout = provenance.get("episode_timeout")
        action_repeat = provenance.get("action_repeat")
        if type(episode_timeout) is not int or episode_timeout < 1:
            errors.append("environment_timeout requires a positive episode_timeout")
        elif type(action_repeat) is not int or action_repeat < 1:
            errors.append("environment_timeout requires a positive action_repeat")
        else:
            previous_tics = (len(transitions) - 1) * action_repeat
            current_tics = len(transitions) * action_repeat
            if not previous_tics < episode_timeout <= current_tics:
                errors.append(
                    "environment_timeout is inconsistent with episode_timeout "
                    "and action_repeat"
                )


def _validate_action_chunks(
    transitions: list,
    metadata: dict,
    errors: list[str],
) -> None:
    if not transitions:
        return

    required_fields = ("chunk_id", "chunk_step", "chunk_length")
    for index, transition in enumerate(transitions):
        if not isinstance(transition, dict):
            return
        for field in required_fields:
            if type(transition.get(field)) is not int:
                errors.append(f"transition {index} {field} must be an integer")
                return
        if transition["chunk_length"] < 1:
            errors.append(f"transition {index} chunk_length must be positive")
            return

    configured_range = metadata.get("provenance", {}).get(
        "action_chunk_length_range"
    )
    if (
        not isinstance(configured_range, list)
        or len(configured_range) != 2
        or any(type(value) is not int for value in configured_range)
    ):
        errors.append("provenance action_chunk_length_range must contain two integers")
        configured_range = None

    current_id = -1
    current_length = 0
    current_count = 0
    current_action = None
    current_vector = None

    for index, transition in enumerate(transitions):
        chunk_id = transition["chunk_id"]
        chunk_step = transition["chunk_step"]
        chunk_length = transition["chunk_length"]

        if index == 0 and chunk_id != 0:
            errors.append(f"first chunk_id must be 0, got {chunk_id}")
            return
        if chunk_id == current_id + 1:
            if current_id >= 0 and current_count != current_length:
                errors.append(
                    f"non-final chunk {current_id} realized {current_count} steps, "
                    f"expected {current_length}"
                )
            if chunk_step != 0:
                errors.append(f"chunk {chunk_id} must start with chunk_step 0")
            current_id = chunk_id
            current_length = chunk_length
            current_count = 0
            current_action = transition.get("action")
            current_vector = transition.get("action_vector")
        elif chunk_id != current_id:
            errors.append(
                f"transition {index} has non-continuous chunk_id {chunk_id}"
            )
            return

        if chunk_step != current_count:
            errors.append(
                f"transition {index} has chunk_step {chunk_step}, "
                f"expected {current_count}"
            )
        if chunk_length != current_length:
            errors.append(f"chunk {chunk_id} changes its declared chunk_length")
        if transition.get("action") != current_action:
            errors.append(f"chunk {chunk_id} changes action label")
        if transition.get("action_vector") != current_vector:
            errors.append(f"chunk {chunk_id} changes action vector")
        if configured_range is not None and not (
            configured_range[0] <= chunk_length <= configured_range[1]
        ):
            errors.append(
                f"transition {index} chunk_length is outside the configured range"
            )
        current_count += 1

    if current_count > current_length:
        errors.append(
            f"final chunk {current_id} realized {current_count} steps, "
            f"exceeding its declared length {current_length}"
        )
    elif current_count < current_length:
        final = transitions[-1]
        if final.get("done") is not True and final.get("truncated") is not True:
            errors.append(
                f"final chunk {current_id} is shorter than declared without "
                "done or truncated"
            )


def _check_referenced_file(
    episode_dir: Path,
    relative_path: str,
    kind: str,
    observation_index: int,
    errors: list[str],
    referenced: set[Path],
    require_exists: bool = True,
) -> None:
    path = Path(relative_path)
    if path.is_absolute() or ".." in path.parts:
        errors.append(
            f"observation {observation_index} has unsafe {kind} path: {relative_path}"
        )
        return
    referenced.add(path)
    if require_exists and not (episode_dir / path).is_file():
        errors.append(
            f"observation {observation_index} references missing {kind} file: "
            f"{relative_path}"
        )


def _check_for_stale_files(
    episode_dir: Path,
    subdirectory: str,
    referenced: set[Path],
    errors: list[str],
) -> None:
    directory = episode_dir / subdirectory
    actual = (
        {
            path.relative_to(episode_dir)
            for path in directory.rglob("*.png")
            if path.is_file()
        }
        if directory.is_dir()
        else set()
    )
    extra = sorted(actual - referenced)
    if extra:
        errors.append(
            f"unreferenced {subdirectory} files: "
            + ", ".join(str(path) for path in extra)
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("episode_dirs", nargs="+", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    invalid = False
    for episode_dir in args.episode_dirs:
        errors = validate_episode(episode_dir)
        if errors:
            invalid = True
            print(f"INVALID {episode_dir}")
            for error in errors:
                print(f"  - {error}")
        else:
            print(f"VALID {episode_dir}")
    if invalid:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
