#!/usr/bin/env python3
"""Build fixed-length clip metadata from one validated ViZDoom episode."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

try:
    from .validate_episode import validate_episode
except ImportError:
    from validate_episode import validate_episode


DEFAULT_CLIP_LENGTH = 16
DEFAULT_STRIDE = 8


def build_clip_index(
    episode_dir: Path,
    output_root: Path = Path("data/clips"),
    clip_length: int = DEFAULT_CLIP_LENGTH,
    stride: int = DEFAULT_STRIDE,
) -> Path:
    if clip_length < 2:
        raise ValueError("clip_length must be at least 2 observations")
    if stride < 1:
        raise ValueError("stride must be at least 1 observation")

    errors = validate_episode(episode_dir)
    if errors:
        details = "\n".join(f"  - {error}" for error in errors)
        raise ValueError(f"invalid source episode {episode_dir}:\n{details}")

    metadata = json.loads(
        (episode_dir / "metadata.json").read_text(encoding="utf-8")
    )
    observations = metadata["observations"]
    transitions_by_step = {
        transition["step"]: transition for transition in metadata["transitions"]
    }

    clips = []
    last_start = len(observations) - clip_length
    for start in range(0, last_start + 1, stride):
        observation_indices = list(range(start, start + clip_length))
        transition_steps = list(range(start, start + clip_length - 1))

        if not _candidate_is_valid(
            observations,
            transitions_by_step,
            observation_indices,
            transition_steps,
        ):
            continue

        source_transitions = [
            transitions_by_step[step] for step in transition_steps
        ]
        clips.append(
            {
                "clip_id": len(clips),
                "start_observation_index": observation_indices[0],
                "end_observation_index": observation_indices[-1],
                "observation_indices": observation_indices,
                "transition_steps": transition_steps,
                "actions": [item["action"] for item in source_transitions],
                "action_vectors": [
                    item["action_vector"] for item in source_transitions
                ],
                "chunk_ids": [item["chunk_id"] for item in source_transitions],
                "num_frames": len(observation_indices),
                "num_actions": len(transition_steps),
            }
        )

    scenario_name = metadata["provenance"]["scenario_name"]
    clip_index = {
        "scenario_name": scenario_name,
        "source_episode": str(episode_dir.resolve()),
        "clip_length": clip_length,
        "stride": stride,
        "source_num_observations": len(observations),
        "source_num_transitions": len(metadata["transitions"]),
        "num_clips": len(clips),
        "clips": clips,
    }

    destination = output_root / scenario_name / episode_dir.name
    destination.mkdir(parents=True, exist_ok=True)
    output_path = destination / "clips.json"
    if output_path.exists():
        output_path.unlink()
    output_path.write_text(
        json.dumps(clip_index, indent=2) + "\n",
        encoding="utf-8",
    )
    return output_path


def _candidate_is_valid(
    observations: list[dict],
    transitions_by_step: dict[int, dict],
    observation_indices: list[int],
    transition_steps: list[int],
) -> bool:
    for index in observation_indices:
        if index >= len(observations):
            return False
        if observations[index].get("observation_index") != index:
            return False

    for step in transition_steps:
        transition = transitions_by_step.get(step)
        if transition is None:
            return False
        if transition.get("observation_index") != step:
            return False
        if transition.get("next_observation_index") != step + 1:
            return False
    return True


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("episode_dir", type=Path)
    parser.add_argument("--clip-length", type=int, default=DEFAULT_CLIP_LENGTH)
    parser.add_argument("--stride", type=int, default=DEFAULT_STRIDE)
    parser.add_argument("--output", type=Path, default=Path("data/clips"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    try:
        output_path = build_clip_index(
            episode_dir=args.episode_dir,
            output_root=args.output,
            clip_length=args.clip_length,
            stride=args.stride,
        )
    except ValueError as error:
        raise SystemExit(str(error)) from error

    clip_index = json.loads(output_path.read_text(encoding="utf-8"))
    print(f"WROTE {output_path} ({clip_index['num_clips']} clips)")


if __name__ == "__main__":
    main()
