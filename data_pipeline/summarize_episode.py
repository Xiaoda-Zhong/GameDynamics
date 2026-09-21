#!/usr/bin/env python3
"""Summarize action chunks in a validated Milestone 1 episode."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

try:
    from .validate_episode import validate_episode
except ImportError:
    from validate_episode import validate_episode


def summarize_episode(episode_dir: Path) -> dict:
    errors = validate_episode(episode_dir)
    if errors:
        details = "\n".join(f"  - {error}" for error in errors)
        raise ValueError(f"invalid episode {episode_dir}:\n{details}")

    metadata = json.loads(
        (episode_dir / "metadata.json").read_text(encoding="utf-8")
    )
    transitions = metadata["transitions"]
    action_labels = [entry["label"] for entry in metadata.get("action_space", [])]

    transition_counts = Counter(
        transition["action"] for transition in transitions
    )
    chunk_counts: Counter[str] = Counter()
    realized_chunk_lengths: list[int] = []
    previous_chunk_id = None

    for transition in transitions:
        chunk_id = transition["chunk_id"]
        if chunk_id != previous_chunk_id:
            chunk_counts[transition["action"]] += 1
            realized_chunk_lengths.append(1)
            previous_chunk_id = chunk_id
        else:
            realized_chunk_lengths[-1] += 1

    final_transition = transitions[-1] if transitions else {}
    return {
        "scenario_name": metadata["provenance"]["scenario_name"],
        "action_space_size": len(metadata.get("action_space", [])),
        "num_transitions": len(transitions),
        "num_action_chunks": len(realized_chunk_lengths),
        "action_frequency_by_transitions": _ordered_counts(
            transition_counts,
            action_labels,
        ),
        "action_frequency_by_chunks": _ordered_counts(
            chunk_counts,
            action_labels,
        ),
        "average_realized_chunk_length": (
            sum(realized_chunk_lengths) / len(realized_chunk_lengths)
            if realized_chunk_lengths
            else None
        ),
        "min_realized_chunk_length": (
            min(realized_chunk_lengths) if realized_chunk_lengths else None
        ),
        "max_realized_chunk_length": (
            max(realized_chunk_lengths) if realized_chunk_lengths else None
        ),
        "done": final_transition.get("done", False),
        "truncated": final_transition.get("truncated", False),
        "termination_reason": metadata["termination_reason"],
    }


def _ordered_counts(counts: Counter, action_labels: list[str]) -> dict[str, int]:
    result = {label: counts.get(label, 0) for label in action_labels}
    for label in sorted(set(counts) - set(action_labels)):
        result[label] = counts[label]
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("episode_dir", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    try:
        summary = summarize_episode(args.episode_dir)
    except ValueError as error:
        raise SystemExit(str(error)) from error
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
