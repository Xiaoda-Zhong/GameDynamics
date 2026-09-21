#!/usr/bin/env python3
"""Summarize validated deterministic action-caption JSONL manifests."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np

try:
    from .validate_action_captions import (
        read_jsonl_records,
        validate_action_caption_manifest,
    )
except ImportError:
    from validate_action_captions import (
        read_jsonl_records,
        validate_action_caption_manifest,
    )


def summarize_action_caption_manifest(path: Path) -> dict:
    path = path.expanduser().resolve()
    errors = validate_action_caption_manifest(path)
    if errors:
        details = "\n".join(f"  - {error}" for error in errors)
        raise ValueError(f"invalid action-caption manifest {path}:\n{details}")
    records = read_jsonl_records(path)
    run_counts = np.asarray(
        [len(record["action_runs"]) for record in records],
        dtype=np.int64,
    )
    total_runs = int(np.sum(run_counts))
    total_actions = sum(record["num_actions"] for record in records)
    distribution = Counter(int(value) for value in run_counts)
    unique_prompts = list(dict.fromkeys(record["prompt"] for record in records))
    return {
        "manifest": str(path),
        "dataset_split": records[0]["dataset_split"],
        "selection_strategy": records[0]["selection_strategy"],
        "number_of_clips": len(records),
        "action_runs_per_clip": {
            "mean": float(np.mean(run_counts)),
            "min": int(np.min(run_counts)),
            "max": int(np.max(run_counts)),
        },
        "action_run_count_distribution": {
            str(count): distribution[count] for count in sorted(distribution)
        },
        "mean_run_length": total_actions / total_runs,
        "number_of_unique_prompt_strings": len(unique_prompts),
        "example_prompts": unique_prompts[:5],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Summarize validated action-caption JSONL manifests."
    )
    parser.add_argument("manifests", nargs="+", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    try:
        summaries = [
            summarize_action_caption_manifest(path) for path in args.manifests
        ]
    except ValueError as error:
        raise SystemExit(str(error)) from error
    print(json.dumps(summaries, indent=2, ensure_ascii=False, allow_nan=False))


if __name__ == "__main__":
    main()
