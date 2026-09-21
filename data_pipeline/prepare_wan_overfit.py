#!/usr/bin/env python3
"""Prepare real 17-observation, 16-action samples from the random-500 subset.

This module only reads existing episode and clip data. Its output contains
references to original RGB PNGs, not copies or padded frames.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from collections import Counter
from pathlib import Path

try:
    from .build_action_captions import (
        generate_action_prompt,
        run_length_encode_actions,
    )
except ImportError:
    from build_action_captions import generate_action_prompt, run_length_encode_actions


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SUBSET = REPOSITORY_ROOT / "data/subsets/my_way_home/train/budget_0500/random.json"
DEFAULT_OUTPUT = REPOSITORY_ROOT / "data/wan_training/my_way_home/overfit_0016"
MANIFEST_VERSION = 1
NUM_FRAMES = 17
NUM_ACTIONS = 16
SELECTION_SEED = 42
SELECTION_COUNT = 16


class IneligibleCandidate(ValueError):
    """A source clip cannot be extended without inventing data."""

    def __init__(self, reason: str):
        self.reason = reason
        super().__init__(reason)


def _load_object(path: Path) -> dict:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"could not read {path}: {error}") from error
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return value


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_source_context(subset_path: Path, dataset_dir: Path) -> dict:
    """Check the archived subset against local formal train/val/test manifests."""
    subset_path = subset_path.resolve()
    dataset_dir = dataset_dir.resolve()
    subset = _load_object(subset_path)
    if (
        subset.get("source_kind") != "global_train_split"
        or subset.get("source_split_name") != "train"
        or subset.get("strategy") != "random"
    ):
        raise ValueError("source must be the global train random subset")
    clips = subset.get("selected_clips")
    ids = subset.get("selected_global_clip_ids")
    if not isinstance(clips, list) or not isinstance(ids, list):
        raise ValueError("subset selected_clips and selected_global_clip_ids must be lists")
    if len(ids) != len(set(ids)) or [clip.get("global_clip_id") for clip in clips] != ids:
        raise ValueError("subset clip IDs must be unique and match selected_clips")
    if subset.get("num_selected_clips") != len(clips):
        raise ValueError("subset num_selected_clips does not match selected_clips")

    splits = {name: _load_object(dataset_dir / f"{name}.json") for name in ("train", "val", "test")}
    for name, split in splits.items():
        if split.get("split_name") != name or split.get("scenario_name") != subset.get("scenario_name"):
            raise ValueError(f"{name}.json has the wrong split or scenario")
        split_ids = split.get("global_clip_ids")
        if not isinstance(split_ids, list) or len(split_ids) != len(set(split_ids)):
            raise ValueError(f"{name}.json has invalid global_clip_ids")
    if subset.get("source_train_manifest_sha256") != _sha256(dataset_dir / "train.json"):
        raise ValueError("subset source_train_manifest_sha256 differs from local train.json")
    global_index = dataset_dir / "all_clips.json"
    if subset.get("source_dataset_index_sha256") != _sha256(global_index):
        raise ValueError("subset source_dataset_index_sha256 differs from local all_clips.json")

    train_clips = splits["train"].get("clips")
    if not isinstance(train_clips, list):
        raise ValueError("train.json clips must be a list")
    train_by_id = {clip["global_clip_id"]: clip for clip in train_clips}
    if len(train_by_id) != len(train_clips):
        raise ValueError("train.json contains duplicate clip IDs")
    for clip in clips:
        if not isinstance(clip, dict) or train_by_id.get(clip.get("global_clip_id")) != clip:
            raise ValueError("subset clip differs from local train.json")
    train_ids = set(splits["train"]["global_clip_ids"])
    val_ids = set(splits["val"]["global_clip_ids"])
    test_ids = set(splits["test"]["global_clip_ids"])
    if set(ids) - train_ids or set(ids) & (val_ids | test_ids):
        raise ValueError("subset contains clips outside train or in val/test")
    if train_ids & val_ids or train_ids & test_ids or val_ids & test_ids:
        raise ValueError("formal train/val/test splits overlap")
    return {
        "subset": subset,
        "subset_path": subset_path,
        "subset_sha256": _sha256(subset_path),
        "clips": clips,
        "ids": ids,
        "dataset_dir": dataset_dir,
    }


def _source_rgb_path(episode_dir: Path, observation: dict) -> Path:
    rgb = observation.get("rgb")
    if not isinstance(rgb, str) or not rgb:
        raise IneligibleCandidate("missing_rgb_reference")
    relative = Path(rgb)
    if relative.is_absolute() or ".." in relative.parts or relative.suffix.lower() != ".png":
        raise IneligibleCandidate("invalid_rgb_reference")
    path = (episode_dir / relative).resolve()
    if not path.is_relative_to(episode_dir.resolve()):
        raise IneligibleCandidate("invalid_rgb_reference")
    if not path.is_file():
        raise IneligibleCandidate("missing_rgb_frame")
    return path


def extend_candidate(clip: dict, episode_dir: Path, metadata: dict, local_clip: dict) -> dict:
    """Return one sample, or a specific ineligibility reason."""
    episode_dir = episode_dir.resolve()
    scenario = clip.get("scenario_name")
    episode_name = clip.get("episode_name")
    if (
        not isinstance(scenario, str)
        or not isinstance(episode_name, str)
        or episode_dir.name != episode_name
        or episode_dir.parent.name != scenario
        or Path(clip.get("source_episode", "")).parts[-2:] != (scenario, episode_name)
        or metadata.get("provenance", {}).get("scenario_name") != scenario
    ):
        raise IneligibleCandidate("source_episode_mismatch")
    local_id = clip.get("local_clip_id")
    if (
        type(local_id) is not int
        or local_clip.get("clip_id") != local_id
        or clip.get("global_clip_id") != f"{scenario}_{episode_name}_clip_{local_id:06d}"
    ):
        raise IneligibleCandidate("original_clip_mismatch")
    for field, local_field in (
        ("observation_indices", "observation_indices"),
        ("transition_steps", "transition_steps"),
        ("actions", "actions"),
        ("start_observation_index", "start_observation_index"),
    ):
        if clip.get(field) != local_clip.get(local_field):
            raise IneligibleCandidate("original_clip_mismatch")
    start = clip.get("start_observation_index")
    if type(start) is not int or start < 0:
        raise IneligibleCandidate("original_clip_mismatch")
    observation_indices = list(range(start, start + NUM_FRAMES))
    transition_steps = observation_indices[:-1]
    if (
        clip.get("observation_indices") != observation_indices[:-1]
        or clip.get("transition_steps") != transition_steps[:-1]
        or clip.get("num_frames") != 16
        or clip.get("num_actions") != 15
    ):
        raise IneligibleCandidate("original_clip_mismatch")

    observations = metadata.get("observations")
    transitions = metadata.get("transitions")
    if not isinstance(observations, list) or not isinstance(transitions, list):
        raise ValueError(f"malformed episode metadata: {episode_dir}")
    frame_paths = []
    for index in observation_indices:
        if index >= len(observations):
            raise IneligibleCandidate("missing_observation")
        observation = observations[index]
        if not isinstance(observation, dict) or observation.get("observation_index") != index:
            raise IneligibleCandidate("observation_alignment")
        frame_paths.append(str(_source_rgb_path(episode_dir, observation)))
    if len(set(frame_paths)) != NUM_FRAMES:
        raise IneligibleCandidate("duplicated_frame_reference")

    actions = []
    for step in transition_steps:
        if step >= len(transitions):
            raise IneligibleCandidate("missing_transition")
        transition = transitions[step]
        if not isinstance(transition, dict) or (
            transition.get("step") != step
            or transition.get("observation_index") != step
            or transition.get("next_observation_index") != step + 1
        ):
            raise IneligibleCandidate("transition_alignment")
        if type(transition.get("done")) is not bool or type(transition.get("truncated")) is not bool:
            raise IneligibleCandidate("transition_alignment")
        if transition["done"] or transition["truncated"]:
            raise IneligibleCandidate("terminal_transition")
        actions.append(transition.get("action"))
    if actions[:-1] != clip.get("actions"):
        raise IneligibleCandidate("original_action_mismatch")
    try:
        runs = run_length_encode_actions(actions)
    except ValueError as error:
        raise IneligibleCandidate("unknown_action") from error

    return {
        "wan_overfit_manifest_version": MANIFEST_VERSION,
        "dataset_split": "train",
        "selection_strategy": "random",
        "global_clip_id": clip["global_clip_id"],
        "episode_name": episode_name,
        "local_clip_id": local_id,
        "source_episode": str(episode_dir),
        "observation_indices": observation_indices,
        "transition_steps": transition_steps,
        "rgb_frames": frame_paths,
        "raw_actions": actions,
        "action_runs": runs,
        "prompt": generate_action_prompt(runs),
        "num_frames": NUM_FRAMES,
        "num_actions": NUM_ACTIONS,
        "fps": clip.get("fps"),
    }


def evaluate_candidates(context: dict, episodes_root: Path) -> tuple[list[dict], Counter]:
    """Inspect every selected candidate; cache source episodes and clip indices."""
    episodes_root = episodes_root.resolve()
    cache: dict[str, tuple[Path, dict, list[dict]]] = {}
    eligible = []
    reasons: Counter = Counter()
    for clip in context["clips"]:
        episode_name = clip.get("episode_name")
        if not isinstance(episode_name, str):
            reasons["source_episode_mismatch"] += 1
            continue
        if episode_name not in cache:
            episode_dir = episodes_root / episode_name
            metadata_path = episode_dir / "metadata.json"
            index_path = episodes_root.parent.parent / "clips" / episodes_root.name / episode_name / "clips.json"
            if not metadata_path.is_file() or not index_path.is_file():
                reasons["missing_source_metadata"] += 1
                continue
            metadata = _load_object(metadata_path)
            clip_index = _load_object(index_path)
            if (
                _sha256(metadata_path) != clip.get("source_episode_metadata_sha256")
                or _sha256(index_path) != clip.get("clips_json_sha256")
            ):
                raise ValueError(f"source checksum mismatch for {episode_name}")
            local_clips = clip_index.get("clips")
            if not isinstance(local_clips, list):
                raise ValueError(f"malformed clip index: {index_path}")
            cache[episode_name] = (episode_dir, metadata, local_clips)
        if episode_name not in cache:
            continue
        episode_dir, metadata, local_clips = cache[episode_name]
        local_id = clip.get("local_clip_id")
        if type(local_id) is not int or local_id < 0 or local_id >= len(local_clips):
            reasons["original_clip_mismatch"] += 1
            continue
        try:
            record = extend_candidate(clip, episode_dir, metadata, local_clips[local_id])
        except IneligibleCandidate as error:
            reasons[error.reason] += 1
            continue
        eligible.append(record)
    return eligible, reasons


def select_records(eligible: list[dict], seed: int = SELECTION_SEED, count: int = SELECTION_COUNT) -> list[dict]:
    """Select by SHA-256(seed:global_clip_id), breaking ties by ID."""
    if type(seed) is not int or count < 1:
        raise ValueError("seed must be an integer and count must be positive")
    if len(eligible) < count:
        raise ValueError(f"only {len(eligible)} eligible clips; need {count}")
    def key(record: dict) -> tuple[str, str]:
        clip_id = record["global_clip_id"]
        digest = hashlib.sha256(f"{seed}:{clip_id}".encode("utf-8")).hexdigest()
        return digest, clip_id
    return sorted(eligible, key=key)[:count]


def _write_atomic(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    try:
        temporary.write_text(content, encoding="utf-8")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def prepare_wan_overfit(
    subset_path: Path = DEFAULT_SUBSET,
    output_dir: Path = DEFAULT_OUTPUT,
    *,
    dataset_dir: Path | None = None,
    episodes_root: Path | None = None,
    seed: int = SELECTION_SEED,
    count: int = SELECTION_COUNT,
) -> dict:
    subset_path = subset_path.resolve()
    output_dir = output_dir.resolve()
    if dataset_dir is None:
        dataset_dir = REPOSITORY_ROOT / "data/datasets" / "my_way_home"
    if episodes_root is None:
        episodes_root = REPOSITORY_ROOT / "data/episodes/my_way_home"
    protected_roots = [
        REPOSITORY_ROOT / "data" / name
        for name in ("episodes", "clips", "datasets", "subsets", "training")
    ]
    protected_roots.append(episodes_root.resolve())
    if any(output_dir == root or output_dir.is_relative_to(root) for root in protected_roots):
        raise ValueError("output must be separate from source data and formal manifests")
    context = load_source_context(subset_path, dataset_dir)
    eligible, reasons = evaluate_candidates(context, episodes_root)
    selected = select_records(eligible, seed, count)
    for record in selected:
        record["source_subset_sha256"] = context["subset_sha256"]
    report = {
        "wan_overfit_manifest_version": MANIFEST_VERSION,
        "source_subset": str(subset_path),
        "source_subset_sha256": context["subset_sha256"],
        "selection_rule": "lowest SHA-256 of UTF-8 seed:global_clip_id, ties by global_clip_id",
        "selection_seed": seed,
        "selection_count": count,
        "candidates_examined": len(context["clips"]),
        "eligible": len(eligible),
        "ineligible": sum(reasons.values()),
        "ineligible_reasons": dict(sorted(reasons.items())),
        "selected_global_clip_ids": [record["global_clip_id"] for record in selected],
        "manifest": str(output_dir / "manifest.jsonl"),
        "media_type": "original_rgb_png_references",
        "new_mp4s_created": 0,
    }
    _write_atomic(
        output_dir / "manifest.jsonl",
        "".join(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n" for record in selected),
    )
    _write_atomic(output_dir / "report.json", json.dumps(report, indent=2) + "\n")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--subset", type=Path, default=DEFAULT_SUBSET)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    try:
        report = prepare_wan_overfit(args.subset, args.output)
    except ValueError as error:
        raise SystemExit(str(error)) from error
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
