#!/usr/bin/env python3
"""Select fixed-budget clip subsets for controlled dataset experiments."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
from collections import Counter
from fractions import Fraction
from pathlib import Path
from typing import TypeAlias

import numpy as np

try:
    from .validate_action_statistics import validate_action_statistics
    from .validate_motion_scores import validate_motion_scores
except ImportError:
    from validate_action_statistics import validate_action_statistics
    from validate_motion_scores import validate_motion_scores


STRATEGIES = (
    "random",
    "motion_stratified",
    "action_balanced",
    "joint_motion_action",
)
MOTION_BIN_LABELS = ("low", "medium", "high")
MANIFEST_VERSION = 1
GLOBAL_MANIFEST_VERSION = 2
SELECTION_VERSION = 1
SOURCE_KIND_LEGACY = "episode_clip_index"
SOURCE_KIND_GLOBAL_TRAIN = "global_train_split"
MOTION_BIN_METHOD = "rank_based_motion_mean_tertiles"
MOTION_BIN_SORT = "motion_mean_ascending_then_clip_id"
MOTION_QUOTA_RULE = (
    "equal_with_remainder_in_label_order_then_capacity_redistribution"
)
MOTION_QUOTA_REDISTRIBUTION_TIE_BREAK = (
    "smallest_current_quota_then_low_medium_high"
)
JOINT_MOTION_BIN_SCHEDULE = (
    "round_robin_low_medium_high_skipping_fulfilled_bins"
)
ACTION_IMBALANCE_METRIC = "l1_distance_to_uniform"
ACTION_IMBALANCE_FORMULA = (
    "sum_a(abs(p_selected[a] - 1 / num_declared_actions))"
)
ACTION_TARGET = "uniform_over_declared_action_space"
GREEDY_TIE_BREAK = "lowest_clip_id"
GLOBAL_GREEDY_TIE_BREAK = "lowest_global_clip_id"
OVERLAP_NOTE = (
    "Aggregate action counts are sample-level statistics across overlapping clips; "
    "they are not the unique source-episode transition distribution."
)
STRATEGY_OBJECTIVES = {
    "random": "uniform_sample_without_replacement",
    "motion_stratified": "equal_motion_tertile_coverage_with_seeded_sampling",
    "action_balanced": "greedy_minimize_global_action_l1_to_uniform",
    "joint_motion_action": (
        "motion_tertile_quotas_then_greedy_global_action_l1_to_uniform"
    ),
}
SEED_USAGE = {
    "random": "uniform_subset_sampling",
    "motion_stratified": "uniform_sampling_within_motion_bins",
    "action_balanced": "recorded_only_selection_is_deterministic",
    "joint_motion_action": "recorded_only_selection_is_deterministic",
}

ClipId: TypeAlias = int | str


def select_dataset_subset(
    clip_index_path: Path,
    budget: int,
    seed: int,
    strategy: str,
    output_path: Path,
) -> Path:
    clip_index_path = _normalize_index_path(clip_index_path)
    _validate_arguments(budget, seed, strategy, clip_index_path, output_path)
    clip_index = _load_json_object(clip_index_path, "selection source")
    _validate_source_statistics(clip_index_path, clip_index)
    _validate_output_not_source_metadata(
        output_path,
        clip_index_path,
        clip_index,
    )
    clips = clip_index["clips"]
    if not clips:
        raise ValueError("source clip index contains no clips to select")
    actual_budget = min(budget, len(clips))
    action_labels = _load_action_labels(clip_index)
    selected_clip_ids, motion_quotas = select_clip_ids(
        clips,
        actual_budget,
        seed,
        strategy,
        action_labels,
    )
    manifest = build_subset_manifest(
        clip_index_path,
        clip_index,
        strategy,
        budget,
        seed,
        selected_clip_ids,
        motion_quotas,
        action_labels,
    )
    _write_json_atomic(output_path, manifest)
    return output_path


def select_clip_ids(
    clips: list[dict],
    budget: int,
    seed: int,
    strategy: str,
    action_labels: list[str],
) -> tuple[list[ClipId], dict[str, int] | None]:
    if budget < 0 or budget > len(clips):
        raise ValueError("selection budget must be between 0 and clip count")

    motion_bins, _, _ = build_motion_bins(clips)
    if strategy == "random":
        population = sorted(
            (_clip_identifier(clip) for clip in clips),
            key=_identifier_sort_key,
        )
        selected = random.Random(seed).sample(population, budget)
        return sorted(selected, key=_identifier_sort_key), None

    if strategy == "motion_stratified":
        quotas = allocate_motion_quotas(motion_bins, budget)
        generator = random.Random(seed)
        selected = []
        for label in MOTION_BIN_LABELS:
            population = sorted(motion_bins[label], key=_identifier_sort_key)
            selected.extend(generator.sample(population, quotas[label]))
        return sorted(selected, key=_identifier_sort_key), quotas

    if strategy == "action_balanced":
        selected = select_action_balanced(clips, budget, action_labels)
        return sorted(selected, key=_identifier_sort_key), None

    if strategy == "joint_motion_action":
        quotas = allocate_motion_quotas(motion_bins, budget)
        selected = select_joint_motion_action(
            clips,
            motion_bins,
            quotas,
            action_labels,
        )
        return sorted(selected, key=_identifier_sort_key), quotas

    raise ValueError(f"unsupported strategy: {strategy}")


def build_motion_bins(
    clips: list[dict],
) -> tuple[dict[str, list[ClipId]], dict[ClipId, str], dict]:
    ordered = sorted(
        clips,
        key=lambda clip: (
            _clip_motion_mean(clip),
            _identifier_sort_key(_clip_identifier(clip)),
        ),
    )
    base_size, remainder = divmod(len(ordered), len(MOTION_BIN_LABELS))
    bin_sizes = [
        base_size + (1 if index < remainder else 0)
        for index in range(len(MOTION_BIN_LABELS))
    ]

    bins: dict[str, list[ClipId]] = {}
    assignments: dict[ClipId, str] = {}
    ranges = {}
    cursor = 0
    clips_by_id = {_clip_identifier(clip): clip for clip in clips}
    for label, size in zip(MOTION_BIN_LABELS, bin_sizes):
        members = ordered[cursor : cursor + size]
        member_ids = [_clip_identifier(clip) for clip in members]
        bins[label] = member_ids
        assignments.update({clip_id: label for clip_id in member_ids})
        ranges[label] = (
            {
                "min": min(
                    _clip_motion_mean(clips_by_id[clip_id])
                    for clip_id in member_ids
                ),
                "max": max(
                    _clip_motion_mean(clips_by_id[clip_id])
                    for clip_id in member_ids
                ),
            }
            if member_ids
            else None
        )
        cursor += size

    definition = {
        "method": MOTION_BIN_METHOD,
        "labels": list(MOTION_BIN_LABELS),
        "sort_order": (
            "motion_mean_ascending_then_global_clip_id"
            if _is_global_train_source_from_clips(clips)
            else MOTION_BIN_SORT
        ),
        "source_bin_counts": {
            label: len(bins[label]) for label in MOTION_BIN_LABELS
        },
        "source_bin_motion_mean_ranges": ranges,
        "quota_rule": MOTION_QUOTA_RULE,
        "quota_redistribution_tie_break": (
            MOTION_QUOTA_REDISTRIBUTION_TIE_BREAK
        ),
    }
    return bins, assignments, definition


def allocate_motion_quotas(
    motion_bins: dict[str, list[ClipId]],
    budget: int,
) -> dict[str, int]:
    base_quota, remainder = divmod(budget, len(MOTION_BIN_LABELS))
    requested = {
        label: base_quota + (1 if index < remainder else 0)
        for index, label in enumerate(MOTION_BIN_LABELS)
    }
    quotas = {
        label: min(requested[label], len(motion_bins[label]))
        for label in MOTION_BIN_LABELS
    }

    unallocated = budget - sum(quotas.values())
    while unallocated:
        eligible = [
            label
            for label in MOTION_BIN_LABELS
            if quotas[label] < len(motion_bins[label])
        ]
        if not eligible:
            raise ValueError("motion bins do not have enough capacity for budget")
        label = min(
            eligible,
            key=lambda item: (quotas[item], MOTION_BIN_LABELS.index(item)),
        )
        quotas[label] += 1
        unallocated -= 1
    return quotas


def select_action_balanced(
    clips: list[dict],
    budget: int,
    action_labels: list[str],
) -> list[ClipId]:
    remaining = {_clip_identifier(clip): clip for clip in clips}
    selected = []
    aggregate_counts: Counter[str] = Counter()
    for _ in range(budget):
        best = _choose_action_balancing_candidate(
            list(remaining.values()),
            aggregate_counts,
            action_labels,
        )
        best_id = _clip_identifier(best)
        selected.append(best_id)
        aggregate_counts.update(best["action_statistics"]["action_counts"])
        del remaining[best_id]
    return selected


def select_joint_motion_action(
    clips: list[dict],
    motion_bins: dict[str, list[ClipId]],
    quotas: dict[str, int],
    action_labels: list[str],
) -> list[ClipId]:
    clips_by_id = {_clip_identifier(clip): clip for clip in clips}
    remaining_by_bin = {
        label: {
            clip_id: clips_by_id[clip_id]
            for clip_id in motion_bins[label]
        }
        for label in MOTION_BIN_LABELS
    }
    remaining_quotas = dict(quotas)
    aggregate_counts: Counter[str] = Counter()
    selected = []

    while sum(remaining_quotas.values()):
        for label in MOTION_BIN_LABELS:
            if remaining_quotas[label] == 0:
                continue
            best = _choose_action_balancing_candidate(
                list(remaining_by_bin[label].values()),
                aggregate_counts,
                action_labels,
            )
            best_id = _clip_identifier(best)
            selected.append(best_id)
            aggregate_counts.update(best["action_statistics"]["action_counts"])
            del remaining_by_bin[label][best_id]
            remaining_quotas[label] -= 1
    return selected


def _choose_action_balancing_candidate(
    candidates: list[dict],
    aggregate_counts: Counter[str],
    action_labels: list[str],
) -> dict:
    if not candidates:
        raise ValueError("no candidate clip is available for greedy selection")

    def candidate_key(clip: dict) -> tuple[Fraction, str]:
        candidate_counts = aggregate_counts.copy()
        candidate_counts.update(clip["action_statistics"]["action_counts"])
        return (
            action_imbalance_fraction(candidate_counts, action_labels),
            _identifier_sort_key(_clip_identifier(clip)),
        )

    return min(candidates, key=candidate_key)


def action_imbalance_fraction(
    action_counts: dict[str, int] | Counter[str],
    action_labels: list[str],
) -> Fraction:
    total = sum(action_counts.get(action, 0) for action in action_labels)
    if total == 0:
        return Fraction(0, 1)
    action_count = len(action_labels)
    numerator = sum(
        abs(action_count * action_counts.get(action, 0) - total)
        for action in action_labels
    )
    return Fraction(numerator, action_count * total)


def build_subset_manifest(
    clip_index_path: Path,
    clip_index: dict,
    strategy: str,
    budget: int,
    seed: int,
    selected_clip_ids: list[ClipId],
    motion_quotas: dict[str, int] | None,
    action_labels: list[str],
) -> dict:
    clips = clip_index["clips"]
    clips_by_id = {_clip_identifier(clip): clip for clip in clips}
    positions = {
        _clip_identifier(clip): index for index, clip in enumerate(clips)
    }
    selected_clips = [clips_by_id[clip_id] for clip_id in selected_clip_ids]
    _, assignments, motion_definition = build_motion_bins(clips)
    summaries = summarize_selected_clips(
        selected_clips,
        assignments,
        action_labels,
    )

    common = {
        "scenario_name": clip_index["scenario_name"],
        "strategy": strategy,
        "budget": budget,
        "seed": seed,
        "num_available_clips": len(clips),
        "num_selected_clips": len(selected_clips),
        "declared_action_space": action_labels,
        "motion_binning": motion_definition,
        "motion_bin_quotas": motion_quotas,
        "selection_metadata": {
            "selection_version": SELECTION_VERSION,
            "objective": STRATEGY_OBJECTIVES[strategy],
            "seed_usage": SEED_USAGE[strategy],
            "greedy_tie_break": (
                (
                    GLOBAL_GREEDY_TIE_BREAK
                    if _is_global_train_manifest(clip_index)
                    else GREEDY_TIE_BREAK
                )
                if strategy in {"action_balanced", "joint_motion_action"}
                else None
            ),
            "joint_motion_bin_schedule": (
                JOINT_MOTION_BIN_SCHEDULE
                if strategy == "joint_motion_action"
                else None
            ),
            "action_target": ACTION_TARGET,
            "action_imbalance_metric": ACTION_IMBALANCE_METRIC,
            "action_imbalance_formula": ACTION_IMBALANCE_FORMULA,
        },
        **summaries,
        "overlap_note": OVERLAP_NOTE,
    }

    if _is_global_train_manifest(clip_index):
        return {
            "manifest_version": GLOBAL_MANIFEST_VERSION,
            "source_kind": SOURCE_KIND_GLOBAL_TRAIN,
            "source_train_manifest": str(clip_index_path.resolve()),
            "source_train_manifest_sha256": _file_sha256(clip_index_path),
            "source_dataset_index": clip_index["source_dataset_index"],
            "source_dataset_index_sha256": clip_index[
                "source_dataset_index_sha256"
            ],
            "source_split_name": "train",
            **common,
            "selected_global_clip_ids": selected_clip_ids,
            "selected_clip_references": [
                {
                    "global_clip_id": clip_id,
                    "source_train_position": positions[clip_id],
                    "source_record": f"/clips/{positions[clip_id]}",
                }
                for clip_id in selected_clip_ids
            ],
            "selected_clips": selected_clips,
        }

    return {
        "manifest_version": MANIFEST_VERSION,
        "source_kind": SOURCE_KIND_LEGACY,
        "source_clip_index": str(clip_index_path.resolve()),
        "source_clip_index_sha256": _file_sha256(clip_index_path),
        "source_episode": clip_index["source_episode"],
        **common,
        "selected_clip_ids": selected_clip_ids,
        "selected_clip_references": [
            {
                "clip_id": clip_id,
                "source_clip_position": positions[clip_id],
                "source_record": f"/clips/{positions[clip_id]}",
            }
            for clip_id in selected_clip_ids
        ],
    }


def summarize_selected_clips(
    selected_clips: list[dict],
    motion_assignments: dict[ClipId, str],
    action_labels: list[str],
) -> dict:
    motion_values = np.asarray(
        [_clip_motion_mean(clip) for clip in selected_clips],
        dtype=np.float64,
    )
    aggregate_counts: Counter[str] = Counter()
    for clip in selected_clips:
        aggregate_counts.update(clip["actions"])
    ordered_counts = {
        action: aggregate_counts.get(action, 0)
        for action in action_labels
    }
    total_actions = sum(ordered_counts.values())
    aggregate_fractions = {
        action: count / total_actions if total_actions else 0.0
        for action, count in ordered_counts.items()
    }
    motion_bin_counts = Counter(
        motion_assignments[_clip_identifier(clip)] for clip in selected_clips
    )
    imbalance = float(action_imbalance_fraction(ordered_counts, action_labels))
    episode_counts = Counter(
        clip.get("source_episode") for clip in selected_clips
    )
    episode_counts.pop(None, None)
    ordered_episode_counts = {
        episode: episode_counts[episode] for episode in sorted(episode_counts)
    }
    top_five_count = sum(sorted(episode_counts.values(), reverse=True)[:5])

    return {
        "motion_bin_counts": {
            label: motion_bin_counts.get(label, 0)
            for label in MOTION_BIN_LABELS
        },
        "aggregate_action_counts": ordered_counts,
        "aggregate_action_fractions": aggregate_fractions,
        "summary_motion_statistics": {
            "motion_mean_mean": float(np.mean(motion_values)),
            "motion_mean_std": float(np.std(motion_values)),
            "motion_mean_min": float(np.min(motion_values)),
            "motion_mean_max": float(np.max(motion_values)),
        },
        "summary_action_balance_statistics": {
            "target": ACTION_TARGET,
            "metric": ACTION_IMBALANCE_METRIC,
            "formula": ACTION_IMBALANCE_FORMULA,
            "uniform_target_fraction": 1.0 / len(action_labels),
            "action_imbalance_score": imbalance,
            "mean_clip_entropy": float(
                np.mean(
                    [
                        clip["action_statistics"]["action_entropy"]
                        for clip in selected_clips
                    ]
                )
            ),
            "mean_dominant_action_fraction": float(
                np.mean(
                    [
                        clip["action_statistics"]["dominant_action_fraction"]
                        for clip in selected_clips
                    ]
                )
            ),
            "mean_noop_fraction": float(
                np.mean(
                    [
                        clip["action_statistics"]["noop_fraction"]
                        for clip in selected_clips
                    ]
                )
            ),
        },
        "source_episode_concentration": {
            "clips_per_source_episode": ordered_episode_counts,
            "unique_source_episode_count": len(episode_counts),
            "max_clips_from_one_episode": max(episode_counts.values(), default=0),
            "top_5_episode_clip_fraction": (
                top_five_count / len(selected_clips) if selected_clips else 0.0
            ),
        },
    }


def _validate_arguments(
    budget: int,
    seed: int,
    strategy: str,
    clip_index_path: Path,
    output_path: Path,
) -> None:
    if type(budget) is not int or budget < 1:
        raise ValueError("budget must be a positive integer")
    if type(seed) is not int:
        raise ValueError("seed must be an integer")
    if strategy not in STRATEGIES:
        raise ValueError(f"strategy must be one of: {', '.join(STRATEGIES)}")
    if output_path.suffix.lower() != ".json":
        raise ValueError("output manifest path must end in .json")
    if clip_index_path.resolve() == output_path.resolve():
        raise ValueError("output manifest must not overwrite the source clips.json")


def _validate_output_not_source_metadata(
    output_path: Path,
    clip_index_path: Path,
    clip_index: dict,
) -> None:
    protected_paths = {
        clip_index_path.resolve(): "source clips.json",
    }
    if _is_global_train_manifest(clip_index):
        protected_paths[Path(clip_index["source_dataset_index"]).resolve()] = (
            "source all_clips.json"
        )
        for split_name in ("train", "val", "test"):
            split_path = (
                clip_index_path.parent / f"{split_name}.json"
            ).resolve()
            protected_paths[split_path] = f"source {split_name}.json"
    else:
        protected_paths[
            (Path(clip_index["source_episode"]) / "metadata.json").resolve()
        ] = "source episode metadata.json"
    description = protected_paths.get(output_path.resolve())
    if description is not None:
        raise ValueError(f"output manifest must not overwrite {description}")


def _validate_source_statistics(clip_index_path: Path, clip_index: dict) -> None:
    if _is_global_train_manifest(clip_index):
        errors = _global_train_source_errors(clip_index_path, clip_index)
        if errors:
            formatted = "\n".join(f"  - {error}" for error in errors)
            raise ValueError(f"invalid global training manifest:\n{formatted}")
        return

    motion_errors = validate_motion_scores(clip_index_path)
    action_errors = validate_action_statistics(clip_index_path)
    if motion_errors or action_errors:
        details = [*(f"motion: {error}" for error in motion_errors)]
        details.extend(f"action: {error}" for error in action_errors)
        formatted = "\n".join(f"  - {error}" for error in details)
        raise ValueError(
            f"source clip index lacks valid motion/action statistics:\n{formatted}"
        )


def _load_action_labels(clip_index: dict) -> list[str]:
    if _is_global_train_manifest(clip_index):
        dataset_index = _load_json_object(
            Path(clip_index["source_dataset_index"]),
            "global dataset index",
        )
        labels = dataset_index.get("declared_action_space")
        if not _valid_action_labels(labels):
            raise ValueError(
                "global dataset index declared_action_space must contain "
                "unique action strings"
            )
        return labels

    source_episode = Path(clip_index["source_episode"])
    metadata = json.loads(
        (source_episode / "metadata.json").read_text(encoding="utf-8")
    )
    return [entry["label"] for entry in metadata["action_space"]]


def _global_train_source_errors(source_path: Path, source: dict) -> list[str]:
    """Validate a train split without consulting validation or test manifests."""
    errors: list[str] = []
    if source.get("split_name") != "train":
        errors.append("global subset selection requires split_name='train'")
    if source.get("scenario_name") != source_path.parent.name:
        errors.append("scenario_name does not match the train manifest directory")

    dataset_value = source.get("source_dataset_index")
    dataset_path = Path(dataset_value) if isinstance(dataset_value, str) else None
    if dataset_path is None or not dataset_path.is_absolute():
        errors.append("source_dataset_index must be an absolute path")
        dataset_path = None
    elif not dataset_path.is_file():
        errors.append(f"source_dataset_index is missing: {dataset_path}")
        dataset_path = None
    elif dataset_value != str(dataset_path.resolve()):
        errors.append("source_dataset_index must be a resolved path")

    dataset_index = None
    if dataset_path is not None:
        if source.get("source_dataset_index_sha256") != _file_sha256(dataset_path):
            errors.append("source_dataset_index_sha256 does not match")
        try:
            dataset_index = _load_json_object(dataset_path, "global dataset index")
        except ValueError as error:
            errors.append(str(error))

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
    clip_ids = []
    for position, clip in enumerate(clips):
        if not isinstance(clip, dict):
            errors.append(f"clips[{position}] must be an object")
            continue
        clip_id = clip.get("global_clip_id")
        if not isinstance(clip_id, str) or not clip_id:
            errors.append(f"clips[{position}].global_clip_id is invalid")
            continue
        clip_ids.append(clip_id)
        _validate_global_clip_for_selection(clip, position, errors)
    if clip_ids != ids:
        errors.append("global_clip_ids must exactly match clips in order")
    if len(clip_ids) != len(set(clip_ids)):
        errors.append("clips contain duplicate global_clip_id values")
    if source.get("num_clips") != len(clips):
        errors.append("num_clips does not match clips length")

    if isinstance(dataset_index, dict):
        if dataset_index.get("scenario_name") != source.get("scenario_name"):
            errors.append("scenario_name differs from the global dataset index")
        labels = dataset_index.get("declared_action_space")
        if not _valid_action_labels(labels):
            errors.append("global dataset declared_action_space is invalid")
        global_clips = dataset_index.get("clips")
        if not isinstance(global_clips, list):
            errors.append("global dataset clips must be a list")
        else:
            global_by_id = {
                clip.get("global_clip_id"): clip
                for clip in global_clips
                if isinstance(clip, dict)
                and isinstance(clip.get("global_clip_id"), str)
            }
            if len(global_by_id) != len(global_clips):
                errors.append("global dataset clip IDs are invalid or duplicated")
            for position, clip in enumerate(clips):
                if not isinstance(clip, dict):
                    continue
                clip_id = clip.get("global_clip_id")
                if global_by_id.get(clip_id) != clip:
                    errors.append(
                        f"clips[{position}] does not exactly match the global dataset"
                    )
    return errors


def _validate_global_clip_for_selection(
    clip: dict,
    position: int,
    errors: list[str],
) -> None:
    label = f"clips[{position}]"
    required_strings = ("source_episode", "rgb_video")
    for field in required_strings:
        if not isinstance(clip.get(field), str) or not clip[field]:
            errors.append(f"{label}.{field} must be a non-empty string")
    if type(clip.get("local_clip_id")) is not int:
        errors.append(f"{label}.local_clip_id must be an integer")
    depth_frames = clip.get("depth_frames")
    if depth_frames is not None and (
        not isinstance(depth_frames, list)
        or any(not isinstance(path, str) or not path for path in depth_frames)
    ):
        errors.append(f"{label}.depth_frames must be null or a list of paths")
    actions = clip.get("actions")
    if not isinstance(actions, list) or any(
        not isinstance(action, str) or not action for action in actions
    ):
        errors.append(f"{label}.actions must be a list of action strings")
        actions = []
    if clip.get("num_actions") is not None and clip.get("num_actions") != len(actions):
        errors.append(f"{label}.num_actions does not match actions length")
    motion = clip.get("motion_statistics")
    motion_mean = motion.get("mean") if isinstance(motion, dict) else None
    if (
        isinstance(motion_mean, bool)
        or not isinstance(motion_mean, (int, float))
        or not np.isfinite(motion_mean)
    ):
        errors.append(f"{label}.motion_statistics.mean must be numeric")
    action_statistics = clip.get("action_statistics")
    if not isinstance(action_statistics, dict):
        errors.append(f"{label}.action_statistics must be an object")
        return
    action_counts = action_statistics.get("action_counts")
    if not isinstance(action_counts, dict) or any(
        not isinstance(action, str)
        or type(count) is not int
        or count < 0
        for action, count in (
            action_counts.items() if isinstance(action_counts, dict) else []
        )
    ):
        errors.append(f"{label}.action_statistics.action_counts is invalid")
    elif Counter(action_counts) != Counter(actions):
        errors.append(f"{label}.action_statistics.action_counts differs from actions")
    for field in (
        "dominant_action_fraction",
        "noop_fraction",
        "action_entropy",
    ):
        value = action_statistics.get(field)
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not np.isfinite(value)
        ):
            errors.append(f"{label}.action_statistics.{field} must be finite")


def _valid_action_labels(value: object) -> bool:
    return (
        isinstance(value, list)
        and bool(value)
        and all(isinstance(label, str) and label for label in value)
        and len(value) == len(set(value))
    )


def _clip_identifier(clip: dict) -> ClipId:
    if "global_clip_id" in clip:
        clip_id = clip["global_clip_id"]
        if not isinstance(clip_id, str) or not clip_id:
            raise ValueError("global_clip_id must be a non-empty string")
        return clip_id
    clip_id = clip.get("clip_id")
    if type(clip_id) is not int:
        raise ValueError("clip_id must be an integer")
    return clip_id


def _clip_motion_mean(clip: dict) -> float:
    if "global_clip_id" in clip:
        motion = clip.get("motion_statistics")
        value = motion.get("mean") if isinstance(motion, dict) else None
    else:
        value = clip.get("motion_mean")
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError("clip motion mean must be numeric")
    return float(value)


def _identifier_sort_key(clip_id: ClipId) -> str:
    if type(clip_id) is int:
        return f"integer:{clip_id:020d}"
    return f"string:{clip_id}"


def _is_global_train_manifest(value: dict) -> bool:
    return "split_manifest_version" in value or "global_clip_ids" in value


def _is_global_train_source_from_clips(clips: list[dict]) -> bool:
    return bool(clips) and "global_clip_id" in clips[0]


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


def _write_json_atomic(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_name(f".{path.name}.tmp")
    try:
        temporary_path.write_text(
            json.dumps(value, indent=2, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary_path, path)
    finally:
        if temporary_path.exists():
            temporary_path.unlink()


def _normalize_index_path(path: Path) -> Path:
    return path / "clips.json" if path.is_dir() else path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Select a fixed-budget subset from train.json or a legacy clips.json."
        )
    )
    parser.add_argument(
        "clip_index",
        type=Path,
        help="source global train.json or legacy episode clips.json",
    )
    parser.add_argument(
        "--budget",
        type=int,
        required=True,
        help="requested number of unique whole clips",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="selection seed (default: 42)",
    )
    parser.add_argument(
        "--strategy",
        choices=STRATEGIES,
        required=True,
        help="selection strategy",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="output .json manifest path",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    try:
        output_path = select_dataset_subset(
            args.clip_index,
            args.budget,
            args.seed,
            args.strategy,
            args.output,
        )
    except ValueError as error:
        raise SystemExit(str(error)) from error
    print(f"WROTE {args.strategy} subset manifest to {output_path}")


if __name__ == "__main__":
    main()
