#!/usr/bin/env python3
"""Compare four validated global or legacy subset manifests."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

try:
    from .select_dataset_subsets import ACTION_IMBALANCE_FORMULA, STRATEGIES
    from .validate_dataset_subset import validate_dataset_subset
except ImportError:
    from select_dataset_subsets import ACTION_IMBALANCE_FORMULA, STRATEGIES
    from validate_dataset_subset import validate_dataset_subset


LEGACY_COMPARISON_KEYS = (
    "source_clip_index",
    "source_clip_index_sha256",
    "source_episode",
    "scenario_name",
    "budget",
    "seed",
    "num_available_clips",
    "declared_action_space",
)
GLOBAL_COMPARISON_KEYS = (
    "source_kind",
    "source_train_manifest",
    "source_train_manifest_sha256",
    "source_dataset_index",
    "source_dataset_index_sha256",
    "source_split_name",
    "scenario_name",
    "budget",
    "seed",
    "num_available_clips",
    "declared_action_space",
)


def compare_dataset_subsets(manifest_paths: list[Path]) -> str:
    """Validate and render a comparison of one manifest per strategy."""
    manifests = _load_validated_manifests(manifest_paths)
    _validate_same_experiment(manifests)
    by_strategy = {manifest["strategy"]: manifest for manifest in manifests}
    ordered = [by_strategy[strategy] for strategy in STRATEGIES]

    lines = [
        _render_summary_table(ordered),
        "",
        "Aggregate action fractions",
        _render_action_fraction_table(ordered),
        "",
        "Selected clips per source episode",
        _render_episode_concentration_table(ordered),
        "",
        f"Action imbalance score: {ACTION_IMBALANCE_FORMULA}.",
        "Lower scores are closer to the uniform target over the declared action space.",
        "Source-episode concentration is diagnostic; no episode quota is enforced.",
        (
            "These manifests define experimental conditions only; this comparison "
            "does not establish that any strategy is superior."
        ),
    ]
    return "\n".join(lines)


def _load_validated_manifests(manifest_paths: list[Path]) -> list[dict]:
    if len(manifest_paths) != len(STRATEGIES):
        raise ValueError(
            f"expected exactly {len(STRATEGIES)} manifests, one per strategy"
        )

    manifests = []
    for path in manifest_paths:
        errors = validate_dataset_subset(path)
        if errors:
            details = "\n".join(f"  - {error}" for error in errors)
            raise ValueError(f"invalid subset manifest {path}:\n{details}")
        manifests.append(json.loads(path.read_text(encoding="utf-8")))
    return manifests


def _validate_same_experiment(manifests: list[dict]) -> None:
    strategies = [manifest["strategy"] for manifest in manifests]
    duplicates = sorted(
        strategy for strategy in set(strategies) if strategies.count(strategy) > 1
    )
    if duplicates:
        raise ValueError(
            "duplicate strategy manifests: " + ", ".join(duplicates)
        )

    missing = [strategy for strategy in STRATEGIES if strategy not in strategies]
    unexpected = sorted(set(strategies) - set(STRATEGIES))
    if missing or unexpected:
        details = []
        if missing:
            details.append("missing " + ", ".join(missing))
        if unexpected:
            details.append("unexpected " + ", ".join(unexpected))
        raise ValueError("strategy set is incomplete: " + "; ".join(details))

    reference = manifests[0]
    comparison_keys = (
        GLOBAL_COMPARISON_KEYS
        if reference.get("source_kind") == "global_train_split"
        else LEGACY_COMPARISON_KEYS
    )
    for manifest in manifests[1:]:
        mismatches = [
            key
            for key in comparison_keys
            if manifest.get(key) != reference.get(key)
        ]
        if mismatches:
            raise ValueError(
                "manifests do not describe the same experiment; mismatched fields "
                f"for strategy {manifest['strategy']}: {', '.join(mismatches)}"
            )


def _render_summary_table(manifests: list[dict]) -> str:
    headers = [
        "Strategy",
        "Clips",
        "Episodes",
        "Low",
        "Medium",
        "High",
        "Motion mean",
        "Motion std",
        "Action L1",
        "Mean entropy",
        "Mean dominant",
        "Mean NOOP",
        "Max/episode",
        "Top-5 fraction",
    ]
    rows = []
    for manifest in manifests:
        motion_bins = manifest["motion_bin_counts"]
        motion = manifest["summary_motion_statistics"]
        action = manifest["summary_action_balance_statistics"]
        concentration = _episode_concentration(manifest)
        rows.append(
            [
                manifest["strategy"],
                str(manifest["num_selected_clips"]),
                str(concentration["unique_source_episode_count"]),
                str(motion_bins["low"]),
                str(motion_bins["medium"]),
                str(motion_bins["high"]),
                _format_float(motion["motion_mean_mean"]),
                _format_float(motion["motion_mean_std"]),
                _format_float(action["action_imbalance_score"]),
                _format_float(action["mean_clip_entropy"]),
                _format_float(action["mean_dominant_action_fraction"]),
                _format_float(action["mean_noop_fraction"]),
                str(concentration["max_clips_from_one_episode"]),
                _format_float(concentration["top_5_episode_clip_fraction"]),
            ]
        )
    return _render_table(headers, rows)


def _render_action_fraction_table(manifests: list[dict]) -> str:
    action_labels = manifests[0]["declared_action_space"]
    headers = ["Strategy", *action_labels]
    rows = [
        [
            manifest["strategy"],
            *(
                _format_float(manifest["aggregate_action_fractions"][action])
                for action in action_labels
            ),
        ]
        for manifest in manifests
    ]
    return _render_table(headers, rows)


def _render_episode_concentration_table(manifests: list[dict]) -> str:
    counts_by_strategy = {
        manifest["strategy"]: _episode_concentration(manifest)[
            "clips_per_source_episode"
        ]
        for manifest in manifests
    }
    source_episodes = sorted(
        {
            source_episode
            for counts in counts_by_strategy.values()
            for source_episode in counts
        }
    )
    headers = ["Source episode", *STRATEGIES]
    rows = [
        [
            Path(source_episode).name,
            *(
                str(counts_by_strategy[strategy].get(source_episode, 0))
                for strategy in STRATEGIES
            ),
        ]
        for source_episode in source_episodes
    ]
    if not rows:
        return "(source episode concentration unavailable for legacy manifests)"
    return _render_table(headers, rows)


def _episode_concentration(manifest: dict) -> dict:
    concentration = manifest.get("source_episode_concentration")
    if isinstance(concentration, dict):
        return concentration
    source_episode = manifest.get("source_episode")
    count = manifest["num_selected_clips"]
    counts = {source_episode: count} if isinstance(source_episode, str) else {}
    return {
        "clips_per_source_episode": counts,
        "unique_source_episode_count": len(counts),
        "max_clips_from_one_episode": count if counts else 0,
        "top_5_episode_clip_fraction": 1.0 if counts and count else 0.0,
    }


def _format_float(value: float) -> str:
    return f"{value:.6f}"


def _render_table(headers: list[str], rows: list[list[str]]) -> str:
    widths = [
        max(len(headers[index]), *(len(row[index]) for row in rows))
        for index in range(len(headers))
    ]

    def format_row(row: list[str]) -> str:
        return " | ".join(
            value.ljust(widths[index]) for index, value in enumerate(row)
        )

    separator = "-+-".join("-" * width for width in widths)
    return "\n".join(
        [format_row(headers), separator, *(format_row(row) for row in rows)]
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare one validated subset manifest for each strategy."
    )
    parser.add_argument("manifests", nargs=4, type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    try:
        report = compare_dataset_subsets(args.manifests)
    except (OSError, ValueError, json.JSONDecodeError) as error:
        raise SystemExit(str(error)) from error
    print(report)


if __name__ == "__main__":
    main()
