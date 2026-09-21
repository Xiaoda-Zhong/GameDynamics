#!/usr/bin/env python3
"""Create and validate deterministic episode-level dataset splits."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import tempfile
from pathlib import Path
from typing import Any

try:
    from .build_dataset_index import validate_dataset_index
except ImportError:
    from build_dataset_index import validate_dataset_index


SPLIT_MANIFEST_VERSION = 1
SPLIT_NAMES = ("train", "val", "test")
DEFAULT_RATIOS = {"train": 0.8, "val": 0.1, "test": 0.1}
DEFAULT_SEED = 42


def split_dataset(
    dataset_index_path: Path,
    output_dir: Path | None = None,
    train_ratio: float = DEFAULT_RATIOS["train"],
    val_ratio: float = DEFAULT_RATIOS["val"],
    test_ratio: float = DEFAULT_RATIOS["test"],
    seed: int = DEFAULT_SEED,
) -> dict[str, Path]:
    """Split a global dataset index by episode and write three manifests."""
    dataset_index_path = dataset_index_path.expanduser().resolve()
    source_errors = validate_dataset_index(dataset_index_path)
    if source_errors:
        details = "\n".join(f"  - {error}" for error in source_errors)
        raise ValueError(f"invalid dataset index {dataset_index_path}:\n{details}")
    source = _read_json_object(dataset_index_path, "dataset index")

    ratios = _validate_ratios(train_ratio, val_ratio, test_ratio)
    if type(seed) is not int:
        raise ValueError("seed must be an integer")

    episode_names = [episode["episode_name"] for episode in source["episodes"]]
    assignments = _assign_episodes(episode_names, ratios, seed)
    episodes_by_name = {
        episode["episode_name"]: episode for episode in source["episodes"]
    }
    clips_by_episode: dict[str, list[dict[str, Any]]] = {
        episode_name: [] for episode_name in episode_names
    }
    for clip in source["clips"]:
        clips_by_episode[clip["episode_name"]].append(clip)

    destination = (output_dir or dataset_index_path.parent).expanduser().resolve()
    destination.mkdir(parents=True, exist_ok=True)
    source_sha256 = _file_sha256(dataset_index_path)
    paths: dict[str, Path] = {}
    for split_name in SPLIT_NAMES:
        assigned_names = sorted(assignments[split_name])
        selected_clips = [
            clip
            for episode_name in assigned_names
            for clip in clips_by_episode[episode_name]
        ]
        selected_clips.sort(key=lambda item: item["global_clip_id"])
        manifest = {
            "split_manifest_version": SPLIT_MANIFEST_VERSION,
            "split_name": split_name,
            "scenario_name": source["scenario_name"],
            "source_dataset_index": str(dataset_index_path),
            "source_dataset_index_sha256": source_sha256,
            "seed": seed,
            "ratios": ratios,
            "num_episodes": len(assigned_names),
            "num_clips": len(selected_clips),
            "episode_names": assigned_names,
            "episode_references": [
                episodes_by_name[episode_name] for episode_name in assigned_names
            ],
            "global_clip_ids": [
                clip["global_clip_id"] for clip in selected_clips
            ],
            "clips": selected_clips,
        }
        path = destination / f"{split_name}.json"
        _atomic_write_json(path, manifest)
        paths[split_name] = path
    return paths


def validate_dataset_splits(
    dataset_index_path: Path,
    output_dir: Path | None = None,
) -> list[str]:
    """Validate source provenance, episode isolation, and all split records."""
    errors: list[str] = []
    dataset_index_path = dataset_index_path.expanduser().resolve()
    try:
        source = _read_json_object(dataset_index_path, "dataset index")
    except ValueError as error:
        return [str(error)]

    source_errors = validate_dataset_index(dataset_index_path)
    errors.extend(f"source dataset index: {error}" for error in source_errors)
    if source_errors:
        return errors

    destination = (output_dir or dataset_index_path.parent).expanduser().resolve()
    manifests: dict[str, dict[str, Any]] = {}
    for split_name in SPLIT_NAMES:
        path = destination / f"{split_name}.json"
        try:
            manifests[split_name] = _read_json_object(
                path, f"{split_name} split manifest"
            )
        except ValueError as error:
            errors.append(str(error))
    if len(manifests) != len(SPLIT_NAMES):
        return errors

    source_sha256 = _file_sha256(dataset_index_path)
    source_episode_by_name = {
        episode["episode_name"]: episode for episode in source["episodes"]
    }
    source_clip_by_id = {
        clip["global_clip_id"]: clip for clip in source["clips"]
    }

    reference_seed: int | None = None
    reference_ratios: dict[str, float] | None = None
    observed_episode_owners: dict[str, str] = {}
    observed_clip_owners: dict[str, str] = {}

    for split_name in SPLIT_NAMES:
        manifest = manifests[split_name]
        prefix = f"{split_name}: "
        manifest_version = manifest.get("split_manifest_version")
        if (
            type(manifest_version) is not int
            or manifest_version != SPLIT_MANIFEST_VERSION
        ):
            errors.append(
                prefix
                + f"split_manifest_version must be {SPLIT_MANIFEST_VERSION}"
            )
        if manifest.get("split_name") != split_name:
            errors.append(prefix + "split_name does not match its filename")
        if manifest.get("scenario_name") != source["scenario_name"]:
            errors.append(prefix + "scenario_name does not match the dataset index")
        if manifest.get("source_dataset_index") != str(dataset_index_path):
            errors.append(prefix + "source_dataset_index must be the resolved source path")
        if manifest.get("source_dataset_index_sha256") != source_sha256:
            errors.append(prefix + "source_dataset_index_sha256 does not match")

        seed = manifest.get("seed")
        if type(seed) is not int:
            errors.append(prefix + "seed must be an integer")
        elif reference_seed is None:
            reference_seed = seed
        elif seed != reference_seed:
            errors.append(prefix + "seed differs from the other split manifests")

        ratios_value = manifest.get("ratios")
        try:
            ratios = _ratios_from_mapping(ratios_value)
        except ValueError as error:
            errors.append(prefix + str(error))
            ratios = None
        if ratios is not None:
            if reference_ratios is None:
                reference_ratios = ratios
            elif ratios != reference_ratios:
                errors.append(prefix + "ratios differ from the other split manifests")

        episode_names = manifest.get("episode_names")
        if not _is_string_list(episode_names):
            errors.append(prefix + "episode_names must be a list of strings")
            episode_names = []
        elif len(episode_names) != len(set(episode_names)):
            errors.append(prefix + "episode_names contains duplicates")
        elif episode_names != sorted(episode_names):
            errors.append(prefix + "episode_names must be sorted")

        episode_references = manifest.get("episode_references")
        if not isinstance(episode_references, list) or not all(
            isinstance(item, dict) for item in episode_references
        ):
            errors.append(prefix + "episode_references must be a list of objects")
            episode_references = []
        reference_names = [
            item.get("episode_name") for item in episode_references
        ]
        if any(not isinstance(name, str) for name in reference_names):
            errors.append(prefix + "episode_references contain malformed episode names")
        elif len(reference_names) != len(set(reference_names)):
            errors.append(prefix + "episode_references contain duplicate episodes")
        if reference_names != episode_names:
            errors.append(prefix + "episode_references do not match episode_names")
        for reference in episode_references:
            name = reference.get("episode_name")
            if isinstance(name, str) and source_episode_by_name.get(name) != reference:
                errors.append(
                    prefix + f"episode reference {name!r} differs from the source record"
                )

        clip_ids = manifest.get("global_clip_ids")
        if not _is_string_list(clip_ids):
            errors.append(prefix + "global_clip_ids must be a list of strings")
            clip_ids = []
        elif len(clip_ids) != len(set(clip_ids)):
            errors.append(prefix + "global_clip_ids contains duplicates")
        elif clip_ids != sorted(clip_ids):
            errors.append(prefix + "global_clip_ids must be sorted")

        clips = manifest.get("clips")
        if not isinstance(clips, list) or not all(
            isinstance(item, dict) for item in clips
        ):
            errors.append(prefix + "clips must be a list of objects")
            clips = []
        record_ids = [clip.get("global_clip_id") for clip in clips]
        if any(not isinstance(clip_id, str) for clip_id in record_ids):
            errors.append(prefix + "clip records contain malformed global_clip_id values")
        elif len(record_ids) != len(set(record_ids)):
            errors.append(prefix + "clip records contain duplicate global_clip_id values")
        if record_ids != clip_ids:
            errors.append(prefix + "clip records do not match global_clip_ids")
        for clip in clips:
            clip_id = clip.get("global_clip_id")
            if isinstance(clip_id, str) and source_clip_by_id.get(clip_id) != clip:
                errors.append(
                    prefix + f"clip record {clip_id!r} differs from the source record"
                )
            episode_name = clip.get("episode_name")
            if (
                not isinstance(episode_name, str)
                or episode_name not in set(episode_names)
            ):
                errors.append(
                    prefix
                    + f"clip {clip_id!r} belongs to unlisted episode {episode_name!r}"
                )

        declared_num_episodes = manifest.get("num_episodes")
        if type(declared_num_episodes) is not int or declared_num_episodes != len(
            episode_names
        ):
            errors.append(prefix + "num_episodes does not match episode_names")
        declared_num_clips = manifest.get("num_clips")
        if type(declared_num_clips) is not int or declared_num_clips != len(clip_ids):
            errors.append(prefix + "num_clips does not match global_clip_ids")

        for episode_name in episode_names:
            previous = observed_episode_owners.setdefault(episode_name, split_name)
            if previous != split_name:
                errors.append(
                    f"episode {episode_name!r} occurs in both {previous} and {split_name}"
                )
        for clip_id in clip_ids:
            previous = observed_clip_owners.setdefault(clip_id, split_name)
            if previous != split_name:
                errors.append(
                    f"global clip {clip_id!r} occurs in both {previous} and {split_name}"
                )

    source_episode_names = set(source_episode_by_name)
    observed_episode_names = set(observed_episode_owners)
    if observed_episode_names != source_episode_names:
        errors.append(
            "split episode union does not equal the global dataset: "
            + _describe_set_difference(source_episode_names, observed_episode_names)
        )
    source_clip_ids = set(source_clip_by_id)
    observed_clip_ids = set(observed_clip_owners)
    if observed_clip_ids != source_clip_ids:
        errors.append(
            "split clip union does not equal the global dataset: "
            + _describe_set_difference(source_clip_ids, observed_clip_ids)
        )

    if reference_seed is not None and reference_ratios is not None:
        expected_assignments = _assign_episodes(
            list(source_episode_by_name), reference_ratios, reference_seed
        )
        for split_name in SPLIT_NAMES:
            actual = manifests[split_name].get("episode_names")
            if _is_string_list(actual) and actual != sorted(
                expected_assignments[split_name]
            ):
                errors.append(
                    f"{split_name}: episode assignment is not reproducible from seed "
                    "and ratios"
                )

            expected_ids = sorted(
                clip_id
                for clip_id, clip in source_clip_by_id.items()
                if clip["episode_name"] in set(expected_assignments[split_name])
            )
            actual_ids = manifests[split_name].get("global_clip_ids")
            if _is_string_list(actual_ids) and actual_ids != expected_ids:
                errors.append(
                    f"{split_name}: global_clip_ids are not exactly the clips from its "
                    "assigned episodes"
                )
    return errors


def _assign_episodes(
    episode_names: list[str], ratios: dict[str, float], seed: int
) -> dict[str, list[str]]:
    counts = _apportion_counts(len(episode_names), ratios)
    shuffled_names = sorted(episode_names)
    random.Random(seed).shuffle(shuffled_names)
    train_end = counts["train"]
    val_end = train_end + counts["val"]
    return {
        "train": shuffled_names[:train_end],
        "val": shuffled_names[train_end:val_end],
        "test": shuffled_names[val_end:],
    }


def _apportion_counts(
    num_episodes: int, ratios: dict[str, float]
) -> dict[str, int]:
    raw_counts = {name: num_episodes * ratios[name] for name in SPLIT_NAMES}
    counts = {name: math.floor(raw_counts[name]) for name in SPLIT_NAMES}
    remainder = num_episodes - sum(counts.values())
    order = sorted(
        SPLIT_NAMES,
        key=lambda name: (-(raw_counts[name] - counts[name]), SPLIT_NAMES.index(name)),
    )
    for name in order[:remainder]:
        counts[name] += 1
    return counts


def _validate_ratios(
    train_ratio: float, val_ratio: float, test_ratio: float
) -> dict[str, float]:
    values = {
        "train": train_ratio,
        "val": val_ratio,
        "test": test_ratio,
    }
    if any(
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        or value < 0
        for value in values.values()
    ):
        raise ValueError("split ratios must be finite non-negative numbers")
    if not math.isclose(sum(values.values()), 1.0, rel_tol=0.0, abs_tol=1e-12):
        raise ValueError("train, val, and test ratios must sum to 1.0")
    return {name: float(values[name]) for name in SPLIT_NAMES}


def _ratios_from_mapping(value: Any) -> dict[str, float]:
    if not isinstance(value, dict) or set(value) != set(SPLIT_NAMES):
        raise ValueError("ratios must contain exactly train, val, and test")
    return _validate_ratios(value["train"], value["val"], value["test"])


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_json_object(path: Path, label: str) -> dict[str, Any]:
    if not path.is_file():
        raise ValueError(f"missing {label}: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"could not read {label} {path}: {error}") from error
    if not isinstance(value, dict):
        raise ValueError(f"{label} must contain a JSON object: {path}")
    return value


def _atomic_write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            json.dump(
                value,
                handle,
                indent=2,
                ensure_ascii=False,
                allow_nan=False,
            )
            handle.write("\n")
            temporary_path = Path(handle.name)
        os.replace(temporary_path, path)
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()


def _is_string_list(value: Any) -> bool:
    return isinstance(value, list) and all(isinstance(item, str) for item in value)


def _describe_set_difference(expected: set[str], actual: set[str]) -> str:
    parts = []
    missing = sorted(expected - actual)
    extra = sorted(actual - expected)
    if missing:
        parts.append("missing " + ", ".join(missing))
    if extra:
        parts.append("extra " + ", ".join(extra))
    return "; ".join(parts) or "unknown mismatch"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Split a global ViZDoom clip index by source episode."
    )
    parser.add_argument("dataset_index", type=Path, help="Path to all_clips.json")
    parser.add_argument("--output", type=Path, help="Split output directory")
    parser.add_argument("--train-ratio", type=float, default=DEFAULT_RATIOS["train"])
    parser.add_argument("--val-ratio", type=float, default=DEFAULT_RATIOS["val"])
    parser.add_argument("--test-ratio", type=float, default=DEFAULT_RATIOS["test"])
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Validate existing train/val/test manifests without rewriting them",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.validate_only:
        try:
            paths = split_dataset(
                args.dataset_index,
                output_dir=args.output,
                train_ratio=args.train_ratio,
                val_ratio=args.val_ratio,
                test_ratio=args.test_ratio,
                seed=args.seed,
            )
        except ValueError as error:
            raise SystemExit(str(error)) from error
        for split_name in SPLIT_NAMES:
            print(f"WROTE {paths[split_name]}")

    errors = validate_dataset_splits(args.dataset_index, args.output)
    if errors:
        print("INVALID dataset splits")
        for error in errors:
            print(f"  - {error}")
        raise SystemExit(1)
    print("VALID dataset splits")


if __name__ == "__main__":
    main()
