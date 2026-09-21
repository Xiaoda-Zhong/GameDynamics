#!/usr/bin/env python3
"""Build one deterministic clip index across a scenario's episodes."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
from pathlib import Path

try:
    from .process_dataset import (
        EpisodeDiscoveryError,
        discover_episode_directories,
    )
    from .validate_action_statistics import validate_action_statistics
    from .validate_depth_clips import validate_depth_clips
    from .validate_motion_scores import validate_motion_scores
    from .validate_rgb_clips import validate_rgb_clips
except ImportError:
    from process_dataset import EpisodeDiscoveryError, discover_episode_directories
    from validate_action_statistics import validate_action_statistics
    from validate_depth_clips import validate_depth_clips
    from validate_motion_scores import validate_motion_scores
    from validate_rgb_clips import validate_rgb_clips


DATASET_INDEX_VERSION = 1
PROCESS_REPORT_FILENAME = "process_report.json"
SUPPORTED_PROCESS_REPORT_VERSION = 1
EPISODE_DIRECTORY_PATTERN = re.compile(r"episode_(\d+)")
MOTION_FIELDS = (
    "motion_mean",
    "motion_std",
    "motion_min",
    "motion_max",
    "motion_pairwise",
)


def build_dataset_index(
    source_clips_root: Path,
    output_root: Path = Path("data/datasets"),
) -> Path:
    """Validate and combine all episode clip indices below one scenario root."""
    source_clips_root = source_clips_root.resolve()
    discovered = discover_episode_clip_indices(source_clips_root)
    scenario_name = source_clips_root.name
    process_info = _load_and_validate_process_report(
        source_clips_root,
        discovered,
    )

    episodes: list[dict] = []
    global_clips: list[dict] = []
    seen_source_episodes: dict[Path, str] = {}
    seen_global_ids: set[str] = set()
    declared_action_space: list[str] | None = None

    for episode_number, episode_name, clips_json in discovered:
        _validate_source_clip_index(clips_json)
        clip_index = _load_json_object(clips_json, "clip index")
        if clip_index.get("scenario_name") != scenario_name:
            raise ValueError(
                f"{clips_json}: scenario_name {clip_index.get('scenario_name')!r} "
                f"does not match source root {scenario_name!r}"
            )
        _validate_clip_processing_parameters(
            clip_index,
            clips_json,
            process_info["parameters"],
        )

        source_episode = _resolve_source_episode(clip_index, clips_json)
        reported_source_episode = process_info["succeeded_sources"][episode_name]
        if source_episode != reported_source_episode:
            raise ValueError(
                f"{clips_json}: source_episode {source_episode} does not match "
                f"the completed process report ({reported_source_episode})"
            )
        if source_episode.name != episode_name:
            raise ValueError(
                f"{clips_json}: source_episode name {source_episode.name!r} "
                f"does not match clip directory {episode_name!r}"
            )
        previous_name = seen_source_episodes.get(source_episode)
        if previous_name is not None:
            raise ValueError(
                f"duplicate source_episode reference in {previous_name} and "
                f"{episode_name}: {source_episode}"
            )
        seen_source_episodes[source_episode] = episode_name

        episode_metadata_path = source_episode / "metadata.json"
        episode_metadata = _load_json_object(
            episode_metadata_path,
            "source episode metadata",
        )
        provenance = episode_metadata.get("provenance")
        if not isinstance(provenance, dict):
            raise ValueError(
                f"{episode_metadata_path}: provenance must be an object"
            )
        if provenance.get("scenario_name") != scenario_name:
            raise ValueError(
                f"{episode_metadata_path}: provenance scenario_name does not "
                f"match {scenario_name!r}"
            )
        if episode_metadata.get("episode_index") != episode_number:
            raise ValueError(
                f"{episode_metadata_path}: episode_index does not match "
                f"directory identity {episode_number}"
            )
        episode_action_space = _action_labels(episode_metadata, episode_metadata_path)
        if declared_action_space is None:
            declared_action_space = episode_action_space
        elif episode_action_space != declared_action_space:
            raise ValueError(
                f"{episode_metadata_path}: declared action space differs from "
                "the other indexed episodes"
            )

        clips_json = clips_json.resolve()
        if (
            clips_json.parent.name != episode_name
            or clips_json.parent.parent != source_clips_root
        ):
            raise ValueError(
                f"{clips_json}: resolved clip index is outside the expected "
                f"{scenario_name}/{episode_name} directory"
            )
        clips_json_sha256 = _file_sha256(clips_json)
        episode_metadata_path = episode_metadata_path.resolve()
        episode_metadata_sha256 = _file_sha256(episode_metadata_path)
        clips = clip_index.get("clips")
        if not isinstance(clips, list):
            raise ValueError(f"{clips_json}: clips must be a list")

        episode_record = {
            "episode_name": episode_name,
            "source_episode": str(source_episode),
            "source_episode_metadata": str(episode_metadata_path),
            "source_episode_metadata_sha256": episode_metadata_sha256,
            "clips_json": str(clips_json),
            "clips_json_sha256": clips_json_sha256,
            "num_clips": len(clips),
            "termination_reason": episode_metadata.get("termination_reason"),
        }
        episodes.append(episode_record)

        for clip in clips:
            global_clip = _build_global_clip_record(
                scenario_name=scenario_name,
                episode_name=episode_name,
                source_episode=source_episode,
                episode_metadata_path=episode_metadata_path,
                episode_metadata_sha256=episode_metadata_sha256,
                clips_json=clips_json,
                clips_json_sha256=clips_json_sha256,
                clip=clip,
            )
            global_id = global_clip["global_clip_id"]
            if global_id in seen_global_ids:
                raise ValueError(f"duplicate global_clip_id: {global_id}")
            seen_global_ids.add(global_id)
            global_clips.append(global_clip)

    dataset_index = {
        "dataset_index_version": DATASET_INDEX_VERSION,
        "scenario_name": scenario_name,
        "source_clips_root": str(source_clips_root),
        "source_episodes_root": str(process_info["episodes_root"]),
        "process_report": str(process_info["path"]),
        "process_report_sha256": process_info["sha256"],
        "processing_parameters": process_info["parameters"],
        "num_discovered_episodes": len(process_info["discovered_names"]),
        "num_failed_episodes": len(process_info["failed_names"]),
        "failed_episode_names": process_info["failed_names"],
        "declared_action_space": declared_action_space,
        "num_episodes": len(episodes),
        "num_clips": len(global_clips),
        "episodes": episodes,
        "clips": global_clips,
    }
    output_path = output_root.resolve() / scenario_name / "all_clips.json"
    _write_json_atomic(output_path, dataset_index)
    return output_path


def validate_dataset_index(index_path: Path) -> list[str]:
    """Validate global records, source checksums, and referenced media."""
    index_path = _normalize_dataset_index_path(index_path)
    if not index_path.is_file():
        return [f"missing dataset index: {index_path}"]
    try:
        dataset = _load_json_object(index_path, "dataset index")
    except ValueError as error:
        return [str(error)]

    errors: list[str] = []
    if dataset.get("dataset_index_version") != DATASET_INDEX_VERSION:
        errors.append(
            f"dataset_index_version must equal {DATASET_INDEX_VERSION}"
        )
    scenario_name = dataset.get("scenario_name")
    if not isinstance(scenario_name, str) or not scenario_name:
        errors.append("scenario_name must be a non-empty string")
        scenario_name = None

    source_root_value = dataset.get("source_clips_root")
    source_root = _validate_absolute_directory(
        source_root_value,
        "source_clips_root",
        errors,
    )
    if (
        source_root is not None
        and scenario_name is not None
        and source_root.name != scenario_name
    ):
        errors.append("source_clips_root name does not match scenario_name")

    process_info: dict | None = None
    if source_root is not None:
        process_report = _validate_absolute_file(
            dataset.get("process_report"),
            "process_report",
            errors,
        )
        expected_process_report = (source_root / PROCESS_REPORT_FILENAME).resolve()
        if (
            process_report is not None
            and process_report != expected_process_report
        ):
            errors.append(
                "process_report must be the report in source_clips_root"
            )
        if process_report is not None:
            _check_sha256(
                process_report,
                dataset.get("process_report_sha256"),
                "process report",
                errors,
            )
            try:
                report_discovered = discover_episode_clip_indices(source_root)
                process_info = _load_and_validate_process_report(
                    source_root,
                    report_discovered,
                )
            except ValueError as error:
                errors.append(str(error))
                process_info = None
            else:
                if dataset.get("source_episodes_root") != str(
                    process_info["episodes_root"]
                ):
                    errors.append(
                        "source_episodes_root does not match process report"
                    )
                if dataset.get("processing_parameters") != process_info["parameters"]:
                    errors.append(
                        "processing_parameters do not match process report"
                    )
                if dataset.get("num_discovered_episodes") != len(
                    process_info["discovered_names"]
                ):
                    errors.append(
                        "num_discovered_episodes does not match process report"
                    )
                if dataset.get("num_failed_episodes") != len(
                    process_info["failed_names"]
                ):
                    errors.append(
                        "num_failed_episodes does not match process report"
                    )
                if dataset.get("failed_episode_names") != process_info["failed_names"]:
                    errors.append(
                        "failed_episode_names do not match process report"
                    )

    declared_action_space = dataset.get("declared_action_space")
    if (
        not isinstance(declared_action_space, list)
        or not declared_action_space
        or any(not isinstance(label, str) or not label for label in declared_action_space)
        or len(set(declared_action_space)) != len(declared_action_space)
    ):
        errors.append(
            "declared_action_space must be a non-empty list of unique strings"
        )
        declared_action_space = None

    episodes = dataset.get("episodes")
    clips = dataset.get("clips")
    if not isinstance(episodes, list):
        errors.append("episodes must be a list")
        return errors
    if not isinstance(clips, list):
        errors.append("clips must be a list")
        return errors
    if dataset.get("num_episodes") != len(episodes):
        errors.append("num_episodes does not match episodes length")
    if dataset.get("num_clips") != len(clips):
        errors.append("num_clips does not match clips length")

    episode_sources: set[Path] = set()
    episode_clip_indices: set[Path] = set()
    episode_names: set[str] = set()
    numeric_identities: set[int] = set()
    expected_records: dict[tuple[str, int], dict] = {}
    expected_order: list[tuple[str, int]] = []
    previous_episode_number = -1

    for position, episode in enumerate(episodes):
        if not isinstance(episode, dict):
            errors.append(f"episode record {position} must be an object")
            continue
        episode_name = episode.get("episode_name")
        match = (
            EPISODE_DIRECTORY_PATTERN.fullmatch(episode_name)
            if isinstance(episode_name, str)
            else None
        )
        if match is None:
            errors.append(f"episode record {position} has an invalid episode_name")
            continue
        episode_number = int(match.group(1))
        if episode_name in episode_names:
            errors.append(f"duplicate episode_name: {episode_name}")
        episode_names.add(episode_name)
        if episode_number in numeric_identities:
            errors.append(f"duplicate numeric episode identity: {episode_number}")
        numeric_identities.add(episode_number)
        if episode_number <= previous_episode_number:
            errors.append("episode records are not in numeric order")
        previous_episode_number = episode_number

        source_episode = _validate_absolute_directory(
            episode.get("source_episode"),
            f"episode {episode_name} source_episode",
            errors,
        )
        source_metadata_path = _validate_absolute_file(
            episode.get("source_episode_metadata"),
            f"episode {episode_name} source_episode_metadata",
            errors,
        )
        clips_json = _validate_absolute_file(
            episode.get("clips_json"),
            f"episode {episode_name} clips_json",
            errors,
        )
        if source_episode is None or source_metadata_path is None or clips_json is None:
            continue
        if source_episode in episode_sources:
            errors.append(f"duplicate source_episode reference: {source_episode}")
        episode_sources.add(source_episode)
        if clips_json in episode_clip_indices:
            errors.append(f"duplicate clips_json reference: {clips_json}")
        episode_clip_indices.add(clips_json)
        if source_episode.name != episode_name:
            errors.append(
                f"episode {episode_name} source_episode has a different name"
            )
        if process_info is not None:
            reported_source = process_info["succeeded_sources"].get(episode_name)
            if reported_source != source_episode:
                errors.append(
                    f"episode {episode_name} source_episode does not match "
                    "process report"
                )
        if source_metadata_path != (source_episode / "metadata.json").resolve():
            errors.append(
                f"episode {episode_name} source_episode_metadata has the wrong path"
            )
        if clips_json.parent.name != episode_name:
            errors.append(f"episode {episode_name} clips_json has the wrong parent")
        if source_root is not None and clips_json.parent.parent != source_root:
            errors.append(f"episode {episode_name} clips_json is outside source_clips_root")

        _check_sha256(
            source_metadata_path,
            episode.get("source_episode_metadata_sha256"),
            f"episode {episode_name} source metadata",
            errors,
        )
        _check_sha256(
            clips_json,
            episode.get("clips_json_sha256"),
            f"episode {episode_name} clip index",
            errors,
        )
        try:
            source_metadata = _load_json_object(
                source_metadata_path,
                "source episode metadata",
            )
            source_index = _load_json_object(clips_json, "source clip index")
        except ValueError as error:
            errors.append(str(error))
            continue

        provenance = source_metadata.get("provenance")
        if not isinstance(provenance, dict):
            errors.append(f"episode {episode_name} source provenance is invalid")
            continue
        if provenance.get("scenario_name") != scenario_name:
            errors.append(f"episode {episode_name} source scenario does not match")
        if source_metadata.get("episode_index") != episode_number:
            errors.append(f"episode {episode_name} source episode_index does not match")
        if episode.get("termination_reason") != source_metadata.get(
            "termination_reason"
        ):
            errors.append(f"episode {episode_name} termination_reason does not match")
        try:
            labels = _action_labels(source_metadata, source_metadata_path)
        except ValueError as error:
            errors.append(str(error))
        else:
            if declared_action_space is not None and labels != declared_action_space:
                errors.append(
                    f"episode {episode_name} action space differs from declared_action_space"
                )

        if source_index.get("scenario_name") != scenario_name:
            errors.append(f"episode {episode_name} clip index scenario does not match")
        if process_info is not None:
            try:
                _validate_clip_processing_parameters(
                    source_index,
                    clips_json,
                    process_info["parameters"],
                )
            except ValueError as error:
                errors.append(str(error))
        try:
            indexed_source_episode = _resolve_source_episode(source_index, clips_json)
        except ValueError as error:
            errors.append(str(error))
            continue
        if indexed_source_episode != source_episode:
            errors.append(f"episode {episode_name} clip index source_episode differs")
        source_clips = source_index.get("clips")
        if not isinstance(source_clips, list):
            errors.append(f"episode {episode_name} source clips must be a list")
            continue
        if episode.get("num_clips") != len(source_clips):
            errors.append(f"episode {episode_name} num_clips does not match source")

        source_metadata_sha256 = _file_sha256(source_metadata_path)
        clips_json_sha256 = _file_sha256(clips_json)
        for source_clip in source_clips:
            if not isinstance(source_clip, dict):
                errors.append(f"episode {episode_name} contains a non-object clip")
                continue
            local_clip_id = source_clip.get("clip_id")
            if type(local_clip_id) is not int or local_clip_id < 0:
                errors.append(f"episode {episode_name} contains an invalid clip_id")
                continue
            key = (episode_name, local_clip_id)
            if key in expected_records:
                errors.append(
                    f"episode {episode_name} has duplicate local clip {local_clip_id}"
                )
                continue
            try:
                expected_records[key] = _build_global_clip_record(
                    scenario_name=scenario_name or "",
                    episode_name=episode_name,
                    source_episode=source_episode,
                    episode_metadata_path=source_metadata_path,
                    episode_metadata_sha256=source_metadata_sha256,
                    clips_json=clips_json,
                    clips_json_sha256=clips_json_sha256,
                    clip=source_clip,
                )
            except ValueError as error:
                errors.append(str(error))
                continue
            expected_order.append(key)

    if source_root is not None:
        try:
            discovered_paths = {
                path.resolve()
                for _, _, path in discover_episode_clip_indices(source_root)
            }
        except ValueError as error:
            errors.append(str(error))
        else:
            if discovered_paths != episode_clip_indices:
                errors.append(
                    "episode records do not exactly cover source_clips_root"
                )

    actual_order: list[tuple[str, int]] = []
    seen_global_ids: set[str] = set()
    seen_clip_keys: set[tuple[str, int]] = set()
    for position, clip in enumerate(clips):
        if not isinstance(clip, dict):
            errors.append(f"global clip record {position} must be an object")
            continue
        episode_name = clip.get("episode_name")
        local_clip_id = clip.get("local_clip_id")
        if not isinstance(episode_name, str) or type(local_clip_id) is not int:
            errors.append(f"global clip record {position} has an invalid reference")
            continue
        key = (episode_name, local_clip_id)
        actual_order.append(key)
        if key in seen_clip_keys:
            errors.append(
                f"duplicate local clip reference: {episode_name} clip {local_clip_id}"
            )
        seen_clip_keys.add(key)
        global_clip_id = clip.get("global_clip_id")
        if not isinstance(global_clip_id, str) or not global_clip_id:
            errors.append(f"global clip record {position} has an invalid global_clip_id")
        elif global_clip_id in seen_global_ids:
            errors.append(f"duplicate global_clip_id: {global_clip_id}")
        else:
            seen_global_ids.add(global_clip_id)
        expected = expected_records.get(key)
        if expected is None:
            errors.append(
                f"global clip record {position} does not reference a source clip"
            )
            continue
        for field, expected_value in expected.items():
            if clip.get(field) != expected_value:
                errors.append(
                    f"global clip {expected['global_clip_id']} field {field} "
                    "does not match its source"
                )

    if actual_order != expected_order:
        errors.append("global clips are missing, extra, duplicated, or out of order")
    return errors


def _load_and_validate_process_report(
    source_clips_root: Path,
    discovered_clip_indices: list[tuple[int, str, Path]],
) -> dict:
    """Load a completed batch report and verify its exact episode coverage."""
    source_clips_root = source_clips_root.resolve()
    report_path = (source_clips_root / PROCESS_REPORT_FILENAME).resolve()
    report = _load_json_object(report_path, "process report")
    errors: list[str] = []

    if report.get("report_version") != SUPPORTED_PROCESS_REPORT_VERSION:
        errors.append(
            "report_version must equal "
            f"{SUPPORTED_PROCESS_REPORT_VERSION}"
        )
    if report.get("status") != "complete":
        errors.append("status must be 'complete'")

    scenario_name = source_clips_root.name
    if report.get("scenario_name") != scenario_name:
        errors.append(
            f"scenario_name must equal source root name {scenario_name!r}"
        )

    episodes_root = _report_absolute_directory(
        report.get("episodes_root"),
        "episodes_root",
        errors,
    )
    if episodes_root is not None and episodes_root.name != scenario_name:
        errors.append("episodes_root name must match scenario_name")

    clips_output = _report_absolute_directory(
        report.get("clips_output"),
        "clips_output",
        errors,
    )
    if (
        clips_output is not None
        and (clips_output / scenario_name).resolve() != source_clips_root
    ):
        errors.append(
            "clips_output/scenario_name must equal source_clips_root"
        )

    parameters = report.get("parameters")
    if not isinstance(parameters, dict):
        errors.append("parameters must be an object")
        parameters = None
    else:
        clip_length = parameters.get("clip_length")
        stride = parameters.get("stride")
        fps = parameters.get("fps")
        policy = parameters.get("invalid_episode_policy")
        reconciliation = parameters.get("output_reconciliation")
        if type(clip_length) is not int or clip_length < 2:
            errors.append("parameters.clip_length must be an integer at least 2")
        if type(stride) is not int or stride < 1:
            errors.append("parameters.stride must be a positive integer")
        if (
            isinstance(fps, bool)
            or not isinstance(fps, (int, float))
            or not math.isfinite(fps)
            or fps <= 0
        ):
            errors.append("parameters.fps must be a positive finite number")
        if policy not in {"fail_fast", "skip_invalid"}:
            errors.append(
                "parameters.invalid_episode_policy must be 'fail_fast' "
                "or 'skip_invalid'"
            )
        if reconciliation != "remove_absent_invalid_or_failed_episode_outputs":
            errors.append(
                "parameters.output_reconciliation has an unsupported value"
            )

    if "dataset_failure" in report:
        errors.append("a completed process report cannot contain dataset_failure")

    discovered_entries = _validate_process_episode_list(
        report.get("discovered"),
        "discovered",
        episodes_root,
        source_clips_root,
        require_clips_json=False,
        errors=errors,
    )
    succeeded_entries = _validate_process_episode_list(
        report.get("succeeded"),
        "succeeded",
        episodes_root,
        source_clips_root,
        require_clips_json=True,
        errors=errors,
    )
    failed_entries = _validate_process_episode_list(
        report.get("failed"),
        "failed",
        episodes_root,
        source_clips_root,
        require_clips_json=False,
        errors=errors,
    )

    discovered_names = list(discovered_entries)
    succeeded_names = list(succeeded_entries)
    failed_names = list(failed_entries)
    if discovered_names != sorted(
        discovered_names,
        key=lambda name: (
            int(EPISODE_DIRECTORY_PATTERN.fullmatch(name).group(1)),
            name,
        ),
    ):
        errors.append("discovered entries are not in numeric episode order")
    if succeeded_names != [
        name for name in discovered_names if name in succeeded_entries
    ]:
        errors.append("succeeded entries do not preserve discovered order")
    if failed_names != [name for name in discovered_names if name in failed_entries]:
        errors.append("failed entries do not preserve discovered order")

    discovered_set = set(discovered_entries)
    succeeded_set = set(succeeded_entries)
    failed_set = set(failed_entries)
    if succeeded_set & failed_set:
        errors.append("an episode cannot appear in both succeeded and failed")
    if succeeded_set | failed_set != discovered_set:
        errors.append(
            "succeeded and failed entries must exactly partition discovered"
        )

    if episodes_root is not None:
        try:
            current_source_episodes = discover_episode_directories(episodes_root)
        except EpisodeDiscoveryError as error:
            errors.append(f"could not verify current source episodes: {error}")
        else:
            current_source_names = [path.name for path in current_source_episodes]
            if current_source_names != discovered_names:
                errors.append(
                    "discovered report entries do not exactly match the current "
                    "source episode directories"
                )

    if parameters is not None:
        policy = parameters.get("invalid_episode_policy")
        if policy == "fail_fast" and failed_entries:
            errors.append("a completed fail_fast report cannot contain failures")
        if policy == "skip_invalid":
            failed_by_name = {
                entry.get("episode_name"): entry
                for entry in report.get("failed", [])
                if isinstance(entry, dict)
            }
            for name in failed_entries:
                entry = failed_by_name[name]
                if entry.get("stage") != "validate_episode":
                    errors.append(
                        f"failed episode {name} was not skipped during "
                        "validate_episode"
                    )

    actual_succeeded_names = [name for _, name, _ in discovered_clip_indices]
    if succeeded_names != actual_succeeded_names:
        errors.append(
            "succeeded report entries do not exactly match episode clip directories"
        )

    if errors:
        details = "\n".join(f"  - {error}" for error in errors)
        raise ValueError(f"invalid process report {report_path}:\n{details}")

    return {
        "path": report_path,
        "sha256": _file_sha256(report_path),
        "episodes_root": episodes_root,
        "parameters": parameters,
        "discovered_names": discovered_names,
        "succeeded_names": succeeded_names,
        "failed_names": failed_names,
        "succeeded_sources": {
            name: succeeded_entries[name]["source_episode"]
            for name in succeeded_names
        },
    }


def _validate_process_episode_list(
    value: object,
    label: str,
    episodes_root: Path | None,
    source_clips_root: Path,
    *,
    require_clips_json: bool,
    errors: list[str],
) -> dict[str, dict]:
    """Validate report episode records and return them keyed by name."""
    if not isinstance(value, list):
        errors.append(f"{label} must be a list")
        return {}

    entries: dict[str, dict] = {}
    numeric_identities: set[int] = set()
    source_paths: set[Path] = set()
    for position, entry in enumerate(value):
        entry_label = f"{label}[{position}]"
        if not isinstance(entry, dict):
            errors.append(f"{entry_label} must be an object")
            continue
        episode_name = entry.get("episode_name")
        match = (
            EPISODE_DIRECTORY_PATTERN.fullmatch(episode_name)
            if isinstance(episode_name, str)
            else None
        )
        if match is None:
            errors.append(f"{entry_label}.episode_name is invalid")
            continue
        episode_number = int(match.group(1))
        if entry.get("episode_index") != episode_number:
            errors.append(f"{entry_label}.episode_index does not match its name")
        if episode_name in entries:
            errors.append(f"{label} contains duplicate episode {episode_name}")
            continue
        if episode_number in numeric_identities:
            errors.append(
                f"{label} contains duplicate numeric episode identity "
                f"{episode_number}"
            )
        numeric_identities.add(episode_number)

        source_episode = _report_absolute_directory(
            entry.get("source_episode"),
            f"{entry_label}.source_episode",
            errors,
        )
        if source_episode is not None:
            if source_episode in source_paths:
                errors.append(f"{label} contains duplicate source_episode")
            source_paths.add(source_episode)
            if source_episode.name != episode_name:
                errors.append(f"{entry_label}.source_episode has the wrong name")
            if (
                episodes_root is not None
                and source_episode != (episodes_root / episode_name).resolve()
            ):
                errors.append(
                    f"{entry_label}.source_episode is outside episodes_root"
                )

        if require_clips_json:
            clips_json = _report_absolute_file(
                entry.get("clips_json"),
                f"{entry_label}.clips_json",
                errors,
            )
            expected_clips_json = (
                source_clips_root / episode_name / "clips.json"
            ).resolve()
            if clips_json is not None and clips_json != expected_clips_json:
                errors.append(f"{entry_label}.clips_json has the wrong path")

        entries[episode_name] = {
            "source_episode": source_episode,
        }
    return entries


def _report_absolute_directory(
    value: object,
    label: str,
    errors: list[str],
) -> Path | None:
    if not isinstance(value, str) or not value:
        errors.append(f"{label} must be a non-empty absolute path string")
        return None
    path = Path(value)
    if not path.is_absolute():
        errors.append(f"{label} must be absolute")
        return None
    resolved = path.resolve()
    if not resolved.is_dir():
        errors.append(f"{label} is missing: {resolved}")
        return None
    return resolved


def _report_absolute_file(
    value: object,
    label: str,
    errors: list[str],
) -> Path | None:
    if not isinstance(value, str) or not value:
        errors.append(f"{label} must be a non-empty absolute path string")
        return None
    path = Path(value)
    if not path.is_absolute():
        errors.append(f"{label} must be absolute")
        return None
    resolved = path.resolve()
    if not resolved.is_file():
        errors.append(f"{label} is missing: {resolved}")
        return None
    return resolved


def discover_episode_clip_indices(
    source_clips_root: Path,
) -> list[tuple[int, str, Path]]:
    """Return episode clip indices ordered by numeric episode identity."""
    source_clips_root = source_clips_root.resolve()
    if not source_clips_root.is_dir():
        raise ValueError(f"source clips root is not a directory: {source_clips_root}")

    discovered: list[tuple[int, str, Path]] = []
    identities: dict[int, str] = {}
    for entry in sorted(source_clips_root.iterdir(), key=lambda path: path.name):
        if not entry.name.startswith("episode_"):
            continue
        match = EPISODE_DIRECTORY_PATTERN.fullmatch(entry.name)
        if match is None or not entry.is_dir():
            raise ValueError(f"malformed episode entry: {entry}")
        episode_number = int(match.group(1))
        previous_name = identities.get(episode_number)
        if previous_name is not None:
            raise ValueError(
                f"duplicate numeric episode identity {episode_number}: "
                f"{previous_name} and {entry.name}"
            )
        identities[episode_number] = entry.name
        clips_json = entry / "clips.json"
        if not clips_json.is_file():
            raise ValueError(f"episode directory has no clips.json: {entry}")
        discovered.append((episode_number, entry.name, clips_json))

    if not discovered:
        raise ValueError(f"no episode_<digits>/clips.json entries in {source_clips_root}")
    return sorted(discovered, key=lambda item: (item[0], item[1]))


def _validate_source_clip_index(clips_json: Path) -> None:
    validators = (
        ("RGB", validate_rgb_clips),
        ("depth", validate_depth_clips),
        ("motion", validate_motion_scores),
        ("action", validate_action_statistics),
    )
    errors: list[str] = []
    for label, validator in validators:
        try:
            validation_errors = validator(clips_json)
        except Exception as error:  # Surface malformed input with validator context.
            errors.append(f"{label} validator failed: {error}")
        else:
            errors.extend(f"{label}: {error}" for error in validation_errors)
    if errors:
        details = "\n".join(f"  - {error}" for error in errors)
        raise ValueError(f"invalid source clip index {clips_json}:\n{details}")


def _build_global_clip_record(
    *,
    scenario_name: str,
    episode_name: str,
    source_episode: Path,
    episode_metadata_path: Path,
    episode_metadata_sha256: str,
    clips_json: Path,
    clips_json_sha256: str,
    clip: dict,
) -> dict:
    local_clip_id = clip.get("clip_id")
    if type(local_clip_id) is not int or local_clip_id < 0:
        raise ValueError(f"{clips_json}: clip_id must be a non-negative integer")
    global_clip_id = (
        f"{scenario_name}_{episode_name}_clip_{local_clip_id:06d}"
    )

    rgb_video = _resolve_required_media_path(
        clips_json.parent,
        clip.get("rgb_video"),
        f"clip {local_clip_id} rgb_video",
    )
    depth_frames_value = clip.get("depth_frames")
    if depth_frames_value is None:
        depth_frames = None
    elif isinstance(depth_frames_value, list):
        depth_frames = [
            str(
                _resolve_required_media_path(
                    clips_json.parent,
                    value,
                    f"clip {local_clip_id} depth frame {position}",
                )
            )
            for position, value in enumerate(depth_frames_value)
        ]
    else:
        raise ValueError(
            f"{clips_json}: clip {local_clip_id} depth_frames must be a list or null"
        )

    motion_statistics = {}
    for field in MOTION_FIELDS:
        if field not in clip:
            raise ValueError(
                f"{clips_json}: clip {local_clip_id} is missing {field}"
            )
        motion_statistics[field.removeprefix("motion_")] = clip[field]
    action_statistics = clip.get("action_statistics")
    if not isinstance(action_statistics, dict):
        raise ValueError(
            f"{clips_json}: clip {local_clip_id} action_statistics must be an object"
        )

    return {
        "global_clip_id": global_clip_id,
        "scenario_name": scenario_name,
        "episode_name": episode_name,
        "source_episode": str(source_episode),
        "source_episode_metadata": str(episode_metadata_path),
        "source_episode_metadata_sha256": episode_metadata_sha256,
        "local_clip_id": local_clip_id,
        "clips_json": str(clips_json),
        "clips_json_sha256": clips_json_sha256,
        "rgb_video": str(rgb_video),
        "fps": clip.get("fps"),
        "depth_frames": depth_frames,
        "num_depth_frames": clip.get("num_depth_frames"),
        "start_observation_index": clip.get("start_observation_index"),
        "end_observation_index": clip.get("end_observation_index"),
        "observation_indices": clip.get("observation_indices"),
        "transition_steps": clip.get("transition_steps"),
        "actions": clip.get("actions"),
        "action_vectors": clip.get("action_vectors"),
        "chunk_ids": clip.get("chunk_ids"),
        "num_frames": clip.get("num_frames"),
        "num_actions": clip.get("num_actions"),
        "motion_statistics": motion_statistics,
        "action_statistics": action_statistics,
    }


def _resolve_source_episode(clip_index: dict, clips_json: Path) -> Path:
    value = clip_index.get("source_episode")
    if not isinstance(value, str) or not value:
        raise ValueError(f"{clips_json}: source_episode must be a path string")
    source_episode = Path(value).expanduser().resolve()
    if not source_episode.is_dir():
        raise ValueError(f"{clips_json}: source_episode is missing: {source_episode}")
    return source_episode


def _validate_clip_processing_parameters(
    clip_index: dict,
    clips_json: Path,
    parameters: dict,
) -> None:
    """Ensure a per-episode index was built with the reported batch settings."""
    mismatches: list[str] = []
    if clip_index.get("clip_length") != parameters["clip_length"]:
        mismatches.append("clip_length")
    if clip_index.get("stride") != parameters["stride"]:
        mismatches.append("stride")

    expected_fps = float(parameters["fps"])
    clips = clip_index.get("clips")
    if isinstance(clips, list):
        for position, clip in enumerate(clips):
            if not isinstance(clip, dict):
                continue
            fps = clip.get("fps")
            if (
                isinstance(fps, bool)
                or not isinstance(fps, (int, float))
                or not math.isfinite(fps)
                or not math.isclose(float(fps), expected_fps, rel_tol=0.0, abs_tol=1e-9)
            ):
                mismatches.append(f"clips[{position}].fps")
                break

    if mismatches:
        raise ValueError(
            f"{clips_json}: {', '.join(mismatches)} do not match "
            "the completed process report"
        )


def _resolve_required_media_path(root: Path, value: object, label: str) -> Path:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{label} must be a non-empty relative path string")
    relative_path = Path(value)
    if relative_path.is_absolute() or ".." in relative_path.parts:
        raise ValueError(f"{label} is not a safe relative path: {value!r}")
    resolved = (root / relative_path).resolve()
    if not resolved.is_file():
        raise ValueError(f"{label} is missing: {resolved}")
    return resolved


def _action_labels(metadata: dict, metadata_path: Path) -> list[str]:
    action_space = metadata.get("action_space")
    if not isinstance(action_space, list) or not action_space:
        raise ValueError(f"{metadata_path}: action_space must be a non-empty list")
    labels = []
    for position, action in enumerate(action_space):
        if not isinstance(action, dict):
            raise ValueError(
                f"{metadata_path}: action_space entry {position} must be an object"
            )
        label = action.get("label")
        if not isinstance(label, str) or not label:
            raise ValueError(
                f"{metadata_path}: action_space entry {position} has no label"
            )
        labels.append(label)
    if len(labels) != len(set(labels)):
        raise ValueError(f"{metadata_path}: action_space labels are duplicated")
    return labels


def _validate_absolute_directory(
    value: object,
    label: str,
    errors: list[str],
) -> Path | None:
    if not isinstance(value, str) or not value:
        errors.append(f"{label} must be a non-empty absolute path string")
        return None
    path = Path(value)
    if not path.is_absolute():
        errors.append(f"{label} must be absolute")
        return None
    path = path.resolve()
    if not path.is_dir():
        errors.append(f"{label} is missing: {path}")
        return None
    return path


def _validate_absolute_file(
    value: object,
    label: str,
    errors: list[str],
) -> Path | None:
    if not isinstance(value, str) or not value:
        errors.append(f"{label} must be a non-empty absolute path string")
        return None
    path = Path(value)
    if not path.is_absolute():
        errors.append(f"{label} must be absolute")
        return None
    path = path.resolve()
    if not path.is_file():
        errors.append(f"{label} is missing: {path}")
        return None
    return path


def _check_sha256(
    path: Path,
    declared: object,
    label: str,
    errors: list[str],
) -> None:
    if not isinstance(declared, str) or len(declared) != 64:
        errors.append(f"{label} SHA-256 is invalid")
    elif declared != _file_sha256(path):
        errors.append(f"{label} SHA-256 does not match")


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
    digest = hashlib.sha256()
    with path.open("rb") as input_file:
        for chunk in iter(lambda: input_file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json_atomic(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_name(f".{path.name}.tmp")
    try:
        temporary_path.write_text(
            json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary_path, path)
    finally:
        if temporary_path.exists():
            temporary_path.unlink()


def _normalize_dataset_index_path(path: Path) -> Path:
    return path / "all_clips.json" if path.is_dir() else path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a validated global clip index for one scenario."
    )
    parser.add_argument("source_clips_root", type=Path)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/datasets"),
        help="dataset output root (default: data/datasets)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    try:
        output_path = build_dataset_index(
            args.source_clips_root,
            output_root=args.output,
        )
    except ValueError as error:
        raise SystemExit(str(error)) from error
    dataset_index = _load_json_object(output_path, "dataset index")
    print(
        f"WROTE {output_path} "
        f"({dataset_index['num_episodes']} episodes, "
        f"{dataset_index['num_clips']} clips)"
    )


if __name__ == "__main__":
    main()
