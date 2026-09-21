#!/usr/bin/env python3
"""Run the per-episode clip pipeline over one scenario dataset."""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import shutil
from pathlib import Path
from typing import Callable, TypeVar

if __package__:
    from .build_clip_index import (
        DEFAULT_CLIP_LENGTH,
        DEFAULT_STRIDE,
        build_clip_index,
    )
    from .build_depth_clips import build_depth_clips
    from .compute_action_statistics import compute_action_statistics
    from .compute_motion_scores import compute_motion_scores
    from .encode_rgb_clips import DEFAULT_FPS, encode_rgb_clips
    from .validate_episode import validate_episode
else:
    from build_clip_index import (
        DEFAULT_CLIP_LENGTH,
        DEFAULT_STRIDE,
        build_clip_index,
    )
    from build_depth_clips import build_depth_clips
    from compute_action_statistics import compute_action_statistics
    from compute_motion_scores import compute_motion_scores
    from encode_rgb_clips import DEFAULT_FPS, encode_rgb_clips
    from validate_episode import validate_episode


EPISODE_DIRECTORY_PATTERN = re.compile(r"episode_(\d+)")
REPORT_FILENAME = "process_report.json"
REPORT_VERSION = 1

T = TypeVar("T")


class DatasetProcessingError(RuntimeError):
    """Raised after a batch-processing failure has been written to the report."""


class EpisodeDiscoveryError(ValueError):
    """Raised when episode directories cannot be identified unambiguously."""


class EpisodeStageError(RuntimeError):
    """Associate an exception with the pipeline stage that raised it."""

    def __init__(self, stage: str, error: Exception) -> None:
        self.stage = stage
        self.error = error
        super().__init__(str(error))


def discover_episode_directories(episodes_root: Path) -> list[Path]:
    """Return matching episode directories in numeric, deterministic order."""
    episodes_root = Path(episodes_root).resolve()
    if not episodes_root.is_dir():
        raise EpisodeDiscoveryError(
            f"episodes root is not a directory: {episodes_root}"
        )

    try:
        children = list(episodes_root.iterdir())
    except OSError as error:
        raise EpisodeDiscoveryError(
            f"could not list episodes root {episodes_root}: {error}"
        ) from error

    indexed_directories: list[tuple[int, str, Path]] = []
    for path in children:
        if not path.name.startswith("episode_"):
            continue
        match = EPISODE_DIRECTORY_PATTERN.fullmatch(path.name)
        if match is None or not path.is_dir():
            raise EpisodeDiscoveryError(f"malformed episode entry: {path}")
        indexed_directories.append((int(match.group(1)), path.name, path))

    indexed_directories.sort(key=lambda item: (item[0], item[1]))
    for previous, current in zip(
        indexed_directories,
        indexed_directories[1:],
    ):
        if previous[0] == current[0]:
            raise EpisodeDiscoveryError(
                "duplicate numeric episode identity "
                f"{current[0]}: {previous[1]} and {current[1]}"
            )

    if not indexed_directories:
        raise EpisodeDiscoveryError(
            f"no episode_<digits> directories in {episodes_root}"
        )

    return [item[2] for item in indexed_directories]


def process_dataset(
    episodes_root: Path,
    clips_output: Path = Path("data/clips"),
    clip_length: int = DEFAULT_CLIP_LENGTH,
    stride: int = DEFAULT_STRIDE,
    fps: float = DEFAULT_FPS,
    skip_invalid: bool = False,
) -> Path:
    """Process every discovered episode and return the batch report path."""
    episodes_root = Path(episodes_root).resolve()
    clips_output = Path(clips_output).resolve()
    scenario_name = episodes_root.name
    scenario_output = clips_output / scenario_name

    _validate_parameters(clip_length, stride, fps)
    _validate_output_separation(episodes_root, scenario_output)
    report_path = scenario_output / REPORT_FILENAME
    report = _new_report(
        episodes_root=episodes_root,
        clips_output=clips_output,
        scenario_name=scenario_name,
        clip_length=clip_length,
        stride=stride,
        fps=fps,
        skip_invalid=skip_invalid,
    )

    try:
        episode_directories = discover_episode_directories(episodes_root)
    except EpisodeDiscoveryError as error:
        report["status"] = "failed"
        report["dataset_failure"] = {
            "stage": "discover_episodes",
            "error": str(error),
            "error_type": type(error).__name__,
        }
        _write_report_atomic(report_path, report)
        raise DatasetProcessingError(
            f"dataset discovery failed: {error}"
        ) from error

    for episode_dir in episode_directories:
        _validate_output_separation(episode_dir, scenario_output)

    report["discovered"] = [
        _episode_record(path) for path in episode_directories
    ]
    _write_report_atomic(report_path, report)

    try:
        _reconcile_derived_episode_outputs(
            scenario_output,
            expected_episode_names={
                path.name for path in episode_directories
            },
        )
    except OSError as error:
        report["status"] = "failed"
        report["dataset_failure"] = {
            "stage": "reconcile_outputs",
            "error": str(error),
            "error_type": type(error).__name__,
        }
        _write_report_atomic(report_path, report)
        raise DatasetProcessingError(
            f"could not reconcile derived episode outputs: {error}"
        ) from error

    for episode_dir in episode_directories:
        episode_record = _episode_record(episode_dir)
        derived_episode_dir = scenario_output / episode_dir.name
        validation_errors = _validate_source_episode(
            episode_dir,
            expected_scenario_name=scenario_name,
        )
        if validation_errors:
            error_message = _format_validation_errors(validation_errors)
            failure = {
                **episode_record,
                "stage": "validate_episode",
                "error": error_message,
                "validation_errors": validation_errors,
            }
            try:
                _remove_derived_path(derived_episode_dir)
            except OSError as cleanup_error:
                failure["output_cleanup_error"] = str(cleanup_error)
                failure["output_cleanup_error_type"] = type(
                    cleanup_error
                ).__name__
                report["failed"].append(failure)
                report["status"] = "failed"
                _write_report_atomic(report_path, report)
                raise DatasetProcessingError(
                    f"{episode_dir.name} failed validation and its derived "
                    f"output could not be removed: {cleanup_error}"
                ) from cleanup_error
            report["failed"].append(failure)
            if skip_invalid:
                _write_report_atomic(report_path, report)
                continue
            report["status"] = "failed"
            _write_report_atomic(report_path, report)
            raise DatasetProcessingError(
                f"{episode_dir.name} failed during validate_episode:\n"
                f"{error_message}"
            )

        try:
            clip_index_path = _run_stage(
                "build_clip_index",
                lambda: build_clip_index(
                    episode_dir=episode_dir,
                    output_root=clips_output,
                    clip_length=clip_length,
                    stride=stride,
                ),
            )
            _run_stage(
                "encode_rgb_clips",
                lambda: encode_rgb_clips(clip_index_path, fps=fps),
            )
            _run_stage(
                "build_depth_clips",
                lambda: build_depth_clips(clip_index_path),
            )
            _run_stage(
                "compute_motion_scores",
                lambda: compute_motion_scores(clip_index_path),
            )
            _run_stage(
                "compute_action_statistics",
                lambda: compute_action_statistics(clip_index_path),
            )
        except EpisodeStageError as error:
            failure = {
                **episode_record,
                "stage": error.stage,
                "error": str(error.error),
                "error_type": type(error.error).__name__,
            }
            try:
                _remove_derived_path(derived_episode_dir)
            except OSError as cleanup_error:
                failure["output_cleanup_error"] = str(cleanup_error)
                failure["output_cleanup_error_type"] = type(
                    cleanup_error
                ).__name__
            report["failed"].append(failure)
            report["status"] = "failed"
            _write_report_atomic(report_path, report)
            cleanup_suffix = (
                "; derived output cleanup also failed: "
                f"{failure['output_cleanup_error']}"
                if "output_cleanup_error" in failure
                else ""
            )
            raise DatasetProcessingError(
                f"{episode_dir.name} failed during {error.stage}: "
                f"{error.error}{cleanup_suffix}"
            ) from error.error

        report["succeeded"].append(
            {
                **episode_record,
                "clips_json": str(clip_index_path.resolve()),
            }
        )
        _write_report_atomic(report_path, report)

    report["status"] = "complete"
    _write_report_atomic(report_path, report)
    return report_path


def _validate_source_episode(
    episode_dir: Path,
    expected_scenario_name: str,
) -> list[str]:
    try:
        errors = list(validate_episode(episode_dir))
    except Exception as error:
        return [
            "validation raised "
            f"{type(error).__name__}: {error}"
        ]

    metadata_path = episode_dir / "metadata.json"
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        if not errors:
            errors.append(f"could not confirm scenario provenance: {error}")
        return errors

    provenance = metadata.get("provenance")
    source_scenario_name = (
        provenance.get("scenario_name")
        if isinstance(provenance, dict)
        else None
    )
    if source_scenario_name != expected_scenario_name:
        errors.append(
            "provenance scenario_name does not match episodes root: "
            f"expected {expected_scenario_name!r}, got {source_scenario_name!r}"
        )
    return errors


def _run_stage(stage: str, operation: Callable[[], T]) -> T:
    try:
        return operation()
    except Exception as error:
        raise EpisodeStageError(stage, error) from error


def _new_report(
    *,
    episodes_root: Path,
    clips_output: Path,
    scenario_name: str,
    clip_length: int,
    stride: int,
    fps: float,
    skip_invalid: bool,
) -> dict:
    serialized_fps: int | float = int(fps) if float(fps).is_integer() else fps
    return {
        "report_version": REPORT_VERSION,
        "status": "running",
        "scenario_name": scenario_name,
        "episodes_root": str(episodes_root),
        "clips_output": str(clips_output),
        "parameters": {
            "clip_length": clip_length,
            "stride": stride,
            "fps": serialized_fps,
            "invalid_episode_policy": (
                "skip_invalid" if skip_invalid else "fail_fast"
            ),
            "output_reconciliation": (
                "remove_absent_invalid_or_failed_episode_outputs"
            ),
        },
        "discovered": [],
        "succeeded": [],
        "failed": [],
    }


def _validate_output_separation(
    source_path: Path,
    scenario_output: Path,
) -> None:
    """Reject layouts where derived-output cleanup could reach source data."""
    resolved_source = source_path.resolve()
    resolved_output = scenario_output.resolve()
    if _path_contains(resolved_source, resolved_output) or _path_contains(
        resolved_output,
        resolved_source,
    ):
        raise ValueError(
            "source episodes root and scenario output must not overlap: "
            f"source={resolved_source}, output={resolved_output}"
        )


def _path_contains(parent: Path, child: Path) -> bool:
    try:
        child.relative_to(parent)
    except ValueError:
        return False
    return True


def _reconcile_derived_episode_outputs(
    scenario_output: Path,
    expected_episode_names: set[str],
) -> None:
    """Remove derived episode entries that have no current source episode."""
    scenario_output.mkdir(parents=True, exist_ok=True)
    for path in scenario_output.iterdir():
        if not path.name.startswith("episode_"):
            continue
        if path.name not in expected_episode_names:
            _remove_derived_path(path)
            continue
        if path.is_symlink() or not path.is_dir():
            _remove_derived_path(path)


def _remove_derived_path(path: Path) -> None:
    """Remove one derived entry without following directory symlinks."""
    if path.is_symlink() or path.is_file():
        path.unlink()
    elif path.is_dir():
        shutil.rmtree(path)
    elif path.exists():
        path.unlink()


def _episode_record(episode_dir: Path) -> dict:
    match = EPISODE_DIRECTORY_PATTERN.fullmatch(episode_dir.name)
    if match is None:
        raise ValueError(f"invalid episode directory name: {episode_dir.name}")
    return {
        "episode_index": int(match.group(1)),
        "episode_name": episode_dir.name,
        "source_episode": str(episode_dir.resolve()),
    }


def _format_validation_errors(errors: list[str]) -> str:
    return "\n".join(f"- {error}" for error in errors)


def _write_report_atomic(path: Path, report: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_name(f".{path.name}.tmp")
    try:
        temporary_path.write_text(
            json.dumps(report, indent=2, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary_path, path)
    finally:
        if temporary_path.exists():
            temporary_path.unlink()


def _validate_parameters(clip_length: int, stride: int, fps: float) -> None:
    if (
        isinstance(clip_length, bool)
        or not isinstance(clip_length, int)
        or clip_length < 2
    ):
        raise ValueError("clip_length must be at least 2 observations")
    if isinstance(stride, bool) or not isinstance(stride, int) or stride < 1:
        raise ValueError("stride must be at least 1 observation")
    if (
        isinstance(fps, bool)
        or not isinstance(fps, (int, float))
        or not math.isfinite(fps)
        or fps <= 0
    ):
        raise ValueError("fps must be a positive finite number")


def _positive_fps(value: str) -> float:
    try:
        fps = float(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("fps must be a number") from error
    if not math.isfinite(fps) or fps <= 0:
        raise argparse.ArgumentTypeError("fps must be a positive finite number")
    return fps


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Process all existing episodes for one scenario."
    )
    parser.add_argument("episodes_root", type=Path)
    parser.add_argument(
        "--clip-length",
        type=int,
        default=DEFAULT_CLIP_LENGTH,
    )
    parser.add_argument("--stride", type=int, default=DEFAULT_STRIDE)
    parser.add_argument("--fps", type=_positive_fps, default=DEFAULT_FPS)
    parser.add_argument(
        "--clips-output",
        type=Path,
        default=Path("data/clips"),
    )
    parser.add_argument(
        "--skip-invalid",
        action="store_true",
        help="record validation failures and continue with valid episodes",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    try:
        report_path = process_dataset(
            episodes_root=args.episodes_root,
            clips_output=args.clips_output,
            clip_length=args.clip_length,
            stride=args.stride,
            fps=args.fps,
            skip_invalid=args.skip_invalid,
        )
    except (ValueError, DatasetProcessingError) as error:
        raise SystemExit(str(error)) from error

    report = json.loads(report_path.read_text(encoding="utf-8"))
    print(
        f"PROCESSED {len(report['succeeded'])}/"
        f"{len(report['discovered'])} episodes for "
        f"{report['scenario_name']} ({len(report['failed'])} failed); "
        f"wrote {report_path}"
    )


if __name__ == "__main__":
    main()
