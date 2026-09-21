#!/usr/bin/env python3
"""Validate the CPU-only 17-frame Wan overfit manifest against source episodes."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from PIL import Image, UnidentifiedImageError

try:
    from .build_action_captions import generate_action_prompt, run_length_encode_actions
    from .prepare_wan_overfit import (
        DEFAULT_OUTPUT,
        DEFAULT_SUBSET,
        NUM_ACTIONS,
        NUM_FRAMES,
        REPOSITORY_ROOT,
        SELECTION_COUNT,
        SELECTION_SEED,
        evaluate_candidates,
        load_source_context,
        select_records,
    )
except ImportError:
    from build_action_captions import generate_action_prompt, run_length_encode_actions
    from prepare_wan_overfit import (
        DEFAULT_OUTPUT,
        DEFAULT_SUBSET,
        NUM_ACTIONS,
        NUM_FRAMES,
        REPOSITORY_ROOT,
        SELECTION_COUNT,
        SELECTION_SEED,
        evaluate_candidates,
        load_source_context,
        select_records,
    )


def _read_jsonl(path: Path) -> list[dict]:
    records = []
    try:
        with path.open("r", encoding="utf-8") as stream:
            for number, line in enumerate(stream, 1):
                if not line.strip():
                    raise ValueError(f"blank line {number} in {path}")
                value = json.loads(line)
                if not isinstance(value, dict):
                    raise ValueError(f"line {number} in {path} is not an object")
                records.append(value)
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"could not read {path}: {error}") from error
    return records


def validate_record(record: dict, expected: dict, subset_sha256: str) -> list[str]:
    """Check temporal alignment, exact source references, actions, and caption."""
    errors = []
    label = record.get("global_clip_id", "unknown clip")
    exact_fields = (
        "wan_overfit_manifest_version",
        "dataset_split",
        "selection_strategy",
        "global_clip_id",
        "episode_name",
        "local_clip_id",
        "source_episode",
        "fps",
    )
    for field in exact_fields:
        if record.get(field) != expected.get(field):
            errors.append(f"{label}: {field} differs from source")
    if record.get("source_subset_sha256") != subset_sha256:
        errors.append(f"{label}: source subset checksum differs")
    if record.get("num_frames") != NUM_FRAMES or record.get("num_actions") != NUM_ACTIONS:
        errors.append(f"{label}: requires exactly 17 observations and 16 actions")

    observations = record.get("observation_indices")
    transitions = record.get("transition_steps")
    if not isinstance(observations, list) or len(observations) != NUM_FRAMES:
        errors.append(f"{label}: requires 17 observation indices")
    elif observations != expected["observation_indices"]:
        errors.append(f"{label}: observation indices must be consecutive source indices")
    if not isinstance(transitions, list) or len(transitions) != NUM_ACTIONS:
        errors.append(f"{label}: requires 16 transition indices")
    elif transitions != expected["transition_steps"] or (
        isinstance(observations, list) and transitions != observations[:-1]
    ):
        errors.append(f"{label}: transitions do not connect consecutive observations")

    if "video_path" in record:
        errors.append(f"{label}: PNG source manifest must not include video_path")
    frames = record.get("rgb_frames")
    if not isinstance(frames, list) or len(frames) != NUM_FRAMES:
        errors.append(f"{label}: requires 17 RGB frame references")
    else:
        if len(set(frames)) != NUM_FRAMES:
            errors.append(f"{label}: duplicated or padded RGB frame reference")
        if frames != expected["rgb_frames"]:
            errors.append(f"{label}: RGB frames differ from real source observations")
        dimensions = set()
        for frame in frames:
            if not isinstance(frame, str) or not Path(frame).is_file():
                errors.append(f"{label}: missing RGB frame: {frame!r}")
                continue
            try:
                with Image.open(frame) as image:
                    if image.format != "PNG" or image.mode != "RGB":
                        errors.append(f"{label}: source frame is not an RGB PNG: {frame}")
                    dimensions.add(image.size)
                    image.verify()
            except (OSError, UnidentifiedImageError) as error:
                errors.append(f"{label}: unreadable RGB PNG {frame}: {error}")
        if len(dimensions) > 1:
            errors.append(f"{label}: RGB frame dimensions differ")

    actions = record.get("raw_actions")
    if not isinstance(actions, list) or len(actions) != NUM_ACTIONS:
        errors.append(f"{label}: requires 16 raw actions")
    elif actions != expected["raw_actions"]:
        errors.append(f"{label}: raw actions differ from source transitions")
    runs = record.get("action_runs")
    if not isinstance(runs, list):
        errors.append(f"{label}: action_runs must be a list")
    else:
        expanded = []
        for run in runs:
            if (
                not isinstance(run, dict)
                or set(run) != {"action", "length"}
                or not isinstance(run["action"], str)
                or type(run["length"]) is not int
                or run["length"] < 1
            ):
                errors.append(f"{label}: malformed action run")
                break
            expanded.extend([run["action"]] * run["length"])
        if len(expanded) != NUM_ACTIONS:
            errors.append(f"{label}: action-run lengths do not sum to 16")
        if expanded != actions:
            errors.append(f"{label}: action runs do not reconstruct raw_actions")
        if isinstance(actions, list):
            try:
                canonical_runs = run_length_encode_actions(actions)
                prompt = generate_action_prompt(canonical_runs)
            except ValueError as error:
                errors.append(f"{label}: {error}")
            else:
                if runs != canonical_runs:
                    errors.append(f"{label}: action runs are not canonical")
                if record.get("prompt") != prompt:
                    errors.append(f"{label}: prompt differs from deterministic 16-action regeneration")
    return errors


def validate_wan_overfit(
    manifest_path: Path = DEFAULT_OUTPUT / "manifest.jsonl",
    *,
    subset_path: Path = DEFAULT_SUBSET,
    dataset_dir: Path | None = None,
    episodes_root: Path | None = None,
) -> list[str]:
    manifest_path = manifest_path.resolve()
    if dataset_dir is None:
        dataset_dir = REPOSITORY_ROOT / "data/datasets/my_way_home"
    if episodes_root is None:
        episodes_root = REPOSITORY_ROOT / "data/episodes/my_way_home"
    context = load_source_context(subset_path, dataset_dir)
    eligible, reasons = evaluate_candidates(context, episodes_root)
    report = json.loads((manifest_path.parent / "report.json").read_text(encoding="utf-8"))
    if not isinstance(report, dict):
        raise ValueError("report.json must contain an object")
    seed = report.get("selection_seed")
    count = report.get("selection_count")
    if type(seed) is not int or type(count) is not int:
        raise ValueError("report selection_seed and selection_count must be integers")
    selected = select_records(eligible, SELECTION_SEED, SELECTION_COUNT)
    records = _read_jsonl(manifest_path)
    errors = []
    if seed != SELECTION_SEED or count != SELECTION_COUNT:
        errors.append("report must use the fixed seed 42 and select exactly 16 clips")
    expected_ids = [record["global_clip_id"] for record in selected]
    actual_ids = [record.get("global_clip_id") for record in records]
    if actual_ids != expected_ids:
        errors.append("manifest IDs differ from deterministic eligible-clip selection")
    if len(records) != SELECTION_COUNT:
        errors.append(f"manifest contains {len(records)} records; expected {SELECTION_COUNT}")
    expected_by_id = {record["global_clip_id"]: record for record in selected}
    for record in records:
        expected = expected_by_id.get(record.get("global_clip_id"))
        if expected is None:
            errors.append(f"unexpected clip ID: {record.get('global_clip_id')!r}")
            continue
        errors.extend(validate_record(record, expected, context["subset_sha256"]))

    expected_report = {
        "source_subset": str(context["subset_path"]),
        "source_subset_sha256": context["subset_sha256"],
        "selection_rule": "lowest SHA-256 of UTF-8 seed:global_clip_id, ties by global_clip_id",
        "selection_seed": SELECTION_SEED,
        "selection_count": SELECTION_COUNT,
        "candidates_examined": len(context["clips"]),
        "eligible": len(eligible),
        "ineligible": sum(reasons.values()),
        "ineligible_reasons": dict(sorted(reasons.items())),
        "selected_global_clip_ids": expected_ids,
        "manifest": str(manifest_path),
        "media_type": "original_rgb_png_references",
        "new_mp4s_created": 0,
    }
    for field, value in expected_report.items():
        if report.get(field) != value:
            errors.append(f"report {field} differs from source validation")
    if any(manifest_path.parent.rglob("*.mp4")):
        errors.append("output directory unexpectedly contains an MP4")
    return errors


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path, nargs="?", default=DEFAULT_OUTPUT / "manifest.jsonl")
    parser.add_argument("--subset", type=Path, default=DEFAULT_SUBSET)
    args = parser.parse_args()
    try:
        errors = validate_wan_overfit(args.manifest, subset_path=args.subset)
    except (OSError, ValueError, json.JSONDecodeError) as error:
        raise SystemExit(f"Wan overfit validation failed: {error}") from error
    if errors:
        raise SystemExit("Wan overfit validation failed:\n" + "\n".join(f"  - {error}" for error in errors))
    print(f"PASS: 17 real RGB observations and 16 aligned actions for every sample in {args.manifest}")


if __name__ == "__main__":
    main()
