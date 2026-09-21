"""Tests for fixed-budget selection from a global training split."""

from __future__ import annotations

import copy
import hashlib
import json
import math
import tempfile
import unittest
from pathlib import Path

from data_pipeline.compare_dataset_subsets import compare_dataset_subsets
from data_pipeline.select_dataset_subsets import STRATEGIES, select_dataset_subset
from data_pipeline.validate_dataset_subset import validate_dataset_subset


ACTIONS = (
    "MOVE_FORWARD",
    "TURN_LEFT",
    "TURN_RIGHT",
    "NOOP",
    "MOVE_FORWARD_LEFT",
    "MOVE_FORWARD_RIGHT",
)


def _write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _clip(
    root: Path,
    episode_index: int,
    local_clip_id: int,
    action_index: int,
) -> dict:
    action = ACTIONS[action_index % len(ACTIONS)]
    actions = [action] * 15
    counts = {label: actions.count(label) for label in ACTIONS}
    fractions = {label: counts[label] / len(actions) for label in ACTIONS}
    source_episode = str(
        (root / "episodes" / "synthetic" / f"episode_{episode_index:04d}").resolve()
    )
    global_id = (
        f"synthetic_episode_{episode_index:04d}_clip_{local_clip_id:06d}"
    )
    motion_mean = (episode_index * 3 + local_clip_id) / 100.0
    return {
        "global_clip_id": global_id,
        "scenario_name": "synthetic",
        "episode_name": f"episode_{episode_index:04d}",
        "source_episode": source_episode,
        "local_clip_id": local_clip_id,
        "rgb_video": str((root / "videos" / f"{global_id}.mp4").resolve()),
        "depth_frames": [
            str((root / "depth" / global_id / "000.png").resolve())
        ],
        "actions": actions,
        "num_frames": 16,
        "num_actions": 15,
        "observation_indices": list(range(16)),
        "motion_statistics": {
            "mean": motion_mean,
            "std": 0.0,
            "min": motion_mean,
            "max": motion_mean,
            "pairwise": [motion_mean] * 15,
        },
        "action_statistics": {
            "action_counts": counts,
            "action_fractions": fractions,
            "dominant_action": action,
            "dominant_action_fraction": 1.0,
            "noop_fraction": 1.0 if action == "NOOP" else 0.0,
            "unique_action_count": 1,
            "action_entropy": -sum(
                fraction * math.log(fraction)
                for fraction in fractions.values()
                if fraction
            ),
        },
    }


def _write_global_fixture(
    root: Path,
) -> tuple[Path, list[dict], list[dict], list[dict]]:
    dataset_dir = root / "datasets" / "synthetic"
    train_clips = [
        _clip(root, index // 3, index % 3, index % len(ACTIONS))
        for index in range(18)
    ]
    val_clips = [_clip(root, 100, index, index) for index in range(2)]
    test_clips = [_clip(root, 101, index, index + 2) for index in range(2)]
    for clip in [*train_clips, *val_clips, *test_clips]:
        video_path = Path(clip["rgb_video"])
        video_path.parent.mkdir(parents=True, exist_ok=True)
        video_path.write_bytes(b"synthetic mp4 fixture")
    dataset_path = dataset_dir / "all_clips.json"
    _write_json(
        dataset_path,
        {
            "scenario_name": "synthetic",
            "declared_action_space": list(ACTIONS),
            "clips": [*train_clips, *val_clips, *test_clips],
        },
    )
    dataset_sha256 = _sha256(dataset_path)

    for split_name, clips in (
        ("train", train_clips),
        ("val", val_clips),
        ("test", test_clips),
    ):
        _write_json(
            dataset_dir / f"{split_name}.json",
            {
                "split_manifest_version": 1,
                "split_name": split_name,
                "scenario_name": "synthetic",
                "source_dataset_index": str(dataset_path.resolve()),
                "source_dataset_index_sha256": dataset_sha256,
                "num_clips": len(clips),
                "global_clip_ids": [clip["global_clip_id"] for clip in clips],
                "clips": clips,
            },
        )
    return dataset_dir / "train.json", train_clips, val_clips, test_clips


class GlobalDatasetSubsetTests(unittest.TestCase):
    def test_all_strategies_select_unique_training_clips(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            train_path, train_clips, val_clips, test_clips = _write_global_fixture(root)
            train_ids = {clip["global_clip_id"] for clip in train_clips}
            holdout_ids = {
                clip["global_clip_id"] for clip in [*val_clips, *test_clips]
            }
            manifests = {}
            manifest_paths = []
            for strategy in STRATEGIES:
                path = root / "subsets" / f"{strategy}.json"
                manifest_paths.append(path)
                select_dataset_subset(train_path, 9, 7, strategy, path)
                self.assertEqual(validate_dataset_subset(path), [])
                manifest = json.loads(path.read_text(encoding="utf-8"))
                manifests[strategy] = manifest
                selected_ids = manifest["selected_global_clip_ids"]
                self.assertEqual(len(selected_ids), 9)
                self.assertEqual(len(set(selected_ids)), 9)
                self.assertTrue(set(selected_ids) <= train_ids)
                self.assertTrue(set(selected_ids).isdisjoint(holdout_ids))
                if strategy in {"motion_stratified", "joint_motion_action"}:
                    self.assertEqual(
                        manifest["motion_bin_counts"],
                        {"low": 3, "medium": 3, "high": 3},
                    )
                self.assertEqual(
                    manifest["selected_clips"],
                    [
                        next(
                            clip
                            for clip in train_clips
                            if clip["global_clip_id"] == clip_id
                        )
                        for clip_id in selected_ids
                    ],
                )
                concentration = manifest["source_episode_concentration"]
                self.assertEqual(
                    sum(concentration["clips_per_source_episode"].values()),
                    9,
                )

            random_l1 = manifests["random"]["summary_action_balance_statistics"][
                "action_imbalance_score"
            ]
            balanced_l1 = manifests["action_balanced"][
                "summary_action_balance_statistics"
            ]["action_imbalance_score"]
            self.assertLessEqual(balanced_l1, random_l1)
            comparison = compare_dataset_subsets(manifest_paths)
            self.assertIn("Selected clips per source episode", comparison)
            self.assertIn("Top-5 fraction", comparison)

    def test_random_reproduction_and_corruption_detection(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            train_path, _, _, _ = _write_global_fixture(root)
            first = root / "first.json"
            second = root / "second.json"
            select_dataset_subset(train_path, 9, 19, "random", first)
            select_dataset_subset(train_path, 9, 19, "random", second)
            self.assertEqual(first.read_bytes(), second.read_bytes())

            corrupted = json.loads(first.read_text(encoding="utf-8"))
            corrupted["summary_action_balance_statistics"][
                "action_imbalance_score"
            ] += 0.1
            _write_json(first, corrupted)
            errors = validate_dataset_subset(first)
            self.assertTrue(
                any("summary_action_balance_statistics" in error for error in errors),
                errors,
            )

    def test_validation_rejects_holdout_overlap_and_non_train_source(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            train_path, _, _, _ = _write_global_fixture(root)
            manifest_path = root / "random.json"
            select_dataset_subset(train_path, 6, 5, "random", manifest_path)
            selected_id = json.loads(manifest_path.read_text(encoding="utf-8"))[
                "selected_global_clip_ids"
            ][0]

            val_path = train_path.parent / "val.json"
            val = json.loads(val_path.read_text(encoding="utf-8"))
            val["global_clip_ids"].append(selected_id)
            _write_json(val_path, val)
            errors = validate_dataset_subset(manifest_path)
            self.assertTrue(
                any("also occur in val.json" in error for error in errors),
                errors,
            )

            with self.assertRaisesRegex(ValueError, "requires split_name='train'"):
                select_dataset_subset(
                    val_path,
                    1,
                    5,
                    "random",
                    root / "from_val.json",
                )


if __name__ == "__main__":
    unittest.main()
