"""Integration tests for the multi-episode dataset pipeline."""

from __future__ import annotations

import copy
import hashlib
import json
import shutil
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from data_pipeline.build_dataset_index import (
    build_dataset_index,
    validate_dataset_index,
)
from data_pipeline.process_dataset import DatasetProcessingError, process_dataset
from data_pipeline.split_dataset import split_dataset, validate_dataset_splits
from data_pipeline.summarize_dataset import summarize_dataset


VIDEO_TOOLS_AVAILABLE = bool(shutil.which("ffmpeg") and shutil.which("ffprobe"))


def _write_episode(scenario_root: Path, episode_index: int) -> Path:
    """Create a small, valid episode with enough observations for three clips."""
    episode_dir = scenario_root / f"episode_{episode_index:04d}"
    rgb_dir = episode_dir / "rgb"
    depth_dir = episode_dir / "depth"
    rgb_dir.mkdir(parents=True)
    depth_dir.mkdir()

    observations = []
    for observation_index in range(9):
        rgb = np.zeros((8, 8, 3), dtype=np.uint8)
        rgb[:, :, 0] = episode_index * 40 + observation_index * 8
        rgb[:, :, 1] = np.arange(8, dtype=np.uint8)[None, :] * 16
        rgb[:, :, 2] = np.arange(8, dtype=np.uint8)[:, None] * 16
        depth = np.full(
            (8, 8),
            episode_index * 20 + observation_index,
            dtype=np.uint8,
        )
        filename = f"{observation_index:06d}.png"
        Image.fromarray(rgb).save(rgb_dir / filename)
        Image.fromarray(depth).save(depth_dir / filename)
        observations.append(
            {
                "observation_index": observation_index,
                "tic": observation_index,
                "rgb": f"rgb/{filename}",
                "depth": f"depth/{filename}",
                "game_variables": {
                    "position_x": float(observation_index),
                    "episode": float(episode_index),
                },
            }
        )

    transitions = []
    for step in range(8):
        second_chunk = step >= 4
        transitions.append(
            {
                "step": step,
                "observation_index": step,
                "next_observation_index": step + 1,
                "action": "NOOP" if second_chunk else "MOVE_FORWARD",
                "action_vector": [False] if second_chunk else [True],
                "reward": 0.0,
                "done": False,
                "truncated": step == 7,
                "chunk_id": 1 if second_chunk else 0,
                "chunk_step": step - 4 if second_chunk else step,
                "chunk_length": 4,
            }
        )

    metadata = {
        "episode_index": episode_index,
        "termination_reason": "collector_truncation",
        "initial_state_available": True,
        "num_observations": len(observations),
        "num_transitions": len(transitions),
        "provenance": {
            "seed": 100 + episode_index,
            "scenario_name": scenario_root.name,
            "config_path": "/tmp/synthetic.cfg",
            "wad_path": None,
            "map": "map01",
            "vizdoom_version": "test",
            "max_steps": len(transitions),
            "screen_resolution": [8, 8],
            "available_buttons": ["MOVE_FORWARD"],
            "action_repeat": 1,
            "action_chunk_length_range": [3, 12],
            "action_policy": "uniform_action_chunks",
            "episode_start_time": 0,
            "episode_timeout": 100,
        },
        "action_space": [
            {
                "label": "MOVE_FORWARD",
                "buttons": ["MOVE_FORWARD"],
                "vector": [True],
            },
            {"label": "NOOP", "buttons": [], "vector": [False]},
        ],
        "observations": observations,
        "transitions": transitions,
    }
    (episode_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n",
        encoding="utf-8",
    )
    return episode_dir


def _write_fixture(root: Path, number_of_episodes: int = 2) -> Path:
    scenario_root = root / "episodes" / "synthetic"
    for episode_index in range(number_of_episodes):
        _write_episode(scenario_root, episode_index)
    return scenario_root


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, value: dict) -> None:
    path.write_text(
        json.dumps(value, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _tree_sha256(root: Path) -> str:
    """Hash relative filenames and file bytes, while ignoring timestamps."""
    digest = hashlib.sha256()
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        relative_path = path.relative_to(root).as_posix().encode("utf-8")
        digest.update(len(relative_path).to_bytes(8, "big"))
        digest.update(relative_path)
        digest.update(path.read_bytes())
    return digest.hexdigest()


@unittest.skipUnless(
    VIDEO_TOOLS_AVAILABLE,
    "the batch integration tests require ffmpeg and ffprobe on PATH",
)
class BatchDatasetPipelineTests(unittest.TestCase):
    def test_end_to_end_is_idempotent_and_splits_only_by_episode(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            episodes_root = _write_fixture(root)
            clips_root = root / "clips"
            datasets_root = root / "datasets"

            report_path = process_dataset(
                episodes_root,
                clips_output=clips_root,
                clip_length=4,
                stride=2,
                fps=8,
            )
            report = _read_json(report_path)
            self.assertEqual(report["status"], "complete")
            self.assertEqual(
                [item["episode_name"] for item in report["succeeded"]],
                ["episode_0000", "episode_0001"],
            )
            self.assertEqual(report["failed"], [])

            for episode_index in range(2):
                clip_index_path = (
                    clips_root
                    / "synthetic"
                    / f"episode_{episode_index:04d}"
                    / "clips.json"
                )
                clip_index = _read_json(clip_index_path)
                self.assertEqual(clip_index["num_clips"], 3)
                for clip in clip_index["clips"]:
                    self.assertEqual(clip["num_frames"], 4)
                    self.assertEqual(len(clip["observation_indices"]), 4)
                    self.assertTrue(
                        (clip_index_path.parent / clip["rgb_video"]).is_file()
                    )
                    self.assertEqual(len(clip["depth_frames"]), 4)
                    self.assertEqual(len(clip["motion_pairwise"]), 3)
                    self.assertIn("action_statistics", clip)

            processed_digest = _tree_sha256(clips_root / "synthetic")
            process_dataset(
                episodes_root,
                clips_output=clips_root,
                clip_length=4,
                stride=2,
                fps=8,
            )
            self.assertEqual(
                _tree_sha256(clips_root / "synthetic"),
                processed_digest,
            )

            dataset_index_path = build_dataset_index(
                clips_root / "synthetic",
                output_root=datasets_root,
            )
            dataset_index = _read_json(dataset_index_path)
            self.assertEqual(validate_dataset_index(dataset_index_path), [])
            self.assertEqual(dataset_index["num_episodes"], 2)
            self.assertEqual(dataset_index["num_clips"], 6)
            self.assertEqual(
                [clip["global_clip_id"] for clip in dataset_index["clips"]],
                [
                    f"synthetic_episode_{episode_index:04d}_clip_{clip_id:06d}"
                    for episode_index in range(2)
                    for clip_id in range(3)
                ],
            )
            self.assertTrue(
                all(
                    len(episode["source_episode_metadata_sha256"]) == 64
                    and len(episode["clips_json_sha256"]) == 64
                    for episode in dataset_index["episodes"]
                )
            )
            self.assertEqual(
                dataset_index["process_report"],
                str(report_path.resolve()),
            )
            self.assertEqual(len(dataset_index["process_report_sha256"]), 64)
            self.assertEqual(dataset_index["num_discovered_episodes"], 2)
            self.assertEqual(dataset_index["num_failed_episodes"], 0)

            split_paths = split_dataset(
                dataset_index_path,
                train_ratio=0.5,
                val_ratio=0.0,
                test_ratio=0.5,
                seed=17,
            )
            self.assertEqual(validate_dataset_splits(dataset_index_path), [])
            manifests = {
                name: _read_json(path) for name, path in split_paths.items()
            }
            episode_owners = {
                episode_name: split_name
                for split_name, manifest in manifests.items()
                for episode_name in manifest["episode_names"]
            }
            self.assertEqual(set(episode_owners), {"episode_0000", "episode_0001"})
            self.assertEqual(sum(item["num_episodes"] for item in manifests.values()), 2)

            split_clip_ids = [
                set(manifests[split_name]["global_clip_ids"])
                for split_name in ("train", "val", "test")
            ]
            self.assertTrue(split_clip_ids[0].isdisjoint(split_clip_ids[1]))
            self.assertTrue(split_clip_ids[0].isdisjoint(split_clip_ids[2]))
            self.assertTrue(split_clip_ids[1].isdisjoint(split_clip_ids[2]))
            self.assertEqual(
                set().union(*split_clip_ids),
                {clip["global_clip_id"] for clip in dataset_index["clips"]},
            )
            for episode_name, owner in episode_owners.items():
                episode_clips = [
                    clip
                    for clip in dataset_index["clips"]
                    if clip["episode_name"] == episode_name
                ]
                self.assertEqual(len(episode_clips), 3)
                self.assertLess(
                    episode_clips[1]["start_observation_index"],
                    episode_clips[0]["end_observation_index"],
                )
                self.assertTrue(
                    all(
                        clip["global_clip_id"]
                        in set(manifests[owner]["global_clip_ids"])
                        for clip in episode_clips
                    )
                )

            index_bytes = dataset_index_path.read_bytes()
            manifest_bytes = {
                name: path.read_bytes() for name, path in split_paths.items()
            }
            build_dataset_index(
                clips_root / "synthetic",
                output_root=datasets_root,
            )
            split_dataset(
                dataset_index_path,
                train_ratio=0.5,
                val_ratio=0.0,
                test_ratio=0.5,
                seed=17,
            )
            self.assertEqual(dataset_index_path.read_bytes(), index_bytes)
            self.assertEqual(
                {name: path.read_bytes() for name, path in split_paths.items()},
                manifest_bytes,
            )

            summary = summarize_dataset(dataset_index_path)
            self.assertEqual(summary["number_of_episodes"], 2)
            self.assertEqual(summary["number_of_clips"], 6)
            self.assertEqual(summary["split_counts"]["train"]["episodes"], 1)
            self.assertEqual(summary["split_counts"]["test"]["episodes"], 1)

    def test_fail_fast_and_skip_invalid_are_recorded(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            episodes_root = _write_fixture(root)
            clips_root = root / "clips"
            (episodes_root / "episode_0000" / "rgb" / "000000.png").unlink()

            with self.assertRaises(DatasetProcessingError):
                process_dataset(
                    episodes_root,
                    clips_output=clips_root,
                    clip_length=4,
                    stride=2,
                    fps=8,
                )
            report_path = clips_root / "synthetic" / "process_report.json"
            fail_fast_report = _read_json(report_path)
            self.assertEqual(fail_fast_report["status"], "failed")
            self.assertEqual(fail_fast_report["succeeded"], [])
            self.assertEqual(
                [item["episode_name"] for item in fail_fast_report["failed"]],
                ["episode_0000"],
            )
            self.assertFalse((clips_root / "synthetic" / "episode_0001").exists())

            process_dataset(
                episodes_root,
                clips_output=clips_root,
                clip_length=4,
                stride=2,
                fps=8,
                skip_invalid=True,
            )
            skip_report = _read_json(report_path)
            self.assertEqual(skip_report["status"], "complete")
            self.assertEqual(
                [item["episode_name"] for item in skip_report["failed"]],
                ["episode_0000"],
            )
            self.assertEqual(
                [item["episode_name"] for item in skip_report["succeeded"]],
                ["episode_0001"],
            )
            self.assertTrue(
                (clips_root / "synthetic" / "episode_0001" / "clips.json").is_file()
            )

    def test_stale_episode_outputs_are_removed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            episodes_root = _write_fixture(root)
            clips_root = root / "clips"
            process_dataset(
                episodes_root,
                clips_output=clips_root,
                clip_length=4,
                stride=2,
                fps=8,
            )

            stale_directory = clips_root / "synthetic" / "episode_0099"
            stale_directory.mkdir()
            (stale_directory / "stale.txt").write_text("stale", encoding="utf-8")
            (episodes_root / "episode_0000" / "rgb" / "000000.png").unlink()

            process_dataset(
                episodes_root,
                clips_output=clips_root,
                clip_length=4,
                stride=2,
                fps=8,
                skip_invalid=True,
            )
            self.assertFalse(stale_directory.exists())
            self.assertFalse((clips_root / "synthetic" / "episode_0000").exists())
            self.assertTrue((clips_root / "synthetic" / "episode_0001").is_dir())

            dataset_index_path = build_dataset_index(
                clips_root / "synthetic",
                output_root=root / "datasets",
            )
            dataset_index = _read_json(dataset_index_path)
            self.assertEqual(dataset_index["num_episodes"], 1)
            self.assertEqual(
                [episode["episode_name"] for episode in dataset_index["episodes"]],
                ["episode_0001"],
            )

            _write_episode(episodes_root, 2)
            with self.assertRaisesRegex(
                ValueError,
                "current source episode directories",
            ):
                build_dataset_index(
                    clips_root / "synthetic",
                    output_root=root / "datasets_after_source_change",
                )

    def test_duplicate_and_malformed_references_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            episodes_root = _write_fixture(root)
            clips_root = root / "clips"
            process_dataset(
                episodes_root,
                clips_output=clips_root,
                clip_length=4,
                stride=2,
                fps=8,
            )
            dataset_index_path = build_dataset_index(
                clips_root / "synthetic",
                output_root=root / "datasets",
            )
            split_dataset(
                dataset_index_path,
                train_ratio=0.5,
                val_ratio=0.0,
                test_ratio=0.5,
                seed=17,
            )

            original_index = dataset_index_path.read_bytes()
            corrupted_index = _read_json(dataset_index_path)
            corrupted_index["episodes"].append(
                copy.deepcopy(corrupted_index["episodes"][0])
            )
            corrupted_index["num_episodes"] += 1
            _write_json(dataset_index_path, corrupted_index)
            index_errors = validate_dataset_index(dataset_index_path)
            self.assertTrue(
                any("duplicate episode_name" in error for error in index_errors),
                index_errors,
            )
            dataset_index_path.write_bytes(original_index)

            train_path = dataset_index_path.parent / "train.json"
            corrupted_train = _read_json(train_path)
            corrupted_train["episode_names"].append(
                corrupted_train["episode_names"][0]
            )
            corrupted_train["episode_references"].append(
                copy.deepcopy(corrupted_train["episode_references"][0])
            )
            corrupted_train["num_episodes"] += 1
            _write_json(train_path, corrupted_train)
            split_errors = validate_dataset_splits(dataset_index_path)
            self.assertTrue(
                any("episode_names contains duplicates" in error for error in split_errors),
                split_errors,
            )

            malformed_root = root / "malformed" / "synthetic"
            _write_episode(malformed_root, 0)
            (malformed_root / "episode_not_a_number").mkdir()
            with self.assertRaises(DatasetProcessingError):
                process_dataset(
                    malformed_root,
                    clips_output=root / "malformed_clips",
                    clip_length=4,
                    stride=2,
                    fps=8,
                )


if __name__ == "__main__":
    unittest.main()
