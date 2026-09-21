"""CPU-only tests for real 17-frame Wan overfit preparation."""

from __future__ import annotations

import copy
import json
import tempfile
import unittest
from pathlib import Path

from PIL import Image

from data_pipeline.build_action_captions import generate_action_prompt, run_length_encode_actions
from data_pipeline.prepare_wan_overfit import (
    DEFAULT_OUTPUT,
    IneligibleCandidate,
    extend_candidate,
    select_records,
)
from data_pipeline.validate_wan_overfit import validate_record, validate_wan_overfit
from training.wan.dataset import ActionCaptionVideoDataset


class WanOverfitUnitTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.episode_dir = Path(self.temporary.name) / "data/episodes/my_way_home/episode_0001"
        rgb_dir = self.episode_dir / "rgb"
        rgb_dir.mkdir(parents=True)
        observations = []
        for index in range(17):
            name = f"{index:06d}.png"
            Image.new("RGB", (8, 8), (index, 0, 0)).save(rgb_dir / name)
            observations.append({"observation_index": index, "rgb": f"rgb/{name}"})
        actions = ["MOVE_FORWARD"] * 8 + ["TURN_LEFT"] * 8
        transitions = [
            {
                "step": index,
                "observation_index": index,
                "next_observation_index": index + 1,
                "action": action,
                "done": False,
                "truncated": False,
            }
            for index, action in enumerate(actions)
        ]
        self.metadata = {
            "provenance": {"scenario_name": "my_way_home"},
            "observations": observations,
            "transitions": transitions,
        }
        self.local_clip = {
            "clip_id": 0,
            "start_observation_index": 0,
            "observation_indices": list(range(16)),
            "transition_steps": list(range(15)),
            "actions": actions[:15],
        }
        self.clip = {
            "global_clip_id": "my_way_home_episode_0001_clip_000000",
            "scenario_name": "my_way_home",
            "episode_name": "episode_0001",
            "source_episode": str(self.episode_dir),
            "local_clip_id": 0,
            "start_observation_index": 0,
            "observation_indices": list(range(16)),
            "transition_steps": list(range(15)),
            "actions": actions[:15],
            "num_frames": 16,
            "num_actions": 15,
            "fps": 8,
        }

    def _extend(self, metadata: dict | None = None) -> dict:
        return extend_candidate(
            self.clip,
            self.episode_dir,
            metadata if metadata is not None else self.metadata,
            self.local_clip,
        )

    def test_normal_valid_extension_uses_real_source_frames(self) -> None:
        record = self._extend()
        self.assertEqual(record["observation_indices"], list(range(17)))
        self.assertEqual(record["transition_steps"], list(range(16)))
        self.assertEqual(record["raw_actions"], ["MOVE_FORWARD"] * 8 + ["TURN_LEFT"] * 8)
        self.assertEqual(record["num_frames"], 17)
        self.assertEqual(record["num_actions"], 16)
        self.assertEqual(len(set(record["rgb_frames"])), 17)
        self.assertEqual(validate_record({**record, "source_subset_sha256": "hash"}, record, "hash"), [])

    def test_near_episode_end_without_transition_is_ineligible(self) -> None:
        metadata = copy.deepcopy(self.metadata)
        metadata["transitions"].pop()
        with self.assertRaises(IneligibleCandidate) as caught:
            self._extend(metadata)
        self.assertEqual(caught.exception.reason, "missing_transition")

    def test_missing_seventeenth_observation_is_ineligible(self) -> None:
        metadata = copy.deepcopy(self.metadata)
        metadata["observations"].pop()
        with self.assertRaises(IneligibleCandidate) as caught:
            self._extend(metadata)
        self.assertEqual(caught.exception.reason, "missing_observation")

    def test_incorrect_transition_alignment_is_ineligible(self) -> None:
        metadata = copy.deepcopy(self.metadata)
        metadata["transitions"][15]["next_observation_index"] = 18
        with self.assertRaises(IneligibleCandidate) as caught:
            self._extend(metadata)
        self.assertEqual(caught.exception.reason, "transition_alignment")

    def test_terminal_extension_is_ineligible(self) -> None:
        metadata = copy.deepcopy(self.metadata)
        metadata["transitions"][15]["done"] = True
        with self.assertRaises(IneligibleCandidate) as caught:
            self._extend(metadata)
        self.assertEqual(caught.exception.reason, "terminal_transition")

    def test_duplicated_padding_frame_is_rejected(self) -> None:
        expected = self._extend()
        padded = copy.deepcopy(expected)
        padded["source_subset_sha256"] = "hash"
        padded["rgb_frames"][-1] = padded["rgb_frames"][-2]
        errors = validate_record(padded, expected, "hash")
        self.assertTrue(any("duplicated or padded" in error for error in errors))
        metadata = copy.deepcopy(self.metadata)
        metadata["observations"][-1]["rgb"] = metadata["observations"][-2]["rgb"]
        with self.assertRaises(IneligibleCandidate) as caught:
            self._extend(metadata)
        self.assertEqual(caught.exception.reason, "duplicated_frame_reference")

    def test_deterministic_sixteen_clip_selection(self) -> None:
        records = [{"global_clip_id": f"clip_{index:03d}"} for index in range(20)]
        first = select_records(records, seed=42, count=16)
        again = select_records(list(reversed(records)), seed=42, count=16)
        self.assertEqual(first, again)
        self.assertEqual(len(first), 16)
        self.assertEqual(len({item["global_clip_id"] for item in first}), 16)

    def test_prompt_is_regenerated_from_all_sixteen_actions(self) -> None:
        expected = self._extend()
        self.assertEqual(
            expected["prompt"],
            generate_action_prompt(run_length_encode_actions(expected["raw_actions"])),
        )
        old_prompt = generate_action_prompt(run_length_encode_actions(expected["raw_actions"][:-1]))
        self.assertNotEqual(old_prompt, expected["prompt"])
        record = {**expected, "source_subset_sha256": "hash", "prompt": old_prompt}
        self.assertTrue(any("prompt differs" in error for error in validate_record(record, expected, "hash")))

    def test_cpu_dataset_reader_accepts_png_sequence(self) -> None:
        record = self._extend()
        manifest = Path(self.temporary.name) / "manifest.jsonl"
        manifest.write_text(json.dumps(record) + "\n", encoding="utf-8")
        dataset = ActionCaptionVideoDataset(manifest, expected_num_frames=17)
        self.assertEqual(dataset.get_media_metadata(0)["frame_count"], 17)
        self.assertEqual(dataset.get_media_metadata(0)["width"], 8)


@unittest.skipUnless((DEFAULT_OUTPUT / "manifest.jsonl").is_file(), "Wan overfit manifest absent")
class WanOverfitIntegrationTests(unittest.TestCase):
    def test_generated_manifest_validates_against_formal_splits_and_sources(self) -> None:
        self.assertEqual(validate_wan_overfit(), [])


if __name__ == "__main__":
    unittest.main()
