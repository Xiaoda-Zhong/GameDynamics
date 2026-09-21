"""CPU-only smoke tests for the Wan dataset interface."""

from __future__ import annotations

import json
import shutil
import tempfile
import unittest
from pathlib import Path

from training.wan.dataset import ActionCaptionVideoDataset, DatasetValidationError
from training.wan.infer import validate_config


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
EXISTING_MANIFEST = (
    REPOSITORY_ROOT
    / "data"
    / "training"
    / "my_way_home"
    / "budget_0500"
    / "random.jsonl"
)
EXISTING_MANIFESTS = {
    EXISTING_MANIFEST.parent / "random.jsonl": 500,
    EXISTING_MANIFEST.parent / "motion_stratified.jsonl": 500,
    EXISTING_MANIFEST.parent / "action_balanced.jsonl": 500,
    EXISTING_MANIFEST.parent / "joint_motion_action.jsonl": 500,
    EXISTING_MANIFEST.parents[1] / "val.jsonl": 305,
    EXISTING_MANIFEST.parents[1] / "test.jsonl": 282,
}


@unittest.skipUnless(EXISTING_MANIFEST.is_file(), "existing Milestone 4A manifest absent")
class ExistingManifestDatasetTests(unittest.TestCase):
    def test_loads_all_existing_manifests_without_torch_or_cuda(self) -> None:
        for manifest, expected_size in EXISTING_MANIFESTS.items():
            with self.subTest(manifest=manifest.name):
                self.assertTrue(manifest.is_file())
                dataset = ActionCaptionVideoDataset(manifest, validate_videos=False)
                self.assertEqual(len(dataset), expected_size)
                sample = dataset[0]
                for field in (
                    "global_clip_id",
                    "video_path",
                    "prompt",
                    "raw_actions",
                    "action_runs",
                    "source_episode",
                ):
                    self.assertIn(field, sample)
                self.assertTrue(sample["prompt"].strip())
                self.assertEqual(sample["num_frames"], 16)

    @unittest.skipUnless(shutil.which("ffprobe"), "ffprobe is not on PATH")
    def test_existing_video_has_sixteen_frames(self) -> None:
        dataset = ActionCaptionVideoDataset(EXISTING_MANIFEST, validate_videos=False)
        metadata = dataset.get_video_metadata(0)
        self.assertEqual(metadata["frame_count"], 16)
        self.assertGreater(metadata["width"], 0)
        self.assertGreater(metadata["height"], 0)


class DatasetValidationTests(unittest.TestCase):
    def test_empty_prompt_is_rejected(self) -> None:
        if not EXISTING_MANIFEST.is_file():
            self.skipTest("existing Milestone 4A manifest absent")
        record = json.loads(EXISTING_MANIFEST.read_text(encoding="utf-8").splitlines()[0])
        record["prompt"] = "  "
        with tempfile.TemporaryDirectory() as temporary_directory:
            manifest = Path(temporary_directory) / "bad.jsonl"
            manifest.write_text(json.dumps(record) + "\n", encoding="utf-8")
            with self.assertRaisesRegex(DatasetValidationError, "empty prompt"):
                ActionCaptionVideoDataset(manifest, validate_videos=False)

    def test_wan_temporal_constraint_is_checked_without_loading_model(self) -> None:
        config = {
            "model_path_or_id": "local-model",
            "prompt": "The player moves forward for 15 steps.",
            "output_path": "output.mp4",
            "seed": 42,
            "num_frames": 16,
            "width": 448,
            "height": 256,
            "fps": 8,
            "inference_steps": 10,
            "precision": "bf16",
        }
        with self.assertRaisesRegex(ValueError, r"4\*k \+ 1"):
            validate_config(config)


if __name__ == "__main__":
    unittest.main()
