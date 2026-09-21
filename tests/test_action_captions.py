"""Tests for deterministic action-to-text caption manifests."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from data_pipeline.build_action_captions import (
    build_action_caption_manifest,
    generate_action_prompt,
    run_length_encode_actions,
)
from data_pipeline.select_dataset_subsets import select_dataset_subset
from data_pipeline.validate_action_captions import (
    read_jsonl_records,
    validate_action_caption_manifest,
)
from tests.test_global_dataset_subsets import _write_global_fixture


def _write_jsonl(path: Path, records: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as output_file:
        for record in records:
            output_file.write(json.dumps(record, allow_nan=False) + "\n")


class ActionPromptTests(unittest.TestCase):
    def test_single_fifteen_step_run(self) -> None:
        runs = run_length_encode_actions(["MOVE_FORWARD"] * 15)
        self.assertEqual(runs, [{"action": "MOVE_FORWARD", "length": 15}])
        self.assertEqual(
            generate_action_prompt(runs),
            "The player moves forward for 15 steps.",
        )

    def test_multiple_runs_and_singular_step(self) -> None:
        actions = [
            "MOVE_FORWARD",
            "MOVE_FORWARD",
            "MOVE_FORWARD",
            "TURN_LEFT",
            "TURN_LEFT",
            "MOVE_FORWARD",
        ]
        runs = run_length_encode_actions(actions)
        self.assertEqual(
            runs,
            [
                {"action": "MOVE_FORWARD", "length": 3},
                {"action": "TURN_LEFT", "length": 2},
                {"action": "MOVE_FORWARD", "length": 1},
            ],
        )
        self.assertEqual(
            generate_action_prompt(runs),
            "The player moves forward for 3 steps, turns left for 2 steps, "
            "then moves forward for 1 step.",
        )

    def test_noop_and_diagonal_phrases_are_fixed(self) -> None:
        actions = ["NOOP", "MOVE_FORWARD_LEFT", "MOVE_FORWARD_RIGHT"]
        runs = run_length_encode_actions(actions)
        expected = (
            "The player stays still for 1 step, moves forward while turning left "
            "for 1 step, then moves forward while turning right for 1 step."
        )
        self.assertEqual(generate_action_prompt(runs), expected)
        self.assertEqual(
            generate_action_prompt(run_length_encode_actions(actions)),
            expected,
        )


class ActionCaptionManifestTests(unittest.TestCase):
    def test_build_validate_and_deterministically_regenerate_all_splits(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            train_path, _, val_clips, test_clips = _write_global_fixture(root)
            subset_path = root / "subset.json"
            select_dataset_subset(train_path, 6, 11, "random", subset_path)

            first_train = root / "training_first.jsonl"
            second_train = root / "training_second.jsonl"
            build_action_caption_manifest(subset_path, first_train)
            build_action_caption_manifest(subset_path, second_train)
            self.assertEqual(first_train.read_bytes(), second_train.read_bytes())
            self.assertEqual(validate_action_caption_manifest(first_train), [])

            val_output = root / "val.jsonl"
            test_output = root / "test.jsonl"
            build_action_caption_manifest(train_path.parent / "val.json", val_output)
            build_action_caption_manifest(train_path.parent / "test.json", test_output)
            self.assertEqual(validate_action_caption_manifest(val_output), [])
            self.assertEqual(validate_action_caption_manifest(test_output), [])
            self.assertEqual(len(read_jsonl_records(val_output)), len(val_clips))
            self.assertEqual(len(read_jsonl_records(test_output)), len(test_clips))

    def test_corrupt_run_reconstruction_and_leakage_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            train_path, _, val_clips, _ = _write_global_fixture(root)
            subset_path = root / "subset.json"
            select_dataset_subset(train_path, 6, 3, "random", subset_path)
            output_path = root / "train.jsonl"
            build_action_caption_manifest(subset_path, output_path)

            records = read_jsonl_records(output_path)
            records[0]["action_runs"][0]["length"] -= 1
            _write_jsonl(output_path, records)
            errors = validate_action_caption_manifest(output_path)
            self.assertTrue(
                any("do not reconstruct raw_actions" in error for error in errors),
                errors,
            )

            build_action_caption_manifest(subset_path, output_path)
            records = read_jsonl_records(output_path)
            records[0]["global_clip_id"] = val_clips[0]["global_clip_id"]
            _write_jsonl(output_path, records)
            errors = validate_action_caption_manifest(output_path)
            self.assertTrue(
                any(
                    "training captions contain validation or test" in error
                    for error in errors
                ),
                errors,
            )


if __name__ == "__main__":
    unittest.main()
