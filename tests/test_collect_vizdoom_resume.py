"""Tests for safe multi-episode collector numbering."""

from __future__ import annotations

import argparse
import hashlib
import io
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

from data_pipeline import collect_vizdoom


def _directory_digest(path: Path) -> str:
    digest = hashlib.sha256()
    for file_path in sorted(item for item in path.rglob("*") if item.is_file()):
        relative_path = file_path.relative_to(path).as_posix().encode("utf-8")
        contents = file_path.read_bytes()
        digest.update(len(relative_path).to_bytes(8, "big"))
        digest.update(relative_path)
        digest.update(len(contents).to_bytes(8, "big"))
        digest.update(contents)
    return digest.hexdigest()


class _FakeGame:
    def close(self) -> None:
        pass


class CollectorResumeTests(unittest.TestCase):
    def test_collection_continues_after_highest_existing_episode(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            output_root = Path(temporary_directory) / "episodes"
            scenario_root = output_root / "my_way_home"
            existing_directories = [
                scenario_root / "episode_0000",
                scenario_root / "episode_0003",
            ]
            for position, episode_dir in enumerate(existing_directories):
                episode_dir.mkdir(parents=True)
                (episode_dir / "existing.bin").write_bytes(
                    bytes([position, 17, 93, 255])
                )
            original_digests = {
                episode_dir.name: _directory_digest(episode_dir)
                for episode_dir in existing_directories
            }

            arguments = argparse.Namespace(
                scenario="my_way_home",
                episodes=2,
                max_steps=8,
                output=output_root,
                visible=False,
                seed=42,
            )
            collected_indices = []

            def fake_collect_episode(**kwargs: object) -> dict:
                episode_index = int(kwargs["episode_index"])
                collected_indices.append(episode_index)
                episode_dir = scenario_root / f"episode_{episode_index:04d}"
                episode_dir.mkdir()
                (episode_dir / "new.bin").write_bytes(bytes([episode_index]))
                return {"num_observations": 1, "num_transitions": 0}

            with (
                patch.object(collect_vizdoom, "parse_args", return_value=arguments),
                patch.object(collect_vizdoom, "build_game", return_value=_FakeGame()),
                patch.object(
                    collect_vizdoom,
                    "collect_episode",
                    side_effect=fake_collect_episode,
                ),
                redirect_stdout(io.StringIO()),
            ):
                collect_vizdoom.main()

            self.assertEqual(collected_indices, [4, 5])
            self.assertTrue((scenario_root / "episode_0004").is_dir())
            self.assertTrue((scenario_root / "episode_0005").is_dir())
            for episode_dir in existing_directories:
                self.assertTrue(episode_dir.is_dir())
                self.assertEqual(
                    _directory_digest(episode_dir),
                    original_digests[episode_dir.name],
                )


if __name__ == "__main__":
    unittest.main()
