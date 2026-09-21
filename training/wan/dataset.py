#!/usr/bin/env python3
"""CPU-only dataset interface for action-captioned gameplay videos."""

from __future__ import annotations

import argparse
import json
import math
import shutil
import subprocess
from fractions import Fraction
from pathlib import Path
from typing import Any

from PIL import Image, UnidentifiedImageError


class DatasetValidationError(ValueError):
    """Raised when a training manifest record is unusable."""


class ActionCaptionVideoDataset:
    """Read and validate the JSONL interface used by the Wan baseline.

    This class deliberately has no Torch, Diffusers, or CUDA dependency. It
    returns dictionaries so a future training adapter can perform decoding and
    tensor conversion without coupling those operations to manifest parsing.
    """

    def __init__(
        self,
        manifest_path: str | Path,
        *,
        expected_num_frames: int = 16,
        validate_videos: bool = True,
    ) -> None:
        self.manifest_path = Path(manifest_path).expanduser().resolve()
        self.expected_num_frames = expected_num_frames
        self._media_metadata: dict[int, dict[str, int | float]] = {}

        if expected_num_frames <= 0:
            raise ValueError("expected_num_frames must be positive")
        if not self.manifest_path.is_file():
            raise FileNotFoundError(f"dataset manifest does not exist: {self.manifest_path}")

        self.records = _read_jsonl(self.manifest_path)
        if not self.records:
            raise DatasetValidationError(f"dataset manifest is empty: {self.manifest_path}")

        seen_ids: set[str] = set()
        for index, record in enumerate(self.records):
            self._validate_record_fields(record, index, seen_ids)

        if validate_videos:
            self.validate_all_videos()

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, Any]:
        return dict(self.records[index])

    def get_video_metadata(self, index: int) -> dict[str, int | float]:
        """Probe one sample and require its MP4 to contain the expected frames."""
        record = self.records[index]
        if "video_path" not in record:
            raise DatasetValidationError("sample references RGB PNG frames, not a video")
        video_path = Path(record["video_path"])
        metadata = self._media_metadata.get(index)
        if metadata is None:
            metadata = probe_video(video_path)
            self._media_metadata[index] = metadata
        if metadata["frame_count"] != self.expected_num_frames:
            raise DatasetValidationError(
                f"record {index} ({record['global_clip_id']}) video has "
                f"{metadata['frame_count']} frames; expected {self.expected_num_frames}: "
                f"{video_path}"
            )
        return dict(metadata)

    def get_media_metadata(self, index: int) -> dict[str, int | float]:
        """Inspect either an MP4 or a source PNG sequence on CPU."""
        record = self.records[index]
        if "video_path" in record:
            return self.get_video_metadata(index)
        metadata = self._media_metadata.get(index)
        if metadata is None:
            metadata = probe_rgb_frames(record["rgb_frames"], record.get("fps"))
            self._media_metadata[index] = metadata
        return dict(metadata)

    def validate_all_videos(self) -> None:
        """Validate every referenced MP4 or source PNG sequence."""
        if any("video_path" in record for record in self.records) and shutil.which("ffprobe") is None:
            raise DatasetValidationError(
                "ffprobe was not found on PATH; install ffmpeg before validating videos"
            )
        for index in range(len(self.records)):
            self.get_media_metadata(index)

    def _validate_record_fields(
        self,
        record: dict[str, Any],
        index: int,
        seen_ids: set[str],
    ) -> None:
        prefix = f"record {index}"
        clip_id = record.get("global_clip_id")
        if not isinstance(clip_id, str) or not clip_id.strip():
            raise DatasetValidationError(f"{prefix} has no non-empty global_clip_id")
        if clip_id in seen_ids:
            raise DatasetValidationError(f"{prefix} duplicates global_clip_id {clip_id!r}")
        seen_ids.add(clip_id)

        prompt = record.get("prompt")
        if not isinstance(prompt, str) or not prompt.strip():
            raise DatasetValidationError(f"{prefix} ({clip_id}) has an empty prompt")

        has_video = "video_path" in record
        has_frames = "rgb_frames" in record
        if has_video == has_frames:
            raise DatasetValidationError(
                f"{prefix} ({clip_id}) must have exactly one of video_path or rgb_frames"
            )
        if has_video:
            video_value = record["video_path"]
            if not isinstance(video_value, str) or not video_value.strip():
                raise DatasetValidationError(f"{prefix} ({clip_id}) has no video_path")
            video_path = resolve_media_path(video_value, self.manifest_path.parent)
            record["video_path"] = str(video_path)
            if not video_path.is_file():
                raise DatasetValidationError(
                    f"{prefix} ({clip_id}) references a missing video: {video_path}"
                )
        else:
            frames = record["rgb_frames"]
            if not isinstance(frames, list) or len(frames) != self.expected_num_frames:
                raise DatasetValidationError(
                    f"{prefix} ({clip_id}) requires {self.expected_num_frames} RGB frames"
                )
            resolved = []
            for frame in frames:
                if not isinstance(frame, str) or not frame.strip():
                    raise DatasetValidationError(f"{prefix} ({clip_id}) has an invalid RGB frame")
                path = resolve_media_path(frame, self.manifest_path.parent)
                if not path.is_file():
                    raise DatasetValidationError(f"{prefix} ({clip_id}) is missing RGB frame: {path}")
                resolved.append(str(path))
            if len(set(resolved)) != len(resolved):
                raise DatasetValidationError(f"{prefix} ({clip_id}) repeats an RGB frame")
            record["rgb_frames"] = resolved

        if record.get("num_frames") != self.expected_num_frames:
            raise DatasetValidationError(
                f"{prefix} ({clip_id}) declares {record.get('num_frames')!r} frames; "
                f"expected {self.expected_num_frames}"
            )
        if not isinstance(record.get("source_episode"), str) or not record["source_episode"]:
            raise DatasetValidationError(f"{prefix} ({clip_id}) has no source_episode")
        if not isinstance(record.get("raw_actions"), list):
            raise DatasetValidationError(f"{prefix} ({clip_id}) raw_actions must be a list")
        if not isinstance(record.get("action_runs"), list):
            raise DatasetValidationError(f"{prefix} ({clip_id}) action_runs must be a list")


def resolve_media_path(value: str, manifest_directory: Path) -> Path:
    """Locate archived absolute media paths after moving the project workspace."""
    path = Path(value).expanduser()
    if not path.is_absolute():
        return (manifest_directory / path).resolve()
    if path.is_file():
        return path.resolve()
    parts = path.parts
    for index in range(len(parts) - 1, -1, -1):
        if parts[index] == "data":
            local = Path(__file__).resolve().parents[2].joinpath(*parts[index:]).resolve()
            if local.is_file():
                return local
    return path


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as manifest_file:
        for line_number, line in enumerate(manifest_file, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                raise DatasetValidationError(
                    f"invalid JSON on line {line_number} of {path}: {error}"
                ) from error
            if not isinstance(record, dict):
                raise DatasetValidationError(
                    f"line {line_number} of {path} is not a JSON object"
                )
            records.append(record)
    return records


def probe_video(video_path: str | Path) -> dict[str, int | float]:
    """Return frame count, dimensions, FPS, and duration from ffprobe."""
    ffprobe = shutil.which("ffprobe")
    if ffprobe is None:
        raise DatasetValidationError(
            "ffprobe was not found on PATH; install ffmpeg before inspecting videos"
        )
    path = Path(video_path)
    command = [
        ffprobe,
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-count_frames",
        "-show_entries",
        "stream=width,height,avg_frame_rate,nb_read_frames,duration",
        "-of",
        "json",
        str(path),
    ]
    result = subprocess.run(command, capture_output=True, text=True, check=False)
    if result.returncode != 0:
        details = result.stderr.strip() or "ffprobe returned no error details"
        raise DatasetValidationError(f"ffprobe failed for {path}: {details}")
    try:
        streams = json.loads(result.stdout)["streams"]
        if len(streams) != 1:
            raise ValueError(f"expected one video stream, found {len(streams)}")
        stream = streams[0]
        metadata: dict[str, int | float] = {
            "frame_count": int(stream["nb_read_frames"]),
            "width": int(stream["width"]),
            "height": int(stream["height"]),
            "fps": float(Fraction(stream["avg_frame_rate"])),
            "duration": float(stream["duration"]),
        }
    except (json.JSONDecodeError, KeyError, TypeError, ValueError, ZeroDivisionError) as error:
        raise DatasetValidationError(
            f"ffprobe returned incomplete metadata for {path}: {error}"
        ) from error
    if (
        metadata["frame_count"] < 0
        or metadata["width"] <= 0
        or metadata["height"] <= 0
        or not math.isfinite(metadata["fps"])
        or metadata["fps"] <= 0
    ):
        raise DatasetValidationError(f"ffprobe returned invalid metadata for {path}")
    return metadata


def probe_rgb_frames(frame_paths: list[str], fps: int | float | None) -> dict[str, int | float]:
    """Verify that every referenced source frame is a same-size RGB PNG."""
    if isinstance(fps, bool) or not isinstance(fps, (int, float)) or not math.isfinite(fps) or fps <= 0:
        raise DatasetValidationError("PNG sequence requires a positive finite fps")
    dimensions = set()
    for frame in frame_paths:
        try:
            with Image.open(frame) as image:
                if image.format != "PNG" or image.mode != "RGB":
                    raise DatasetValidationError(f"source frame is not an RGB PNG: {frame}")
                dimensions.add(image.size)
                image.verify()
        except (OSError, UnidentifiedImageError) as error:
            raise DatasetValidationError(f"unreadable RGB PNG {frame}: {error}") from error
    if len(dimensions) != 1:
        raise DatasetValidationError("PNG sequence has inconsistent frame dimensions")
    width, height = dimensions.pop()
    return {
        "frame_count": len(frame_paths),
        "width": width,
        "height": height,
        "fps": float(fps),
        "duration": len(frame_paths) / fps,
    }


def inspect_dataset(
    manifest_path: str | Path,
    *,
    index: int = 0,
    expected_num_frames: int = 16,
    validate_all_videos: bool = True,
) -> dict[str, Any]:
    dataset = ActionCaptionVideoDataset(
        manifest_path,
        expected_num_frames=expected_num_frames,
        validate_videos=validate_all_videos,
    )
    try:
        sample = dataset[index]
    except IndexError as error:
        raise DatasetValidationError(
            f"sample index {index} is outside dataset of size {len(dataset)}"
        ) from error
    media = dataset.get_media_metadata(index)
    return {
        "dataset_size": len(dataset),
        "example_id": sample["global_clip_id"],
        "prompt": sample["prompt"],
        "media_path": sample.get("video_path", sample.get("rgb_frames", [None])[0]),
        "frame_count": media["frame_count"],
        "resolution": f"{media['width']}x{media['height']}",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path, help="action-caption JSONL manifest")
    parser.add_argument("--index", type=int, default=0, help="example index to print")
    parser.add_argument("--expected-frames", type=int, default=16)
    parser.add_argument(
        "--skip-full-video-validation",
        action="store_true",
        help="probe only the displayed sample instead of every video",
    )
    args = parser.parse_args()
    try:
        summary = inspect_dataset(
            args.manifest,
            index=args.index,
            expected_num_frames=args.expected_frames,
            validate_all_videos=not args.skip_full_video_validation,
        )
    except (DatasetValidationError, FileNotFoundError, ValueError) as error:
        raise SystemExit(f"Dataset inspection failed: {error}") from error

    print(f"Dataset size: {summary['dataset_size']}")
    print(f"Example ID: {summary['example_id']}")
    print(f"Prompt: {summary['prompt']}")
    print(f"Media path: {summary['media_path']}")
    print(f"Frame count: {summary['frame_count']}")
    print(f"Resolution: {summary['resolution']}")


if __name__ == "__main__":
    main()
