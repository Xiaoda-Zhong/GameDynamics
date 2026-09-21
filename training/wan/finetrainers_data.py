"""Lossless, one-to-one video input for the official Finetrainers Wan trainer.

The source manifest and PNGs are read only. Derived MP4s and trainer file lists
live under the experiment output directory, never under the formal dataset.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

import imageio_ffmpeg
import numpy as np
from decord import VideoReader, cpu
from PIL import Image

from training.wan.dataset import ActionCaptionVideoDataset


FRAMES = 17
HEIGHT = 256
WIDTH = 448
SAMPLES = 16


def _encode_lossless_rgb(frames: list[str], fps: int, destination: Path) -> None:
    """Encode exactly the referenced PNG observations, without frame synthesis."""
    paths = [Path(frame) for frame in frames]
    start = int(paths[0].stem)
    if any(path.parent != paths[0].parent or path.name != f"{start + i:06d}.png"
           for i, path in enumerate(paths)):
        raise ValueError("RGB observations must be contiguous numbered PNGs in one directory")
    temporary = destination.with_name(destination.stem + ".partial.mp4")
    command = [
        imageio_ffmpeg.get_ffmpeg_exe(), "-hide_banner", "-loglevel", "error", "-nostdin", "-y",
        "-framerate", str(fps), "-start_number", str(start),
        "-i", str(paths[0].parent / "%06d.png"), "-frames:v", str(FRAMES),
        "-c:v", "libx264rgb", "-crf", "0", "-preset", "ultrafast",
        "-pix_fmt", "rgb24", "-movflags", "+faststart", str(temporary),
    ]
    try:
        subprocess.run(command, check=True, capture_output=True, text=True)
        _verify_exact_frames(temporary, frames)
        temporary.replace(destination)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def _verify_exact_frames(video_path: Path, frames: list[str]) -> None:
    video = VideoReader(str(video_path), ctx=cpu(0))
    if len(video) != FRAMES:
        raise ValueError(f"{video_path} decodes to {len(video)} frames, expected {FRAMES}")
    for index, frame_path in enumerate(frames):
        with Image.open(frame_path) as image:
            expected = np.asarray(image.convert("RGB"))
        frame = video[index]
        decoded = frame.asnumpy() if hasattr(frame, "asnumpy") else frame.numpy()
        if not np.array_equal(decoded, expected):
            raise ValueError(f"{video_path} decoded frame {index} differs from {frame_path}")


def prepare_finetrainers_data(
    manifest_path: Path, output_dir: Path, *, sample_id: str | None = None,
    sample_ids: tuple[str, ...] | None = None,
    prompt_override: str | None = None,
    prompt_prefix: str | None = None,
) -> Path:
    """Make Finetrainers' video/prompt list for all, one, or ordered selected clips.

    A prompt override applies to selected clips; a prefix applies to all clips.
    The source manifest and action labels remain unchanged.
    """
    if prompt_prefix is not None:
        if prompt_override is not None or not prompt_prefix.strip() or "\n" in prompt_prefix:
            raise ValueError("Prompt prefix must be one nonempty line and cannot accompany an override")
    if sample_id is not None and sample_ids is not None:
        raise ValueError("Select either one sample ID or a tuple of sample IDs")
    if sample_ids is not None and (not sample_ids or len(set(sample_ids)) != len(sample_ids)):
        raise ValueError("Sample IDs must be a nonempty tuple of distinct IDs")
    if prompt_override is not None:
        if (sample_id is None and sample_ids is None) or not prompt_override.strip() or "\n" in prompt_override:
            raise ValueError("Prompt override requires selected samples and one nonempty line")
    manifest_path = manifest_path.resolve()
    dataset = ActionCaptionVideoDataset(manifest_path, expected_num_frames=FRAMES)
    if len(dataset) != SAMPLES:
        raise ValueError(f"Expected {SAMPLES} samples, found {len(dataset)}")
    if sample_ids is not None:
        by_id = {record["global_clip_id"]: index for index, record in enumerate(dataset.records)}
        missing = [item for item in sample_ids if item not in by_id]
        if missing:
            raise ValueError(f"Selected sample IDs are absent from the manifest: {missing}")
        selected_indices = [by_id[item] for item in sample_ids]
    else:
        selected_indices = [
            index for index, record in enumerate(dataset.records)
            if sample_id is None or record["global_clip_id"] == sample_id
        ]
    if sample_id is not None and len(selected_indices) != 1:
        raise ValueError(f"Expected exactly one sample with ID {sample_id!r}")

    data_root = output_dir / "derived_dataset"
    videos_dir = data_root / "videos"
    videos_dir.mkdir(parents=True, exist_ok=True)
    prompts: list[str] = []
    video_names: list[str] = []
    mapping: list[dict] = []
    for index, source_index in enumerate(selected_indices):
        record = dataset.records[source_index]
        if (
            record.get("num_actions") != FRAMES - 1
            or len(record["raw_actions"]) != FRAMES - 1
            or not all(isinstance(action, str) for action in record["raw_actions"])
        ):
            raise ValueError(f"Sample {index} does not have 16 string actions")
        if "rgb_frames" not in record:
            raise ValueError(f"Sample {index} must reference the real RGB observations")
        media = dataset.get_media_metadata(source_index)
        fps = record.get("fps")
        if type(fps) is not int or fps <= 0 or media["fps"] != float(fps):
            raise ValueError(f"Sample {index} has an invalid FPS")
        video_name = f"videos/{index:04d}.mp4"
        video_path = data_root / video_name
        if video_path.exists():
            _verify_exact_frames(video_path, record["rgb_frames"])
        else:
            _encode_lossless_rgb(record["rgb_frames"], fps, video_path)
        prompt = prompt_override if prompt_override is not None else record["prompt"]
        if prompt_prefix is not None:
            prompt = f"{prompt_prefix} {prompt}"
        prompts.append(prompt)
        video_names.append(video_name)
        source = {
            "global_clip_id": record["global_clip_id"],
            "video_path": str(video_path),
            "source_rgb_frames": record["rgb_frames"],
            "raw_actions": record["raw_actions"],
            "prompt": prompt,
        }
        if prompt_override is not None or prompt_prefix is not None:
            source["source_prompt"] = record["prompt"]
        mapping.append(source)

    (data_root / "prompt.txt").write_text("\n".join(prompts) + "\n", encoding="utf-8")
    (data_root / "videos.txt").write_text("\n".join(video_names) + "\n", encoding="utf-8")
    (data_root / "source_map.jsonl").write_text(
        "".join(json.dumps(item, ensure_ascii=False) + "\n" for item in mapping), encoding="utf-8"
    )
    config_path = output_dir / "finetrainers_dataset.json"
    config_path.write_text(json.dumps({"datasets": [{
        "data_root": str(data_root.resolve()),
        "dataset_type": "video",
        "video_resolution_buckets": [[FRAMES, HEIGHT, WIDTH]],
        "reshape_mode": "bicubic",
    }]}, indent=2) + "\n", encoding="utf-8")
    (output_dir / "source_manifest.sha256").write_text(
        hashlib.sha256(manifest_path.read_bytes()).hexdigest() + "  " + str(manifest_path) + "\n",
        encoding="utf-8",
    )
    return config_path
