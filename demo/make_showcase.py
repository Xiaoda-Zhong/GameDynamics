"""Compose one saved M6F-2 held-out trio for presentation; never run the model."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import imageio.v2 as imageio
from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]
EVAL = ROOT / "data/experiments/m6f2_action_contrastive_128_6400/eval"
OUT = ROOT / "demo"
STATE, SEED = "D", 43
CONDITIONS = ("original", "reversed", "all_noop")
LABELS = ("ORIGINAL", "REVERSED", "ALL NOOP")
TIMELINES = (
    "A0–1: forward + LEFT   |   A2–9: NOOP   |   A10–15: forward + RIGHT",
    "A0–1: forward + RIGHT  |   A2–9: NOOP   |   A10–15: forward + LEFT",
    "A0–15: NOOP",
)


def font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    path = Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf")
    return ImageFont.truetype(str(path), size) if path.exists() else ImageFont.load_default()


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    metrics = json.loads((EVAL / "metrics.json").read_text())
    pair = next(p for p in metrics["matched_pairs"] if (p["state"], p["seed"]) == (STATE, SEED))
    assert pair["first_segment_expected_flip"] and pair["final_segment_expected_flip"]
    entries = [next(v for v in metrics["videos"] if (v["state"], v["seed"], v["condition"])
                    == (STATE, SEED, condition)) for condition in CONDITIONS]
    assert len({(e["sample_id"], e["episode_id"], e["split"]) for e in entries}) == 1
    assert entries[0]["split"] in {"val", "test"}

    paths = [Path(entry["video_path"]) for entry in entries]
    videos = []
    fps_values = []
    for entry, path in zip(entries, paths):
        assert sha256(path) == entry["video_sha256"], path
        reader = imageio.get_reader(str(path), format="ffmpeg")
        try:
            fps_values.append(reader.get_meta_data()["fps"])
            videos.append([Image.fromarray(frame).convert("RGB") for frame in reader])
        finally:
            reader.close()
    assert len(set(fps_values)) == 1 and all(len(video) == 17 for video in videos)
    width, height = videos[0][0].size
    assert (width, height) == (448, 256)
    assert all(frame.size == (width, height) for video in videos for frame in video)

    heading, body, small = font(27), font(18), font(15)
    canvas_width, canvas_height = width * 3, 450
    video_y = 112
    frames = []
    for index in range(17):
        canvas = Image.new("RGB", (canvas_width, canvas_height), "#111722")
        draw = ImageDraw.Draw(canvas)
        draw.text((22, 12), "GameDynamics  |  held-out state D", font=heading, fill="white")
        draw.text((22, 52), "Same conditioned O0 / same seed 43 and noise / only action IDs changed",
                  font=body, fill="#d0d9eb")
        for column, video in enumerate(videos):
            x = column * width
            draw.rectangle((x, 88, x + width - 1, 111), fill="#243550")
            draw.text((x + 13, 90), LABELS[column], font=small, fill="white")
            # Paste each decoded source frame unchanged at its native 448x256 size.
            canvas.paste(video[index], (x, video_y))
            draw.text((x + 11, 381), TIMELINES[column], font=font(12), fill="#dbe5f2")
        draw.text((22, 420), f"Frame O{index:02d} / O16  •  original held-out generation, 17 frames at 8 fps",
                  font=small, fill="#a9b7cc")
        frames.append(canvas)

    OUT.mkdir(parents=True, exist_ok=True)
    mp4 = OUT / "heldout_state_D_seed_43.mp4"
    writer = imageio.get_writer(str(mp4), format="ffmpeg", fps=fps_values[0], codec="libx264",
                                quality=9, pixelformat="yuv420p", macro_block_size=2)
    try:
        for frame in frames:
            writer.append_data(__import__("numpy").asarray(frame))
    finally:
        writer.close()

    gif = OUT / "heldout_state_D_seed_43.gif"
    preview = [frame.resize((1008, 338), Image.Resampling.LANCZOS) for frame in frames]
    preview[0].save(gif, save_all=True, append_images=preview[1:], duration=125,
                    loop=0, optimize=True)

    manifest = {
        "source_milestone": "M6F-2",
        "state": STATE,
        "seed": SEED,
        "sample_id": entries[0]["sample_id"],
        "episode_id": entries[0]["episode_id"],
        "split": entries[0]["split"],
        "first_segment_expected_flip": pair["first_segment_expected_flip"],
        "final_segment_expected_flip": pair["final_segment_expected_flip"],
        "source_videos": [{"condition": c, "path": str(p.relative_to(ROOT)), "sha256": sha256(p)}
                          for c, p in zip(CONDITIONS, paths)],
        "composition": "Native-size decoded frames pasted without crop, image filtering, frame replacement, or retiming; labels and timelines outside video panels. MP4 is presentation re-encoding; GIF is a scaled preview.",
        "source_frames_per_video": 17,
        "source_fps": fps_values[0],
        "mp4_sha256": sha256(mp4),
        "gif_sha256": sha256(gif),
    }
    (OUT / "showcase.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Saved {mp4} and {gif}")


if __name__ == "__main__":
    main()
