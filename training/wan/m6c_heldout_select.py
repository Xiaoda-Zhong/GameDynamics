"""Select four deterministic, visually diverse, episode-disjoint VAL/TEST O0 frames."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image

from training.wan.current_frame_metrics import source_frame_rgb, ssim_rgb

ROOT = Path(__file__).resolve().parents[2]
SPLITS = ROOT / "data/datasets/my_way_home"
TRAIN_16 = ROOT / "data/wan_training/my_way_home/overfit_0016/manifest.jsonl"
OUT = ROOT / "data/experiments/m6c_heldout_action_generalization"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def run() -> dict:
    split_data = {name: json.loads((SPLITS / f"{name}.json").read_text()) for name in ("train", "val", "test")}
    trained = [json.loads(line) for line in TRAIN_16.read_text().splitlines() if line.strip()]
    train_ids = {row["global_clip_id"] for row in trained}
    train_episodes = set(split_data["train"]["episode_names"])
    if len(trained) != 16 or any(row["episode_name"] not in train_episodes for row in trained):
        raise RuntimeError("M6B training set or official train split changed")
    candidates = []
    features = []
    for split in ("val", "test"):
        manifest = split_data[split]
        for clip in manifest["clips"]:
            episode = clip["episode_name"]
            clip_id = clip["global_clip_id"]
            if episode in train_episodes or clip_id in train_ids:
                raise RuntimeError(f"Held-out leakage: {clip_id}")
            source = ROOT / "data/episodes/my_way_home" / episode / "rgb" / f"{clip['start_observation_index']:06d}.png"
            if not source.is_file():
                raise FileNotFoundError(source)
            with Image.open(source) as image:
                # Fixed feature extraction for deterministic max-min selection only.
                thumbnail = image.convert("RGB").resize((56, 32), Image.Resampling.BICUBIC)
                feature = np.asarray(thumbnail, dtype=np.float32).reshape(-1) / 255.0
            candidates.append({"split": split, "sample_id": clip_id, "episode_id": episode,
                "source_o0": str(source), "source_manifest": str(SPLITS / f"{split}.json"),
                "start_observation_index": clip["start_observation_index"]})
            features.append(feature)
    if len(candidates) != split_data["val"]["num_clips"] + split_data["test"]["num_clips"]:
        raise RuntimeError("Held-out candidate count changed")
    matrix = np.stack(features)
    # Start with the O0 farthest from the held-out mean. Subsequent choices maximize
    # distance to the closest selected O0 while requiring a new episode.
    center = matrix.mean(axis=0)
    distances = np.mean((matrix - center) ** 2, axis=1)
    chosen = [max(range(len(candidates)), key=lambda i: (float(distances[i]), -i))]
    rgb_cache = {chosen[0]: source_frame_rgb(candidates[chosen[0]]["source_o0"])}
    while len(chosen) < 4:
        used_episodes = {candidates[i]["episode_id"] for i in chosen}
        nearest = np.min(np.stack([np.mean((matrix - matrix[j]) ** 2, axis=1) for j in chosen]), axis=0)
        eligible = [i for i, row in enumerate(candidates) if row["episode_id"] not in used_episodes]
        ranked = sorted(eligible, key=lambda i: (-float(nearest[i]), i))
        selected = None
        for i in ranked:
            candidate_rgb = source_frame_rgb(candidates[i]["source_o0"])
            if all(float(ssim_rgb(candidate_rgb, rgb_cache[j])) < 0.45 for j in chosen):
                selected = i
                rgb_cache[i] = candidate_rgb
                break
        if selected is None:
            selected = ranked[0]
            rgb_cache[selected] = source_frame_rgb(candidates[selected]["source_o0"])
        chosen.append(selected)
    states = []
    for label, i in zip("ABCD", chosen):
        row = candidates[i]
        path = Path(row["source_o0"])
        states.append({"state": label, **row, "source_o0_sha256": sha256(path),
            "feature_distance_from_previous_minimum":
                None if label == "A" else float(np.min([np.mean((matrix[i] - matrix[j]) ** 2) for j in chosen[:len(states)]]))})
    rgb = [rgb_cache[i] for i in chosen]
    similarities = [[float(ssim_rgb(a, b)) for b in rgb] for a in rgb]
    result = {"milestone": "M6C", "selection_rule":
        "All 587 VAL/TEST clip O0s, sorted VAL then TEST by official manifest order. Bicubic RGB 56x32 features; first is maximum mean-squared distance from candidate centroid. At each later pick, rank new-episode candidates by descending minimum mean-squared feature distance, ties by earlier manifest order; take the first whose exact 448x256 O0 SSIM is <0.45 to every selected O0. If none qualifies, take the highest-ranked candidate.",
        "candidate_count": len(candidates), "source_split_sha256":
            {name: sha256(SPLITS / f"{name}.json") for name in ("train", "val", "test")},
        "tiny_train_manifest_sha256": sha256(TRAIN_16),
        "train_sample_ids": sorted(train_ids), "train_episode_ids": sorted(train_episodes),
        "states": states, "pairwise_o0_ssim": similarities,
        "all_episodes_distinct": len({row["episode_id"] for row in states}) == 4,
        "none_in_training_16": all(row["sample_id"] not in train_ids for row in states),
        "none_in_train_split": all(row["episode_id"] not in train_episodes for row in states)}
    if not all(result[key] for key in ("all_episodes_distinct", "none_in_training_16", "none_in_train_split")):
        raise RuntimeError("Held-out state selection failed disjointness")
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "selected_states.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


if __name__ == "__main__":
    chosen = run()
    print(json.dumps({"states": chosen["states"], "pairwise_o0_ssim": chosen["pairwise_o0_ssim"]}, indent=2))
