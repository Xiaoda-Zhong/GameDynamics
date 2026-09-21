"""Build a deterministic, episode-balanced 128-clip TRAIN-only 17-frame subset."""

from __future__ import annotations

import collections
import hashlib
import json
from pathlib import Path

from data_pipeline.prepare_wan_overfit import evaluate_candidates
from training.wan.vace_action_adapter import ACTION_NAMES

ROOT = Path(__file__).resolve().parents[2]
DATASETS = ROOT / "data/datasets/my_way_home"
EPISODES = ROOT / "data/episodes/my_way_home"
M6C = ROOT / "data/experiments/m6c_heldout_action_generalization/selected_states.json"
OUT = ROOT / "data/experiments/m6d_128clip_action_scaleup"
SEED = 42
COUNT = 128


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        while data := stream.read(1024 * 1024):
            h.update(data)
    return h.hexdigest()


def rank(key: str) -> str:
    return hashlib.sha256(f"{SEED}:{key}".encode()).hexdigest()


def run() -> dict:
    splits = {name: json.loads((DATASETS / f"{name}.json").read_text())
              for name in ("train", "val", "test")}
    heldout = json.loads(M6C.read_text())
    heldout_ids = {row["sample_id"] for row in heldout["states"]}
    forbidden_episodes = set(splits["val"]["episode_names"] + splits["test"]["episode_names"])
    train = splits["train"]
    eligible, ineligible = evaluate_candidates({"clips": train["clips"]}, EPISODES)
    by_episode = collections.defaultdict(list)
    for row in eligible:
        if row["dataset_split"] != "train" or len(row["rgb_frames"]) != 17 or len(row["raw_actions"]) != 16:
            raise RuntimeError("Ineligible 17-frame TRAIN record passed source validation")
        if row["global_clip_id"] in heldout_ids or row["episode_name"] in forbidden_episodes:
            raise RuntimeError("VAL/TEST leakage into eligible TRAIN clips")
        by_episode[row["episode_name"]].append(row)
    episode_order = sorted(by_episode, key=lambda name: (rank(name), name))
    if len(episode_order) != train["num_episodes"] or len(episode_order) > COUNT:
        raise RuntimeError("Cannot cover all TRAIN episodes with 128 clips")
    for episode in episode_order:
        by_episode[episode].sort(key=lambda row: (rank(row["global_clip_id"]), row["global_clip_id"]))
        if len(by_episode[episode]) < 4:
            raise RuntimeError(f"Episode lacks enough eligible clips: {episode}")
    # Round robin gives each of 41 episodes three clips; five receive a fourth.
    selected = [by_episode[episode][round_index]
                for round_index in range(4) for episode in episode_order
                if round_index < len(by_episode[episode])][:COUNT]
    ids = [row["global_clip_id"] for row in selected]
    if len(ids) != COUNT or len(set(ids)) != COUNT:
        raise RuntimeError("Selected clip count or uniqueness changed")
    counts = collections.Counter(action for row in selected for action in row["raw_actions"])
    by_id = {str(index): counts[name] for index, name in enumerate(ACTION_NAMES)}
    if any(value == 0 for value in by_id.values()):
        raise RuntimeError(f"Missing action before training: {by_id}")
    selected_episodes = collections.Counter(row["episode_name"] for row in selected)
    if min(selected_episodes.values()) != 3 or max(selected_episodes.values()) != 4:
        raise RuntimeError("128-clip episode balancing failed")
    result = {
        "milestone": "M6D", "dataset_split": "train", "selection_seed": SEED,
        "selection_rule": "Validate all official TRAIN clips with existing extend_candidate; SHA256(seed:episode_id) episode order and SHA256(seed:clip_id) within-episode order, then round-robin episodes to 128 clips. Each of 41 episodes gets three clips; first five in ranked episode order get a fourth.",
        "source_train_manifest": str(DATASETS / "train.json"),
        "source_train_manifest_sha256": digest(DATASETS / "train.json"),
        "source_val_manifest_sha256": digest(DATASETS / "val.json"),
        "source_test_manifest_sha256": digest(DATASETS / "test.json"),
        "m6c_heldout_selection_sha256": digest(M6C),
        "eligible_clip_count": len(eligible), "ineligible_counts": dict(ineligible),
        "clip_count": COUNT, "episode_count": len(selected_episodes),
        "clip_ids": ids, "episode_ids": [row["episode_name"] for row in selected],
        "episode_counts": dict(sorted(selected_episodes.items())),
        "action_id_counts": by_id, "action_names_by_id": list(ACTION_NAMES),
        "action_transition_count": sum(by_id.values()),
        "all_six_actions_present": True, "val_test_disjoint": True,
        "m6c_heldout_clips_disjoint": True, "records": selected,
    }
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / "subset_manifest.json"
    if path.exists() and json.loads(path.read_text()) != result:
        raise RuntimeError("Existing M6D subset differs; refusing to reselect")
    path.write_text(json.dumps(result, indent=2) + "\n")
    (OUT / "subset_manifest.sha256").write_text(digest(path) + "\n")
    return {key: result[key] for key in ("clip_count", "episode_count", "eligible_clip_count",
        "ineligible_counts", "action_id_counts", "action_transition_count")}


if __name__ == "__main__":
    print(json.dumps(run(), indent=2))
