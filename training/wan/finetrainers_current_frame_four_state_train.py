#!/usr/bin/env python3
"""M6A-3: four exact TRAIN clips with one generic current-state prompt."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from training.wan.finetrainers_current_frame_train import RANK, ALPHA, LR, run_training
from training.wan.finetrainers_smoke import FINETRAINERS_COMMIT, MANIFEST, MODEL


ROOT = Path(__file__).resolve().parents[2] / "data/experiments/m6a3_four_state_1600"
TRAIN = ROOT / "train_1600"
SELECTED = ROOT / "selected_states.json"
PROMPT = "A first-person gameplay view in a retro 3D stone maze."
CHECKPOINTS = (200, 400, 800, 1600)


def main() -> None:
    selected = json.loads(SELECTED.read_text())
    states = selected["states"]
    if ([state["state"] for state in states] != ["A", "B", "C", "D"]
            or len({state["sample_id"] for state in states}) != 4
            or selected["generic_prompt"] != PROMPT
            or selected["selection"]["source_manifest_sha256"] != hashlib.sha256(MANIFEST.read_bytes()).hexdigest()
            or selected["selection"]["maximum_pairwise_ssim"] >= 0.45):
        raise ValueError("Selected-state metadata violates the M6A-3 protocol")
    for state in states:
        if (state["dataset_split"] != "train" or state["source_o0"] != state["source_rgb_frames"][0]
                or len(state["source_rgb_frames"]) != 17 or len(state["raw_actions"]) != 16
                or not all(isinstance(action, str) for action in state["raw_actions"])):
            raise ValueError(f"Invalid exact source clip for state {state['state']}")
    protocol = {
        "experiment": "M6A-3 four-state current-state conditioning",
        "selected_states": str(SELECTED),
        "sample_ids": [state["sample_id"] for state in states],
        "generic_prompt": PROMPT,
        "source_manifest_sha256": hashlib.sha256(MANIFEST.read_bytes()).hexdigest(),
        "model": str(MODEL), "finetrainers_commit": FINETRAINERS_COMMIT,
        "rank": RANK, "alpha": ALPHA, "learning_rate": LR,
        "frames": 17, "width": 448, "height": 256,
        "dtype": "bf16", "batch_size": 1, "gradient_accumulation_steps": 1,
        "seed": 42, "optimizer_steps": 1600,
        "checkpoints": list(CHECKPOINTS),
        "expected_presentations_per_state": 400,
        "pass_thresholds": {
            "state_identification_correct_min": 10,
            "own_state_mean_ssim_min": 0.60,
            "median_separation_margin_min": 0.05,
        },
    }
    path = ROOT / "training_protocol.json"
    if path.exists() and json.loads(path.read_text()) != protocol:
        raise ValueError("Existing training protocol differs")
    path.write_text(json.dumps(protocol, indent=2) + "\n")
    result = run_training(
        TRAIN, 1600, CHECKPOINTS,
        sample_ids=tuple(protocol["sample_ids"]), prompt_override=PROMPT,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
