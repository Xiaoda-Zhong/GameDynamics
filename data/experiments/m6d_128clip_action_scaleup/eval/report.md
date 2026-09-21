# M6D: 128-clip action generalization scale-up pilot

**Classification: C. GENERALIZATION FAIL** (0/4 predeclared criteria passed at the predetermined step-2000 checkpoint).

The unchanged M6B-4 cross-attention architecture was trained from a fresh seed-42 zero-initialized adapter on 128 real TRAIN clips. VACE, VAE, and text weights remained frozen; native O0 and mask behavior stayed unchanged. The official evaluation uses exactly the four M6C held-out O0s, three conditions, three seeds, generic prompt, inference settings, and future-only optical-flow implementation.

## Direct held-out comparison

| Criterion | Required | M6C: 16 clips | M6D: 128 clips | M6D result |
|---|---:|---:|---:|---|
| Turn direction accuracy | ≥0.70 | 0.827 | 0.542 | FAIL |
| Median NOOP motion ratio | ≤0.50 | 0.630 | 1.095 | FAIL |
| Median absolute action-motion correlation | ≥0.50 | 0.621 | 0.236 | FAIL |
| Matched-seed flip | ≥9/12 | 8/12 | 1/12 | FAIL |

## Per held-out O0

| State | Split / episode / sample | Turn direction | Median NOOP ratio | Median abs(r) | Flip |
|---|---|---:|---:|---:|---:|
| A | test / episode_0040 / my_way_home_episode_0040_clip_000026 | 23/42 = 0.548 | 0.890 | 0.485 | 1/3 |
| B | test / episode_0017 / my_way_home_episode_0017_clip_000031 | 17/42 = 0.405 | 0.642 | 0.168 | 0/3 |
| C | val / episode_0014 / my_way_home_episode_0014_clip_000017 | 31/42 = 0.738 | 1.383 | 0.253 | 0/3 |
| D | test / episode_0007 / my_way_home_episode_0007_clip_000006 | 20/42 = 0.476 | 1.228 | 0.192 | 0/3 |

## Training coverage and validity

The deterministic seed-42 subset uses 128 eligible TRAIN clips across all 41 TRAIN episodes (three clips per episode, with a fourth from five episodes). Action-ID counts 0–5 are **364, 339, 285, 348, 296, 416**, totaling 2,048 real transitions; all six IDs are represented. The subset and its checksum are saved separately, with no VAL/TEST or M6C held-out overlap. Training ran exactly 2,000 optimizer steps with AdamW at 2e-4 and batch/accumulation 1/1. Adapter-only checkpoints at 500, 1000, 1500, and 2000 reload successfully. All pretrained VACE parameters remained frozen and unchanged; VAE and text weights were not trained. Training runtime was 761.368 seconds, with peak allocated/reserved VRAM of 4,624.4/5,018.0 MiB.

All 36 evaluation videos decode to 17 frames and record native black-O0/white-future masks, active O0 conditioning, and 60 action-hook calls at each selected block. Generation runtime was 734.404 seconds, with 14,046 MiB peak observed VRAM. The per-state table shows that the aggregate failure is not confined to one held-out O0: only state C reaches the turn-direction threshold, while every state fails NOOP suppression and at least two fail the flip test.

## Protocol and artifacts

The M6C evaluator's `score_future` and `flip_future` functions were reused unchanged. Flow uses generated O1→O2 through O15→O16, aligned to A1…A15; O0→O1 is excluded. The fixed source sign calibration and OpenCV 4.12 Farneback settings are unchanged. The four held-out episodes remain absent from the 128-clip TRAIN subset. The step-2000 checkpoint was specified before evaluation; no checkpoint was selected using held-out scores.

Subset: `/root/autodl-tmp/GameDynamics_starter/data/experiments/m6d_128clip_action_scaleup/subset_manifest.json`. Training log: `/root/autodl-tmp/GameDynamics_starter/data/experiments/m6d_128clip_action_scaleup/train/steps.jsonl`. Checkpoints: `/root/autodl-tmp/GameDynamics_starter/data/experiments/m6d_128clip_action_scaleup/checkpoints`. Generation: `/root/autodl-tmp/GameDynamics_starter/data/experiments/m6d_128clip_action_scaleup/eval/generation_summary.json`. 36 videos: `/root/autodl-tmp/GameDynamics_starter/data/experiments/m6d_128clip_action_scaleup/eval/videos`. Full metrics: `/root/autodl-tmp/GameDynamics_starter/data/experiments/m6d_128clip_action_scaleup/eval/metrics.json`. Contact sheet: `/root/autodl-tmp/GameDynamics_starter/data/experiments/m6d_128clip_action_scaleup/eval/contact_sheet.png`.
