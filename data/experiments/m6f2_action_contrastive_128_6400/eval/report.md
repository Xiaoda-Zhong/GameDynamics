# M6F-2: controlled action-contrastive held-out test

Predetermined step-6400 contrastive checkpoint; 2/4 predeclared criteria pass.

The 128-clip subset, initial adapter, sample/timestep/noise streams, model architecture, frozen VACE, inference settings, and four held-out O0 states match M6E. Only the TRAIN objective changed. Margin 0.01 and lambda 0.25 were locked from the TRAIN-only M6F-1 audit before this run.

## Controlled training validity

The fresh seed-42 adapter was tensor-identical to M6D/M6E's saved zero initialization (SHA-256 `06ad38440c82a16576632c7f89922c0d780449ad817392c2935fe89a0ac23cf1`). All 6,400 sample indices, timestep indices, and sigmas matched the M6D/M6E logs; the first correct-action loss was exactly equal in both runs (`0.0779081583`). AdamW, LR 2e-4, batch 1, gradient accumulation 1, BF16 backbone, native O0/mask behavior, and frozen pretrained weights were unchanged.

Within each eligible batch, both actions used the same O0 control, real target, timestep, BF16 noise tensor, text, and unchanged model weights; the optimizer updated only after both branch gradients were accumulated. All 128 clips were presented 50 times. There were 6,200 swappable presentations and 200 unchanged presentations; 2,360 eligible presentations (38.1%) had an active ranking hinge. Mean correct/ranking/total training losses were 0.147405/0.002900/0.148130. Every logged loss and gradient norm was finite. Adapter and full training-state checkpoints were saved and verified at steps 1600, 3200, 4800, and 6400. Training runtime was 4,241.7 seconds; peak allocated/reserved VRAM was 4,640.1/5,018.0 MiB. Only step 6400 was evaluated.

## Direct held-out comparison

| Criterion | Threshold | M6E standard objective | M6F-2 contrastive | M6F-2 result |
|---|---:|---:|---:|---|
| Turn direction accuracy | ≥0.70 | 0.857 | 0.887 | PASS |
| Median NOOP motion ratio | ≤0.50 | 0.619 | 0.572 | FAIL |
| Median absolute action-motion correlation | ≥0.50 | 0.718 | 0.786 | PASS |
| Matched-seed LEFT/RIGHT flip | ≥9/12 | 8/12 | 7/12 | FAIL |

## Per held-out state

| State | Split / episode / sample | Turn direction | Median NOOP ratio | Median abs(r) | Flip |
|---|---|---:|---:|---:|---:|
| A | test / episode_0040 / my_way_home_episode_0040_clip_000026 | 37/42 = 0.881 | 0.540 | 0.800 | 1/3 |
| B | test / episode_0017 / my_way_home_episode_0017_clip_000031 | 32/42 = 0.762 | 0.387 | 0.760 | 1/3 |
| C | val / episode_0014 / my_way_home_episode_0014_clip_000017 | 40/42 = 0.952 | 0.745 | 0.748 | 2/3 |
| D | test / episode_0007 / my_way_home_episode_0007_clip_000006 | 40/42 = 0.952 | 0.627 | 0.835 | 3/3 |

## Interpretation

Matched-seed flips changed by **−1/12** (8/12 to 7/12), so the primary LEFT/RIGHT discrimination goal did not improve. Turn accuracy changed by +0.030 and absolute action-motion correlation by +0.068, without degrading those controls. The NOOP ratio improved by 0.048 but still misses 0.50. The contrastive run passes 2/4 criteria, the same count as M6E. State-level flip counts changed from M6E A/B/C/D = 0/2/3/3 to M6F-2 = 1/1/2/3. These are the predetermined step-6400 results; no checkpoint was selected from held-out performance.

## Protocol and artifacts

The unchanged M6C scorer uses generated O1→O2 through O15→O16 and excludes O0→O1. All 36 videos decode to 17 frames; native VACE O0 masking and 60 action-hook calls per selected block are validated by the scorer. Each matched O0/seed group resets the CUDA noise generator identically.

Checkpoint: `/root/autodl-tmp/GameDynamics_starter/data/experiments/m6f2_action_contrastive_128_6400/checkpoints/step_6400/adapter.pt`. Training log: `/root/autodl-tmp/GameDynamics_starter/data/experiments/m6f2_action_contrastive_128_6400/train/steps.jsonl`. Videos: `/root/autodl-tmp/GameDynamics_starter/data/experiments/m6f2_action_contrastive_128_6400/eval/videos`. Full metrics: `/root/autodl-tmp/GameDynamics_starter/data/experiments/m6f2_action_contrastive_128_6400/eval/metrics.json`. Contact sheet: `/root/autodl-tmp/GameDynamics_starter/data/experiments/m6f2_action_contrastive_128_6400/eval/contact_sheet.png`. Generation summary: `/root/autodl-tmp/GameDynamics_starter/data/experiments/m6f2_action_contrastive_128_6400/eval/generation_summary.json`.
