# M6E: step-6400 held-out action budget test

**Classification: B. PARTIALLY BUDGET-LIMITED** (2/4 predeclared criteria pass).

Only the predetermined step-6400 adapter was evaluated. The exact M6C/M6D held-out states, conditions, seeds, matched noise, prompt, inference settings, and future-only optical-flow scorer were reused. No training or model change was made for this evaluation.

## Direct held-out comparison

| Criterion | Threshold | 16 clips @ 800 | 128 clips @ 2000 | 128 clips @ 6400 | Step-6400 result |
|---|---:|---:|---:|---:|---|
| Turn direction accuracy | ≥0.70 | 0.827 | 0.542 | 0.857 | PASS |
| Median NOOP motion ratio | ≤0.50 | 0.630 | 1.095 | 0.619 | FAIL |
| Median abs action-motion correlation | ≥0.50 | 0.621 | 0.236 | 0.718 | PASS |
| Matched-seed flip | ≥9/12 | 8/12 | 1/12 | 8/12 | FAIL |

## Per held-out O0 at step 6400

| State | Split / episode / sample | Turn direction | Median NOOP ratio | Median abs(r) | Flip |
|---|---|---:|---:|---:|---:|
| A | test / episode_0040 / my_way_home_episode_0040_clip_000026 | 31/42 = 0.738 | 0.694 | 0.743 | 0/3 |
| B | test / episode_0017 / my_way_home_episode_0017_clip_000031 | 38/42 = 0.905 | 0.493 | 0.796 | 2/3 |
| C | val / episode_0014 / my_way_home_episode_0014_clip_000017 | 38/42 = 0.905 | 0.730 | 0.638 | 3/3 |
| D | test / episode_0007 / my_way_home_episode_0007_clip_000006 | 37/42 = 0.881 | 0.782 | 0.718 | 3/3 |

## Protocol and artifacts

The unchanged scorer uses generated O1→O2 through O15→O16 aligned to A1…A15. The observed O0→generated O1 transition is excluded. All videos decode to 17 frames; native VACE O0 masking and action-hook calls are checked by the existing evaluator. The same seed resets the CUDA generator for each action condition within an O0/seed group.

Official adapter: `/root/autodl-tmp/GameDynamics_starter/data/experiments/m6d_128clip_action_scaleup/checkpoints/step_6400/adapter.pt`. Protocol: `/root/autodl-tmp/GameDynamics_starter/data/experiments/m6e_128clip_6400_heldout/eval/protocol.json`. 36 videos: `/root/autodl-tmp/GameDynamics_starter/data/experiments/m6e_128clip_6400_heldout/eval/videos`. Metrics: `/root/autodl-tmp/GameDynamics_starter/data/experiments/m6e_128clip_6400_heldout/eval/metrics.json`. Contact sheet: `/root/autodl-tmp/GameDynamics_starter/data/experiments/m6e_128clip_6400_heldout/eval/contact_sheet.png`. Runtime/VRAM: `/root/autodl-tmp/GameDynamics_starter/data/experiments/m6e_128clip_6400_heldout/eval/generation_summary.json`.
