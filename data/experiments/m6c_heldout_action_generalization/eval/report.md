# M6C: held-out action generalization test

**Classification: B. PARTIAL GENERALIZATION** (2/4 aggregate criteria passed).

Frozen M6B-4 step-800 cross-attention adapter, native VACE O0 conditioning, fixed generic prompt, 36 videos (four held-out O0 states × three action conditions × seeds 42–44). No training or model changes.

## Aggregate decision

| Criterion | Required | Observed | Result |
|---|---:|---:|---|
| Turn direction accuracy | ≥0.70 | 139/168 = 0.827 | PASS |
| Median NOOP motion ratio | ≤0.50 | 0.630 | FAIL |
| Median abs action-motion correlation | ≥0.50 | 0.621 | PASS |
| Matched-seed flip | ≥9/12 | 8/12 | FAIL |

## Per held-out O0

| State | Split / episode / sample | Turn direction | Median NOOP ratio | Median abs(r) | Flip |
|---|---|---:|---:|---:|---:|
| A | test / episode_0040 / my_way_home_episode_0040_clip_000026 | 29/42 = 0.690 | 0.886 | 0.604 | 1/3 |
| B | test / episode_0017 / my_way_home_episode_0017_clip_000031 | 36/42 = 0.857 | 0.433 | 0.754 | 2/3 |
| C | val / episode_0014 / my_way_home_episode_0014_clip_000017 | 38/42 = 0.905 | 0.513 | 0.668 | 3/3 |
| D | test / episode_0007 / my_way_home_episode_0007_clip_000006 | 36/42 = 0.857 | 0.697 | 0.597 | 2/3 |

All four episodes are absent from the official TRAIN split and the 16-sample M6B training set. The six pairwise source-O0 SSIM values range from 0.154 to 0.287.

## Protocol and artifacts

The exact validated M5C-5 Farneback ROI and M6B-4 source sign calibration were reused. Primary flow uses only generated O1→O2 through O15→O16, aligned to A1…A15. The median NOOP ratio is across 24 mixed-action videos; ALL-NOOP matched-seed motion ratios are descriptive. Within each O0/seed group the generator is reset to the same seed, so only the action IDs change. All 36 sidecars verify 17 frames, native black O0/white future masks, nonzero O0 control, and 60 action-hook calls per selected block. No generated frame was replaced. Generation runtime was 724.363 seconds; peak observed VRAM was 14,046 MiB.

Selection: `/root/autodl-tmp/GameDynamics_starter/data/experiments/m6c_heldout_action_generalization/selected_states.json`. Videos: `/root/autodl-tmp/GameDynamics_starter/data/experiments/m6c_heldout_action_generalization/eval/videos`. Full metrics: `/root/autodl-tmp/GameDynamics_starter/data/experiments/m6c_heldout_action_generalization/eval/metrics.json`. Per-transition flow: `/root/autodl-tmp/GameDynamics_starter/data/experiments/m6c_heldout_action_generalization/eval/transitions.jsonl`. Contact sheet: `/root/autodl-tmp/GameDynamics_starter/data/experiments/m6c_heldout_action_generalization/eval/contact_sheet.png`. Generation runtime/VRAM: `/root/autodl-tmp/GameDynamics_starter/data/experiments/m6c_heldout_action_generalization/eval/generation_summary.json`.
