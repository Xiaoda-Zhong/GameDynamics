# M6A-3: four-state current-state conditioning

**Classification: C. MULTI-STATE CONDITIONING FAIL**
Engineering validity: PASS.

Generic prompt for every state and seed: A first-person gameplay view in a retro 3D stone maze.

## Real source states

| State | TRAIN sample ID | Exact source O0 |
|---|---|---|
| A | `my_way_home_episode_0028_clip_000011` | `/root/autodl-tmp/GameDynamics_starter/data/episodes/my_way_home/episode_0028/rgb/000088.png` |
| B | `my_way_home_episode_0005_clip_000053` | `/root/autodl-tmp/GameDynamics_starter/data/episodes/my_way_home/episode_0005/rgb/000424.png` |
| C | `my_way_home_episode_0010_clip_000058` | `/root/autodl-tmp/GameDynamics_starter/data/episodes/my_way_home/episode_0010/rgb/000464.png` |
| D | `my_way_home_episode_0038_clip_000028` | `/root/autodl-tmp/GameDynamics_starter/data/episodes/my_way_home/episode_0038/rgb/000224.png` |

Source O0 pairwise SSIM (same metric as evaluation):

| | A | B | C | D |
|---|---:|---:|---:|---:|
| A | 1.000 | 0.252 | 0.277 | 0.186 |
| B | 0.252 | 1.000 | 0.439 | 0.415 |
| C | 0.277 | 0.439 | 1.000 | 0.338 |
| D | 0.186 | 0.415 | 0.338 | 1.000 |

## Step-1600 result

State ID: 7/12 (target ≥10/12). Mean own-state SSIM: 0.351 (target ≥0.60). Median separation margin: 0.026 (target ≥0.05).
Median own-state SSIM: 0.379; RGB MAE mean/median: 51.27/43.53 (0–255 units).

| True | Seed | SSIM A | SSIM B | SSIM C | SSIM D | Predicted | Own MAE | Margin |
|---|---:|---:|---:|---:|---:|---|---:|---:|
| A | 42 | 0.553 | 0.069 | 0.025 | 0.050 | A | 35.13 | 0.484 |
| A | 43 | 0.560 | 0.068 | 0.021 | 0.042 | A | 35.55 | 0.492 |
| A | 44 | 0.554 | 0.085 | 0.026 | 0.050 | A | 35.47 | 0.470 |
| B | 42 | 0.188 | 0.165 | 0.070 | 0.169 | A | 42.96 | -0.024 |
| B | 43 | 0.145 | 0.318 | 0.144 | 0.258 | B | 37.44 | 0.060 |
| B | 44 | 0.250 | 0.178 | 0.089 | 0.191 | A | 41.22 | -0.072 |
| C | 42 | 0.189 | 0.661 | 0.482 | 0.494 | B | 86.74 | -0.179 |
| C | 43 | 0.249 | 0.676 | 0.628 | 0.511 | B | 71.11 | -0.048 |
| C | 44 | 0.206 | 0.675 | 0.440 | 0.510 | B | 91.12 | -0.235 |
| D | 42 | 0.045 | 0.083 | 0.080 | 0.107 | D | 46.70 | 0.024 |
| D | 43 | 0.048 | 0.079 | 0.067 | 0.108 | D | 47.68 | 0.029 |
| D | 44 | 0.065 | 0.089 | 0.070 | 0.121 | D | 44.10 | 0.032 |

## Per-state fidelity

| State | ID | Mean SSIM | Median SSIM | Mean RGB MAE | Median RGB MAE | Median margin |
|---|---:|---:|---:|---:|---:|---:|
| A | 3/3 | 0.556 | 0.554 | 35.38 | 35.47 | 0.484 |
| B | 1/3 | 0.220 | 0.178 | 40.54 | 41.22 | -0.024 |
| C | 0/3 | 0.517 | 0.482 | 82.99 | 86.74 | -0.179 |
| D | 3/3 | 0.112 | 0.108 | 46.16 | 46.70 | 0.029 |

## Checkpoint trajectory (seed 42)

| Checkpoint | State ID | Mean own-state SSIM | A | B | C | D |
|---|---:|---:|---:|---:|---:|---:|
| Untrained control | 1/4 | 0.114 | 0.132 | 0.111 | 0.123 | 0.090 |
| 200 | 2/4 | 0.193 | 0.111 | 0.248 | 0.240 | 0.172 |
| 400 | 1/4 | 0.224 | 0.121 | 0.326 | 0.293 | 0.155 |
| 800 | 2/4 | 0.316 | 0.520 | 0.116 | 0.408 | 0.218 |
| 1600 | 2/4 | 0.327 | 0.553 | 0.165 | 0.482 | 0.107 |

## Engineering validity and runtime

Finite 1600 loss/gradient records: True. Balanced samples (400/state): True. Observed counts: {'my_way_home_episode_0028_clip_000011': 400, 'my_way_home_episode_0005_clip_000053': 400, 'my_way_home_episode_0010_clip_000058': 400, 'my_way_home_episode_0038_clip_000028': 400}.
All four checkpoints load into 241 targets: True. All 28 videos decode to 17 frames with active O0: True.
Training runtime: 968.14 s; training peak GPU memory: 11586 MiB. Evaluation runtime: 543.42 s; evaluation peak GPU memory: 11494 MiB.

## Artifacts

Selected states: `/root/autodl-tmp/GameDynamics_starter/data/experiments/m6a3_four_state_1600/selected_states.json`. Pairwise source O0 SSIM: `/root/autodl-tmp/GameDynamics_starter/data/experiments/m6a3_four_state_1600/pairwise_source_o0_ssim.json`. All 28 videos: `/root/autodl-tmp/GameDynamics_starter/data/experiments/m6a3_four_state_1600/eval/videos`. Metrics: `/root/autodl-tmp/GameDynamics_starter/data/experiments/m6a3_four_state_1600/eval/metrics.json`. Contact sheet: `/root/autodl-tmp/GameDynamics_starter/data/experiments/m6a3_four_state_1600/eval/contact_sheet.png`.
Per-step training log: `/root/autodl-tmp/GameDynamics_starter/data/experiments/m6a3_four_state_1600/train_1600/steps.jsonl`.
Checkpoints: step 200 `/root/autodl-tmp/GameDynamics_starter/data/experiments/m6a3_four_state_1600/train_1600/lora_weights/000200`, step 400 `/root/autodl-tmp/GameDynamics_starter/data/experiments/m6a3_four_state_1600/train_1600/lora_weights/000400`, step 800 `/root/autodl-tmp/GameDynamics_starter/data/experiments/m6a3_four_state_1600/train_1600/lora_weights/000800`, step 1600 `/root/autodl-tmp/GameDynamics_starter/data/experiments/m6a3_four_state_1600/train_1600/lora_weights/001600`.
