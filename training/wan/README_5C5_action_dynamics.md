# M5C-5: action-dynamics diagnostic

**Decision: FAIL natural-language action conditioning under the pre-declared temporal-motion criteria.** The M5C-4 scene+action LoRA gives a useful corridor and red-wall visual prior, but these 15 counterfactual videos do not show reliable action-aligned turn direction or timing. No model was retrained.

## Fixed protocol

Output root: `/root/autodl-tmp/outputs/wan/overfit_0016/single_sample_action_dynamics_0800/`. The checkpoint is the existing M5C-4 `train/lora_weights/000800/`, loaded once with `WanPipeline.load_lora_weights()` at scale 1.0. For each condition, seeds 42–46 were generated with Wan2.1-T2V-1.3B, 17 frames, 448×256, BF16 pipeline, FP32 VAE, 30 inference steps, UniPC scheduler with flow shift 3.0, guidance 5.0, and no negative prompt. The ORIGINAL seed-42 video is pixel-identical to the prior M5C-4 step-800 video across all 17 frames.

Every prompt begins with the same prefix: **“A first-person gameplay view in a retro 3D stone corridor with a red wall.”** The action sentences use the existing deterministic caption renderer:

| Condition | Action sequence | Action sentence |
| --- | --- | --- |
| ORIGINAL | `MOVE_FORWARD_LEFT` ×2, `NOOP` ×8, `MOVE_FORWARD_RIGHT` ×6 | “The player moves forward while turning left for 2 steps, stays still for 8 steps, then moves forward while turning right for 6 steps.” |
| REVERSED | `MOVE_FORWARD_RIGHT` ×2, `NOOP` ×8, `MOVE_FORWARD_LEFT` ×6 | “The player moves forward while turning right for 2 steps, stays still for 8 steps, then moves forward while turning left for 6 steps.” |
| NOOP | `NOOP` ×16 | “The player stays still for 16 steps.” |

`protocol.json` records every raw action string, complete prompt, checkpoint and source hashes, inference settings, flow procedure, and thresholds. It was written before the GPU generations. No source manifest or earlier experiment output was modified.

## Optical-flow method and source calibration

The flow measurement uses OpenCV 4.12.0 Farneback dense flow on consecutive grayscale frames. The real source is resized with cubic interpolation to 448×256; generated videos are already that size. The fixed region is `[x=45:403, y=26:230]`, the central 80% of the frame; no texture mask is applied. Farneback parameters are `pyr_scale=0.5`, `levels=3`, `winsize=15`, `iterations=3`, `poly_n=5`, `poly_sigma=1.2`, and `flags=0`, with one OpenCV thread. Units are pixels per 8-fps frame transition. `mean_u`, `median_u`, and mean flow magnitude are saved for each of 16 transitions in every video.

The **real source establishes the sign**: its two left-turn transitions average **+3.753 px** horizontal flow, while its six right-turn transitions average **−3.242 px**. Thus positive raw `u` denotes the source's left-turn camera motion and negative raw `u` denotes right-turn camera motion. All direction, correlation, and flip decisions use **mean** `u`; median `u` is reported as a diagnostic. The broad red wall makes the left-turn median nearly zero even though the mean and the actual camera motion are clearly positive.

| Source transition | Action | Mean u | Median u | Mean magnitude |
| ---: | --- | ---: | ---: | ---: |
| 0 | LEFT | +2.954 | +0.002 | 3.014 |
| 1 | LEFT | +4.553 | +0.004 | 4.593 |
| 2 | NOOP | +0.907 | +0.009 | 0.954 |
| 3 | NOOP | +0.901 | +0.012 | 0.939 |
| 4 | NOOP | +0.994 | +0.020 | 1.027 |
| 5 | NOOP | +1.044 | +0.054 | 1.079 |
| 6 | NOOP | +0.969 | +0.794 | 0.996 |
| 7 | NOOP | +1.016 | +1.007 | 1.040 |
| 8 | NOOP | +1.123 | +1.142 | 1.145 |
| 9 | NOOP | +1.070 | +1.087 | 1.093 |
| 10 | RIGHT | −2.947 | −3.781 | 3.338 |
| 11 | RIGHT | −2.482 | −3.234 | 3.288 |
| 12 | RIGHT | −2.309 | −3.643 | 3.516 |
| 13 | RIGHT | −1.791 | −3.409 | 3.650 |
| 14 | RIGHT | −1.321 | −3.282 | 3.969 |
| 15 | RIGHT | −8.600 | −10.928 | 8.699 |

The source scores **8/8 = 1.000** turn direction, **R_noop = 0.243**, and **signed r = +0.843**. This validates that the fixed procedure detects the known turn segments and NOOP period in the real clip. The positive flow during source NOOP likely reflects continuing camera movement; the pre-declared ratio still distinguishes it from turns.

## Per-video results

`R_noop` below is mean flow magnitude on the eight NOOP transitions divided by mean magnitude on the eight turn transitions **within that video**. The correlation uses action signal `+1` for left, `0` for NOOP, and `−1` for right against calibrated mean `u`. A positive signed `r` means action-aligned direction; a negative `r` means anti-alignment. For a 16-NOOP video, direction accuracy, within-video `R_noop`, and correlation are mathematically undefined. Its separate `matched NOOP/turn` ratio compares its mean motion to the average turn motion of ORIGINAL and REVERSED at the same seed.

| Condition | Seed | Video path under `videos/` | Directed turns | R_noop | Signed r | \|r\| | Matched NOOP/turn |
| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: |
| ORIGINAL | 42 | `original_seed_42.mp4` | 1/8 | 0.090 | +0.438 | 0.438 | — |
| ORIGINAL | 43 | `original_seed_43.mp4` | 3/8 | 0.838 | −0.608 | 0.608 | — |
| ORIGINAL | 44 | `original_seed_44.mp4` | 4/8 | 1.099 | +0.152 | 0.152 | — |
| ORIGINAL | 45 | `original_seed_45.mp4` | 6/8 | 6.006 | −0.096 | 0.096 | — |
| ORIGINAL | 46 | `original_seed_46.mp4` | 3/8 | 1.443 | −0.207 | 0.207 | — |
| REVERSED | 42 | `reversed_seed_42.mp4` | 4/8 | 0.115 | −0.572 | 0.572 | — |
| REVERSED | 43 | `reversed_seed_43.mp4` | 2/8 | 1.141 | −0.004 | 0.004 | — |
| REVERSED | 44 | `reversed_seed_44.mp4` | 3/8 | 0.969 | −0.418 | 0.418 | — |
| REVERSED | 45 | `reversed_seed_45.mp4` | 2/8 | 1.990 | +0.207 | 0.207 | — |
| REVERSED | 46 | `reversed_seed_46.mp4` | 5/8 | 0.913 | −0.078 | 0.078 | — |
| NOOP | 42 | `noop_seed_42.mp4` | — | — | — | — | 0.157 |
| NOOP | 43 | `noop_seed_43.mp4` | — | — | — | — | 0.307 |
| NOOP | 44 | `noop_seed_44.mp4` | — | — | — | — | 1.285 |
| NOOP | 45 | `noop_seed_45.mp4` | — | — | — | — | 0.041 |
| NOOP | 46 | `noop_seed_46.mp4` | — | — | — | — | 0.956 |

The ORIGINAL and REVERSED directional accuracies are **0.425** and **0.400** respectively. Their median signed correlations are **−0.096** and **−0.078**; their median absolute correlations are **0.207** and **0.207**. Median within-video `R_noop` is **1.099** for ORIGINAL and **0.969** for REVERSED. All 10 turn-containing videos pooled yield **33/80 = 0.413** direction accuracy, median **R_noop = 1.034**, and median **|r| = 0.207**.

The five all-NOOP controls have median matched NOOP/turn ratio **0.307**, but it ranges from **0.041 to 1.285**. This suggests that the all-NOOP sentence can reduce overall motion in some seeds. It does **not** establish that motion occurs during requested turn intervals or is suppressed during the middle NOOP interval of a mixed-action prompt.

## Counterfactual flip test

Segment values are means of calibrated per-transition mean `u`, in pixels. A seed succeeds only if ORIGINAL first two are positive and last six negative **and** REVERSED first two are negative and last six positive.

| Seed | ORIGINAL first 2 | REVERSED first 2 | ORIGINAL last 6 | REVERSED last 6 | Success |
| ---: | ---: | ---: | ---: | ---: | --- |
| 42 | +0.292 | +0.591 | +0.010 | +0.005 | No |
| 43 | −4.085 | −1.150 | +0.116 | −1.316 | No |
| 44 | +0.0069 | +0.0108 | +0.00058 | −0.00020 | No |
| 45 | −0.133 | −1.007 | −0.347 | −0.986 | No |
| 46 | −0.0070 | −0.00040 | +0.00091 | +0.00129 | No |

**Flip success: 0/5 seeds.** No seed had both expected segment reversals.

## Pre-declared decision

| Criterion | Required | Observed | Result |
| --- | ---: | ---: | --- |
| Aggregate turn direction accuracy | ≥0.70 | 0.413 | Fail |
| Median within-video R_noop | ≤0.50 | 1.034 | Fail |
| Median \|action–motion r\| | ≥0.50 | 0.207 | Fail |
| Counterfactual flip success | ≥4/5 | 0/5 | Fail |

The primary verdict is **no reliable temporally aligned natural-language turn control under this protocol**. The dense-flow measure includes any moving foreground object, so it is not a pure camera-pose estimator. Its strong response to the real source, fixed region, matched seeds, and large failures across all four criteria support this diagnostic conclusion; they do not rule out weaker effects in other samples or prompts.

## Artifacts and reproducibility

- `videos/`: all 15 MP4s and a JSON metadata file beside each one. All decode to 17 frames at 448×256. The prompt, raw action strings, seed, checkpoint, scale, and inference settings are recorded per video.
- `metrics/transitions.jsonl`: 256 records, 16 source plus 16 for each generated video. Every record includes mean/median `u`, mean `v`, mean magnitude, calibrated `u`, inferred direction, action, and turn correctness where applicable.
- `metrics/source.json`: source calibration, 16 source transitions, and source video metrics.
- `metrics/video_metrics.jsonl`: source plus 15 generated per-video summaries.
- `metrics/aggregate.json`: all aggregate metrics, flip tests, thresholds, and pass/fail flags.
- `generation_summary.json`: 15 paths, generation runtime **268.55 s**, peak observed VRAM **11,494 MiB**, and adapter/reference checks.
- `summary.json`: condensed experiment result. Console logs are in the parent output directory.

The flow analysis was run twice with byte-identical metric files. OpenCV was installed only under `/tmp/m5c5_opencv` because it was absent from the environment; the project's requirements were not changed. To reproduce with the existing model and checkpoint, install `opencv-python-headless==4.12.0.88` in that isolated target, then run `python -m training.wan.finetrainers_action_dynamics generate` and `PYTHONPATH=/tmp/m5c5_opencv:$PYTHONPATH python -m training.wan.finetrainers_action_dynamics analyze`. The runner is `training/wan/finetrainers_action_dynamics.py` and the flow implementation is `training/wan/action_dynamics_metrics.py`.
