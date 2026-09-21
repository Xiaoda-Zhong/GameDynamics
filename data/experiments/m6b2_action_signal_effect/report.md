# M6B-2: action-signal effect audit

**Classification: B. ACTION SIGNAL IS STRONG BUT SEMANTIC CONTROL FAILS**

M6B-1 remains **C. EXPLICIT ACTION CONTROL FAIL** under its four predeclared motion criteria. This audit makes no model or data change and does not retrain. It tests the internal effect of the saved action adapter, not whether a generated video follows an action.

## Fixed comparison

The familiar real clip `my_way_home_episode_0028_clip_000011` supplies O0, native VACE control latents, and the generic prompt. A single float32 seed-42 initial noise tensor is reused for ORIGINAL (O), REVERSED (R), and ALL-NOOP (N). At each actual 30-step scheduler position, the same cached real latent and noise form `z_t = (1−σ)z_real + σ·noise`; only integer action IDs vary across the three frozen-transformer calls. This is a forward-noised real trajectory, not a video rollout. All comparisons use the conditional transformer branch at VACE control scale 1.0; classifier-free guidance and video decoding are outside this audit.

| Scheduler phase | Index | Timestep | Sigma |
|---|---:|---:|---:|
| early | 0 | 999 | 0.999999 |
| middle | 14 | 774 | 0.774521 |
| late | 29 | 96 | 0.096294 |

## 1. Projected action features at step 800

Each entry under a condition is **L2 norm / mean absolute value** of its 1,536-dimensional projected action vector. Global concatenates the four future positions. Cosines compare projected vectors, not action IDs. Latent position 0 has an exact zero action vector.

| Future latent position | ORIGINAL norm / mean abs | REVERSED norm / mean abs | ALL-NOOP norm / mean abs | Cos O/R | Cos O/N | Cos R/N |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 17.491 / 0.354 | 16.485 / 0.334 | 20.070 / 0.397 | 0.768 | 0.777 | 0.643 |
| 2 | 20.070 / 0.397 | 20.070 / 0.397 | 20.070 / 0.397 | 1.000 | 1.000 | 1.000 |
| 3 | 17.915 / 0.358 | 18.214 / 0.366 | 20.070 / 0.397 | 0.763 | 0.673 | 0.781 |
| 4 | 22.590 / 0.455 | 21.128 / 0.428 | 20.070 / 0.397 | 0.472 | 0.115 | 0.366 |
| Global future | 39.243 / 0.391 | 38.114 / 0.381 | 40.139 / 0.397 | 0.735 | 0.616 | 0.690 |

The second future position receives A4...A7; all three tested sequences contain four NOOPs there, so its projected vectors are identical by design. The other positions and global vectors are distinct. Global feature cosines are 0.735 (O/R), 0.616 (O/N), and 0.690 (R/N).

## 2. Action bias relative to the frozen patch embedding

For every future position, the action vector is spatially broadcast to all 16×28 patch tokens before taking its L2 norm. The denominator is the matching frozen noisy patch-embedding output before addition. Values above 1 mean the broadcast action term has larger L2 norm than that base patch slice.

| Phase | Condition | Latent 1 | Latent 2 | Latent 3 | Latent 4 | Mean | Range |
|---|---|---:|---:|---:|---:|---:|---:|
| early | original | 1.628 | 1.879 | 1.677 | 2.116 | 1.825 | 1.628–2.116 |
| early | reversed | 1.534 | 1.879 | 1.705 | 1.979 | 1.774 | 1.534–1.979 |
| early | all_noop | 1.868 | 1.879 | 1.879 | 1.880 | 1.876 | 1.868–1.880 |
| middle | original | 1.972 | 2.276 | 2.027 | 2.585 | 2.215 | 1.972–2.585 |
| middle | reversed | 1.859 | 2.276 | 2.061 | 2.418 | 2.153 | 1.859–2.418 |
| middle | all_noop | 2.263 | 2.276 | 2.271 | 2.297 | 2.277 | 2.263–2.297 |
| late | original | 1.676 | 1.997 | 1.729 | 2.359 | 1.940 | 1.676–2.359 |
| late | reversed | 1.579 | 1.997 | 1.758 | 2.206 | 1.885 | 1.579–2.206 |
| late | all_noop | 1.923 | 1.997 | 1.937 | 2.095 | 1.988 | 1.923–2.095 |

Across all 36 condition/phase/position values, the mean ratio is **1.993**, range **1.534–2.585**. The action perturbation is therefore not small at the injection site.

## 3. Frozen denoiser output differences at step 800

Primary table values use only future latent positions 1...4. Relative L2 is `||A−B|| / ((||A||+||B||)/2)`, symmetric in pair order. All metrics are calculated in FP32 from the frozen transformer outputs. Full all-position and per-position versions of every metric are in `metrics.json`.

| Phase | Pair | Absolute L2 | Relative L2 | Cosine | Mean abs diff | Max abs diff |
|---|---|---:|---:|---:|---:|---:|
| early | O/R | 144.312 | 0.380 | 0.931 | 0.343 | 1.439 |
| early | O/N | 111.729 | 0.296 | 0.958 | 0.253 | 1.711 |
| early | R/N | 175.563 | 0.448 | 0.900 | 0.382 | 2.594 |
| middle | O/R | 120.043 | 0.329 | 0.946 | 0.280 | 1.365 |
| middle | O/N | 94.546 | 0.255 | 0.969 | 0.216 | 1.805 |
| middle | R/N | 136.635 | 0.364 | 0.934 | 0.293 | 2.477 |
| late | O/R | 159.228 | 0.454 | 0.898 | 0.356 | 2.648 |
| late | O/N | 161.020 | 0.449 | 0.902 | 0.358 | 2.605 |
| late | R/N | 173.991 | 0.477 | 0.887 | 0.389 | 2.735 |

Pairwise future-output relative L2 differences span **0.255–0.477**. Against a no-adapter frozen-VACE call, individual conditions differ by relative L2 of 0.294–0.663 over these phases. A repeat ORIGINAL call at the middle timestep was **bit-identical** (maximum output difference 0), so the pair differences are not repeatability noise.

### Per-position relative L2

Latent 0 receives no direct action vector; its smaller nonzero difference reflects downstream temporal mixing. The full per-position absolute L2, cosine, mean absolute difference, and maximum difference are saved in `metrics.json`.

| Phase | Pair | Latent 0 | Latent 1 | Latent 2 | Latent 3 | Latent 4 |
|---|---|---:|---:|---:|---:|---:|
| early | O/R | 0.055 | 0.352 | 0.322 | 0.342 | 0.485 |
| early | O/N | 0.079 | 0.288 | 0.268 | 0.260 | 0.364 |
| early | R/N | 0.108 | 0.310 | 0.292 | 0.432 | 0.646 |
| middle | O/R | 0.031 | 0.270 | 0.192 | 0.346 | 0.470 |
| middle | O/N | 0.048 | 0.252 | 0.186 | 0.228 | 0.344 |
| middle | R/N | 0.044 | 0.279 | 0.190 | 0.347 | 0.551 |
| late | O/R | 0.081 | 0.424 | 0.213 | 0.499 | 0.617 |
| late | O/N | 0.103 | 0.581 | 0.234 | 0.388 | 0.525 |
| late | R/N | 0.099 | 0.520 | 0.240 | 0.464 | 0.616 |

## 4. Checkpoint trajectory at the middle timestep

Feature norms are global over four future vectors. Output values are pairwise future-only relative L2 at the same fixed noisy latent and timestep.

| Adapter step | Feature L2 O / R / N | Feature cosine O/R / O/N / R/N | Output relative L2 O/R / O/N / R/N |
|---:|---:|---:|---:|
| 100 | 30.609 / 26.074 / 32.632 | 0.669 / 0.452 / 0.595 | 0.537 / 0.514 / 0.419 |
| 200 | 34.401 / 31.342 / 34.941 | 0.665 / 0.501 / 0.618 | 0.408 / 0.388 / 0.396 |
| 400 | 33.615 / 30.580 / 34.438 | 0.709 / 0.540 / 0.662 | 0.261 / 0.273 / 0.258 |
| 800 | 39.243 / 38.114 / 40.139 | 0.735 / 0.616 / 0.690 | 0.329 / 0.255 / 0.364 |

Feature norms rise overall from step 100 to 800, while pairwise denoiser separation decreases through step 400 and partly rebounds at 800. Action influence does **not** grow monotonically over checkpoints.

## Interpretation and validity

The learned sequences are distinguishable outside their intentionally shared NOOP group, the broadcast action term is larger than the base patch slice, and the frozen denoiser changes substantially for matched inputs. This rules out a weak internal action signal and global representation collapse for these three sequences. M6B-1 nevertheless failed all four motion criteria, so the exploratory M6B-2 label is **B. ACTION SIGNAL IS STRONG BUT SEMANTIC CONTROL FAILS**. The audit does not identify why the strong internal perturbation fails to produce action-aligned motion; it measures a conditional forward pass on a fixed real trajectory, not the full guided sampling trajectory.

All pretrained transformer parameters remained frozen and unchanged; no optimizer or backward pass ran. Native VACE O0 conditioning and the installed all-white compressed latent mask were reused unchanged. Audit runtime was 5.322 s; peak PyTorch GPU allocation was 4359.1 MiB. No video or training data was generated.

## Artifacts

Protocol: `/root/autodl-tmp/GameDynamics_starter/data/experiments/m6b2_action_signal_effect/protocol.json`. Full numerical results: `/root/autodl-tmp/GameDynamics_starter/data/experiments/m6b2_action_signal_effect/metrics.json`. Read-only runner: `training/wan/vace_action_effect_audit.py`. The M6B-1 checkpoints and official report were not modified.
