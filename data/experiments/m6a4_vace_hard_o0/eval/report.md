# M6A-4: VACE hard first-frame conditioning smoke test

**Classification: C. VACE CONDITIONING FAILS**

Model: `Wan-AI/Wan2.1-VACE-1.3B-diffusers` at `ec4d2cb062b548996b179d493fdd05340de702a1` using Diffusers 0.40.0. Inference only; no LoRA or training.

Prompt for every state and seed: A first-person gameplay view in a retro 3D stone maze.

## Decision

Decoded frame-0 SSIM minimum/mean/median: 0.780/0.921/0.918 (every output must be >= 0.95).
Early-future state ID: 12/12 (required >= 10/12). Median early-future margin: 0.162 (required >= 0.05).
The native O0 mask and conditioning latent were active. Classification C means the decoded generated frame 0 fails the predeclared hard-anchor threshold; it does not mean that O0 conditioning was absent. Early-future state identification passes.

## Decoded-video results

| True | Seed | Frame-0 SSIM | Future A | Future B | Future C | Future D | Predicted | Margin |
|---|---:|---:|---:|---:|---:|---:|---|---:|
| A | 42 | 0.904 | 0.431 | 0.365 | 0.338 | 0.290 | A | 0.066 |
| A | 43 | 0.780 | 0.445 | 0.337 | 0.266 | 0.279 | A | 0.108 |
| A | 44 | 0.911 | 0.580 | 0.336 | 0.281 | 0.263 | A | 0.245 |
| B | 42 | 0.985 | 0.284 | 0.948 | 0.412 | 0.424 | B | 0.524 |
| B | 43 | 0.989 | 0.269 | 0.618 | 0.375 | 0.394 | B | 0.224 |
| B | 44 | 0.984 | 0.286 | 0.971 | 0.403 | 0.430 | B | 0.541 |
| C | 42 | 0.962 | 0.258 | 0.405 | 0.719 | 0.331 | C | 0.314 |
| C | 43 | 0.927 | 0.246 | 0.413 | 0.629 | 0.335 | C | 0.216 |
| C | 44 | 0.924 | 0.244 | 0.405 | 0.452 | 0.325 | C | 0.047 |
| D | 42 | 0.894 | 0.217 | 0.429 | 0.350 | 0.479 | D | 0.051 |
| D | 43 | 0.896 | 0.217 | 0.420 | 0.335 | 0.466 | D | 0.046 |
| D | 44 | 0.892 | 0.226 | 0.436 | 0.325 | 0.479 | D | 0.042 |

## Per-state results

| State | State ID | Mean frame-0 SSIM | Min frame-0 SSIM | Mean own future score | Median margin |
|---|---:|---:|---:|---:|---:|
| A | 3/3 | 0.865 | 0.780 | 0.485 | 0.108 |
| B | 3/3 | 0.986 | 0.984 | 0.846 | 0.524 |
| C | 3/3 | 0.938 | 0.924 | 0.600 | 0.216 |
| D | 3/3 | 0.894 | 0.892 | 0.475 | 0.046 |

## Preparation and validity

Load the exact real O0 PNG; bicubic-resize in memory to 448x256 using the pinned M6A source_frame_rgb preprocessing; round to uint8 PIL RGB. Give VACE this image at frame 0 and uniform RGB(128,128,128) placeholders at frames 1..16.

17 full-frame PIL L masks at 448x256: frame 0 is black (0, conditioned), frames 1..16 white (255, generated). Verify the installed pipeline preprocesses them to 0 and 1 respectively and passes nonzero O0 conditioning latents.

No post-generation frame replacement or compositing. Save the pipeline output as lossless RGB H.264 MP4, then compute all official metrics from decoded MP4 frames.

The installed VACE pipeline's preprocessed frame-0 mask was exactly 0, future masks were exactly 1, and the O0 conditioning latent had nonzero norm in every generation. All saved videos decoded to exactly 17 frames. Only O0 and seed changed across runs.

Runtime: 259.09 seconds; peak observed VRAM: 14038 MiB.

## Artifacts

Protocol: `/root/autodl-tmp/GameDynamics_starter/data/experiments/m6a4_vace_hard_o0/eval/protocol.json`. Metrics: `/root/autodl-tmp/GameDynamics_starter/data/experiments/m6a4_vace_hard_o0/eval/metrics.json`. All 12 videos: `/root/autodl-tmp/GameDynamics_starter/data/experiments/m6a4_vace_hard_o0/eval/videos`. Contact sheet: `/root/autodl-tmp/GameDynamics_starter/data/experiments/m6a4_vace_hard_o0/eval/contact_sheet.png`.
Source selection: `/root/autodl-tmp/GameDynamics_starter/data/experiments/m6a3_four_state_1600/selected_states.json`. Preflight: `/root/autodl-tmp/GameDynamics_starter/data/experiments/m6a4_vace_hard_o0/preflight.json`.
