# M6A: current-frame conditioning prototype

**Decision: FAIL as a reliable current-state anchor.** O0 control changes the generated image, but the first-frame identity, early-scene retention, and counterfactual-state criteria all miss their predeclared targets. Engineering validity passes. This milestone does not evaluate action control.

## Architecture and data

- Pinned Finetrainers commit `7c238443b7f102d3cfb4425caabd629e2ee1c02b` and Wan2.1-T2V-1.3B. Official control example: [train.sh](/tmp/finetrainers-wan-5c/examples/training/control/wan/image_condition/train.sh); inference pattern: [wan.md](/tmp/finetrainers-wan-5c/docs/models/wan.md); implementation: [control_specification.py](/tmp/finetrainers-wan-5c/finetrainers/models/wan/control_specification.py) and [control_trainer/trainer.py](/tmp/finetrainers-wan-5c/finetrainers/trainer/control_trainer/trainer.py).
- Source: [original 16-sample manifest](/root/autodl-tmp/GameDynamics_starter/data/wan_training/my_way_home/overfit_0016/manifest.jsonl) (SHA-256 `7d56952b72cc7c1075319ee1aa038c04b6b250913826f111f3bf63c9c8def991`). Each sample has 17 exact RGB observations and 16 action strings. The derived lossless MP4s decode pixel-identically to the source PNGs. Original manifest and PNGs were not modified. Source mapping: [source_map.jsonl](/root/autodl-tmp/outputs/wan/overfit_0016/current_frame_control_lora_0320/train_0320/derived_dataset/source_map.jsonl).
- Text: the existing fixed scene prefix plus each unchanged deterministic action caption. Control type `none` supplies the source video to the pinned trainer; `frame_conditioning_type=index`, index `0`, masks all latent frames except O0. A direct check on the first training clip found the VAE latent from O0 alone and the first latent from all 17 frames identical (max absolute difference `0.0`).
- Control LoRA: patch embedding expanded from 16 to 32 input channels with zeroed new base weights; masked O0 latent is concatenated to the noisy video latent channel axis. Rank/alpha `128/128` follows the pinned image-conditioning example. The patch-embedding injection LoRA uses rank `1536`. Targets: 240 transformer attention/feed-forward projections plus the patch embedding, 241 total. Trainable LoRA parameters: **96,927,744**. Original transformer weights, VAE, and text encoder were frozen.
- Training: BF16, 17 frames, 448×256, batch 1, accumulation 1, seed 42, gradient checkpointing, 320 optimizer steps; AdamW, recommended LR `2e-05`, constant schedule with 32-step warmup. An 8-step smoke completed before the full run. No packages were upgraded or changed.

## Training and engineering

| Measure | Value |
|---|---:|
| First / final step loss | 0.2687 / 0.2079 |
| Mean loss, steps 1–80 | 0.1870 |
| Mean loss, steps 81–160 | 0.1344 |
| Mean loss, steps 161–240 | 0.1339 |
| Mean loss, steps 241–320 | 0.1311 |
| Loss range | 0.0381–0.5810 |
| Gradient norm range | 0.1095–2.7305 |
| Training runtime / observed peak VRAM | 210.03 s / 12,354 MiB |
| Generation runtime / observed peak VRAM | 268.61 s / 11,496 MiB |

Per-step loss, gradients, LR, GPU allocation, and elapsed time: [steps.jsonl](/root/autodl-tmp/outputs/wan/overfit_0016/current_frame_control_lora_0320/train_0320/steps.jsonl). All 320 losses and gradient norms were finite; no NaN, Inf, or OOM occurred. The final step loss is higher than the last-window mean, so loss alone does not establish identity.

| Checkpoint | Adapter | Official loader validation |
|---:|---|---|
| 80 | [step 80](/root/autodl-tmp/outputs/wan/overfit_0016/current_frame_control_lora_0320/train_0320/lora_weights/000080) | 241 LoRA layers; injection B norm 0.248 |
| 160 | [step 160](/root/autodl-tmp/outputs/wan/overfit_0016/current_frame_control_lora_0320/train_0320/lora_weights/000160) | 241 LoRA layers; injection B norm 0.330 |
| 320 | [step 320](/root/autodl-tmp/outputs/wan/overfit_0016/current_frame_control_lora_0320/train_0320/lora_weights/000320) | 241 LoRA layers; injection B norm 0.488 |

Each checkpoint loaded through the pinned `finetrainers.patches.load_lora_weights()` implementation. The step-320 adapter alone was activated at scale 1.0. All 15 exported videos decode to exactly 17 frames. The control hook received a nonzero latent at temporal index 0 and zeros elsewhere; every O0 swap changed the first output frame.

## Fixed evaluation protocol

Five disjoint source pairs were selected by largest resized O0 RGB difference before generation. For each pair, the A prompt, seed 42, initial noise tensor, 17-frame 448×256 output, UniPC scheduler (flow shift 3.0), 30 inference steps, guidance 5.0, and step-320 LoRA were fixed. “Unconditioned” means this same Control LoRA with an all-zero control latent. The A and B counterfactual outputs changed only the O0 control. The pipeline transformer/text weights ran in BF16; the inference VAE ran in FP32, identically for every condition. SSIM is channel-averaged 11×11 Gaussian (σ=1.5) over valid pixels; RGB MAE uses 0–255 units. The reference is the exact source PNG resized with the pinned trainer’s bicubic preprocessing.

Full protocol and pair selection: [protocol.json](/root/autodl-tmp/outputs/wan/overfit_0016/current_frame_control_lora_0320/eval/protocol.json). Raw per-pair values: [metrics.json](/root/autodl-tmp/outputs/wan/overfit_0016/current_frame_control_lora_0320/eval/metrics.json). Generation metadata: [generation_summary.json](/root/autodl-tmp/outputs/wan/overfit_0016/current_frame_control_lora_0320/eval/generation_summary.json).

## Identity, retention, and counterfactual results

| Metric (five A clips) | Zero control | O0 control | Gain | Required | Result |
|---|---:|---:|---:|---|---|
| Mean first-frame SSIM | 0.252 | 0.379 | +0.127 | conditioned ≥0.75 and gain ≥0.20 | **FAIL** |
| Mean first-frame RGB MAE | 43.33 | 34.19 | -9.14 | descriptive | — |
| Mean SSIM, frames 0–3 | 0.263 | 0.358 | +0.095 | gain ≥0.10 | **FAIL** |
| Counterfactual O0 swaps | — | 1/5 | — | ≥4/5 pairs | **FAIL** |
| Engineering validity | — | finite loss/gradients, 3 loads, 15×17 frames, active control | — | all required | **PASS** |

| Pair | Source A | Source B | Frame-0 SSIM zero → O0 | Early SSIM zero → O0 | Cross-state SSIM: A→A / A→B / B→B / B→A | Pair pass |
|---:|---|---|---:|---:|---|---|
| 1 | `my_way_home_episode_0050_clip_000037` | `my_way_home_episode_0010_clip_000058` | 0.165 → 0.778 | 0.119 → 0.539 | 0.778 / 0.001 / 0.248 / 0.152 | yes |
| 2 | `my_way_home_episode_0039_clip_000002` | `my_way_home_episode_0041_clip_000019` | 0.347 → 0.326 | 0.395 → 0.442 | 0.326 / 0.120 / 0.274 / 0.479 | no |
| 3 | `my_way_home_episode_0028_clip_000011` | `my_way_home_episode_0030_clip_000048` | 0.274 → 0.352 | 0.252 → 0.297 | 0.352 / 0.225 / 0.293 / 0.317 | no |
| 4 | `my_way_home_episode_0038_clip_000028` | `my_way_home_episode_0000_clip_000038` | 0.224 → 0.154 | 0.260 → 0.183 | 0.154 / 0.177 / 0.261 / 0.248 | no |
| 5 | `my_way_home_episode_0000_clip_000008` | `my_way_home_episode_0004_clip_000022` | 0.247 → 0.284 | 0.290 → 0.331 | 0.284 / 0.370 / 0.138 / 0.054 | no |

The first pair improves strongly because its very dark O0 makes the conditioned output nearly black. The other pairs mostly retain the learned red-wall corridor and stylized human figure even when the true O0 is a gray wall or a different Doom room. Output changes are substantial, but the paired SSIM test shows they do not reliably follow the supplied state. See the source/zero/A/B sequence at frames 0, 1, and 3 in the [contact sheet](/root/autodl-tmp/outputs/wan/overfit_0016/current_frame_control_lora_0320/eval/contact_sheet.png).

## All videos

The 16 pixel-exact training source videos are mapped in `source_map.jsonl`; the five selected source A/B videos and all 15 generated evaluation videos are linked below.

| Pair | Source A | Source B | Zero control | O0 A | O0 B |
|---:|---|---|---|---|---|
| 1 | [source A](/root/autodl-tmp/outputs/wan/overfit_0016/current_frame_control_lora_0320/train_0320/derived_dataset/videos/0001.mp4) | [source B](/root/autodl-tmp/outputs/wan/overfit_0016/current_frame_control_lora_0320/train_0320/derived_dataset/videos/0010.mp4) | [video](/root/autodl-tmp/outputs/wan/overfit_0016/current_frame_control_lora_0320/eval/videos/pair_01_unconditioned.mp4) | [video](/root/autodl-tmp/outputs/wan/overfit_0016/current_frame_control_lora_0320/eval/videos/pair_01_conditioned_a.mp4) | [video](/root/autodl-tmp/outputs/wan/overfit_0016/current_frame_control_lora_0320/eval/videos/pair_01_conditioned_b.mp4) |
| 2 | [source A](/root/autodl-tmp/outputs/wan/overfit_0016/current_frame_control_lora_0320/train_0320/derived_dataset/videos/0013.mp4) | [source B](/root/autodl-tmp/outputs/wan/overfit_0016/current_frame_control_lora_0320/train_0320/derived_dataset/videos/0014.mp4) | [video](/root/autodl-tmp/outputs/wan/overfit_0016/current_frame_control_lora_0320/eval/videos/pair_02_unconditioned.mp4) | [video](/root/autodl-tmp/outputs/wan/overfit_0016/current_frame_control_lora_0320/eval/videos/pair_02_conditioned_a.mp4) | [video](/root/autodl-tmp/outputs/wan/overfit_0016/current_frame_control_lora_0320/eval/videos/pair_02_conditioned_b.mp4) |
| 3 | [source A](/root/autodl-tmp/outputs/wan/overfit_0016/current_frame_control_lora_0320/train_0320/derived_dataset/videos/0000.mp4) | [source B](/root/autodl-tmp/outputs/wan/overfit_0016/current_frame_control_lora_0320/train_0320/derived_dataset/videos/0007.mp4) | [video](/root/autodl-tmp/outputs/wan/overfit_0016/current_frame_control_lora_0320/eval/videos/pair_03_unconditioned.mp4) | [video](/root/autodl-tmp/outputs/wan/overfit_0016/current_frame_control_lora_0320/eval/videos/pair_03_conditioned_a.mp4) | [video](/root/autodl-tmp/outputs/wan/overfit_0016/current_frame_control_lora_0320/eval/videos/pair_03_conditioned_b.mp4) |
| 4 | [source A](/root/autodl-tmp/outputs/wan/overfit_0016/current_frame_control_lora_0320/train_0320/derived_dataset/videos/0006.mp4) | [source B](/root/autodl-tmp/outputs/wan/overfit_0016/current_frame_control_lora_0320/train_0320/derived_dataset/videos/0009.mp4) | [video](/root/autodl-tmp/outputs/wan/overfit_0016/current_frame_control_lora_0320/eval/videos/pair_04_unconditioned.mp4) | [video](/root/autodl-tmp/outputs/wan/overfit_0016/current_frame_control_lora_0320/eval/videos/pair_04_conditioned_a.mp4) | [video](/root/autodl-tmp/outputs/wan/overfit_0016/current_frame_control_lora_0320/eval/videos/pair_04_conditioned_b.mp4) |
| 5 | [source A](/root/autodl-tmp/outputs/wan/overfit_0016/current_frame_control_lora_0320/train_0320/derived_dataset/videos/0011.mp4) | [source B](/root/autodl-tmp/outputs/wan/overfit_0016/current_frame_control_lora_0320/train_0320/derived_dataset/videos/0015.mp4) | [video](/root/autodl-tmp/outputs/wan/overfit_0016/current_frame_control_lora_0320/eval/videos/pair_05_unconditioned.mp4) | [video](/root/autodl-tmp/outputs/wan/overfit_0016/current_frame_control_lora_0320/eval/videos/pair_05_conditioned_a.mp4) | [video](/root/autodl-tmp/outputs/wan/overfit_0016/current_frame_control_lora_0320/eval/videos/pair_05_conditioned_b.mp4) |

## Code and scope

Experiment code: [training runner](/root/autodl-tmp/GameDynamics_starter/training/wan/finetrainers_current_frame_train.py), [evaluation runner](/root/autodl-tmp/GameDynamics_starter/training/wan/finetrainers_current_frame_eval.py), [metrics](/root/autodl-tmp/GameDynamics_starter/training/wan/current_frame_metrics.py), and [lossless dataset derivation](/root/autodl-tmp/GameDynamics_starter/training/wan/finetrainers_data.py). No action embedding, formal 500-clip training, or action-turn evaluation was performed.

