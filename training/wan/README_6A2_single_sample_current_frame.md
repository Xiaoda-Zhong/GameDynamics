# M6A-2: single-sample current-state overfit diagnostic

**Final classification: C. trained-state FAIL.** The step-800 output visually approaches the red-wall training state, but its decoded frame-0 SSIM is below the predeclared 0.75 target. All five counterfactual B states remain closer in SSIM to training state A than to the supplied B state.

## Decision lock

The **predeclared decoded-MP4 frame metrics are the official result**. Step 800 reaches SSIM 0.725974 versus the fixed 0.75 threshold, while its gain over untrained control is 0.461276 versus the 0.30 threshold and its RGB MAE is 37.15% of baseline versus the 50% threshold. The required conjunction fails. The counterfactual criterion also fails at 0/5 versus the fixed 4/5 threshold. Engineering validity passes. The final M6A-2 classification is **C. trained-state FAIL**.

The post-hoc raw pre-encoding frame check is **sensitivity analysis only**; it cannot replace decoded-MP4 measurements or change the predeclared thresholds. Its preserved [log](/root/autodl-tmp/outputs/wan/overfit_0016/current_frame_single_sample_0800/raw_frame_check.log) ends with an `AttributeError` before any raw-frame metric is recorded, so there is no valid raw-frame value to report. No original M6A-2 protocol, metrics, videos, checkpoints, or source manifests were modified for this closure.

## Protocol and architecture

- Only training sample: `my_way_home_episode_0028_clip_000011`. Source: [pixel-exact 17-frame video](/root/autodl-tmp/outputs/wan/overfit_0016/current_frame_single_sample_0800/train_0800/derived_dataset/videos/0000.mp4); O0: [exact source PNG](/root/autodl-tmp/GameDynamics_starter/data/episodes/my_way_home/episode_0028/rgb/000088.png). The derived video was verified pixel-for-pixel against all 17 source PNGs; the original [manifest](/root/autodl-tmp/GameDynamics_starter/data/wan_training/my_way_home/overfit_0016/manifest.jsonl) stayed unchanged (SHA-256 `7d56952b72cc7c1075319ee1aa038c04b6b250913826f111f3bf63c9c8def991`). The 16 original action strings and scene+action prompt were preserved.
- Pinned Finetrainers commit `7c238443b7f102d3cfb4425caabd629e2ee1c02b` with Wan2.1-T2V-1.3B. Official Control LoRA: expanded patch embedding 16→32 channels, O0 VAE latent retained at temporal index 0, then concatenated to the noisy latent. Rank/alpha 128/128; injection rank 1536; 241 LoRA targets and 96,927,744 trainable parameters. VAE, text encoder, and original transformer weights frozen.
- Training: BF16, 17 frames, 448×256, batch 1, gradient accumulation 1, seed 42, gradient checkpointing, AdamW LR 2e-05, constant schedule with 80-step warmup, 800 optimizer steps. No dependency changes.
- Evaluation: exact same A prompt, seed 42, initial noise, 17 frames, 448×256, 30 UniPC inference steps (flow shift 3.0), guidance 5.0, and scale 1.0. Model/text BF16 and inference VAE FP32 for every condition, matching the M6A evaluation path. “Untrained control” uses the real O0 and active control hook with all trained adapters disabled; added base patch channels are zero initialized. Five B states were selected before generation by deterministic max-min O0 RGB distance from the existing 16-clip M6A set. Only O0 changes in the B runs.
- Evaluation protocol and fixed B selection: [protocol.json](/root/autodl-tmp/outputs/wan/overfit_0016/current_frame_single_sample_0800/eval/protocol.json). SSIM uses the same channel-averaged 11×11 Gaussian, valid pixels, as M6A; RGB MAE is in 0–255 units. Metrics use saved video frames decoded from MP4.

## Training and engineering

| Measure | Value |
|---|---:|
| First / final step loss | 0.1647 / 0.0635 |
| Mean loss, steps 1–400 / 401–800 | 0.1076 / 0.0754 |
| Mean loss, steps 1–100 | 0.1406 |
| Mean loss, steps 101–200 | 0.1061 |
| Mean loss, steps 201–300 | 0.0950 |
| Mean loss, steps 301–400 | 0.0888 |
| Mean loss, steps 401–500 | 0.0913 |
| Mean loss, steps 501–600 | 0.0762 |
| Mean loss, steps 601–700 | 0.0723 |
| Mean loss, steps 701–800 | 0.0617 |
| Gradient norm range | 0.0887–3.6485 |
| Training runtime / observed peak VRAM | 484.96 s / 11,378 MiB |
| Evaluation runtime / observed peak VRAM | 200.58 s / 11,498 MiB |

All 800 loss and gradient records are finite; no NaN, Inf, or OOM occurred. Per-step details: [steps.jsonl](/root/autodl-tmp/outputs/wan/overfit_0016/current_frame_single_sample_0800/train_0800/steps.jsonl). Four adapter-only checkpoints loaded into all 241 LoRA layers through the pinned official loader:

| Step | Checkpoint file | Injection B norm |
|---:|---|---:|
| 100 | [adapter](/root/autodl-tmp/outputs/wan/overfit_0016/current_frame_single_sample_0800/train_0800/lora_weights/000100/pytorch_lora_weights.safetensors) | 0.221 |
| 200 | [adapter](/root/autodl-tmp/outputs/wan/overfit_0016/current_frame_single_sample_0800/train_0800/lora_weights/000200/pytorch_lora_weights.safetensors) | 0.356 |
| 400 | [adapter](/root/autodl-tmp/outputs/wan/overfit_0016/current_frame_single_sample_0800/train_0800/lora_weights/000400/pytorch_lora_weights.safetensors) | 0.565 |
| 800 | [adapter](/root/autodl-tmp/outputs/wan/overfit_0016/current_frame_single_sample_0800/train_0800/lora_weights/000800/pytorch_lora_weights.safetensors) | 0.978 |

All ten saved videos decode to exactly 17 frames. The control hook received a nonzero O0 latent at temporal index 0 and zeros elsewhere. The untrained baseline had adapters disabled; each B swap changed the output first frame. **Engineering validity: PASS.**

## Training-state trajectory

| Condition | Frame-0 SSIM vs A | Frame-0 RGB MAE vs A | Mean SSIM, frames 0–3 | Video |
|---|---:|---:|---:|---|
| untrained_control | 0.265 | 44.17 | 0.296 | [video](/root/autodl-tmp/outputs/wan/overfit_0016/current_frame_single_sample_0800/eval/videos/untrained_control.mp4) |
| step_0100 | 0.196 | 41.87 | 0.209 | [video](/root/autodl-tmp/outputs/wan/overfit_0016/current_frame_single_sample_0800/eval/videos/step_0100.mp4) |
| step_0200 | 0.223 | 35.63 | 0.233 | [video](/root/autodl-tmp/outputs/wan/overfit_0016/current_frame_single_sample_0800/eval/videos/step_0200.mp4) |
| step_0400 | 0.576 | 14.11 | 0.521 | [video](/root/autodl-tmp/outputs/wan/overfit_0016/current_frame_single_sample_0800/eval/videos/step_0400.mp4) |
| step_0800 | 0.726 | 16.41 | 0.484 | [video](/root/autodl-tmp/outputs/wan/overfit_0016/current_frame_single_sample_0800/eval/videos/step_0800.mp4) |

Step 800 gains **+0.461 SSIM** over untrained control (target ≥0.30) and reduces MAE to **37.1%** of baseline (target ≤50%). Its frame-0 SSIM is **0.726**, below the required **0.75**. The three-part primary trained-state criterion therefore **FAILS**. The best early-frame mean SSIM is 0.521 at step 400.

## Counterfactual O0 results at step 800

The A prompt, model, adapter, seed, noise, and inference settings are fixed. The only changed input is the supplied real O0_B. Success requires SSIM(output_B, B) > SSIM(output_B, A).

| B state | Real O0_B | B vs A source SSIM | Output to B SSIM | Output to training A SSIM | Success | Video |
|---|---|---:|---:|---:|---|---|
| `my_way_home_episode_0010_clip_000058` | [PNG](/root/autodl-tmp/GameDynamics_starter/data/episodes/my_way_home/episode_0010/rgb/000464.png) | 0.277 | 0.111 | 0.477 | no | [video](/root/autodl-tmp/outputs/wan/overfit_0016/current_frame_single_sample_0800/eval/videos/counter_01.mp4) |
| `my_way_home_episode_0005_clip_000048` | [PNG](/root/autodl-tmp/GameDynamics_starter/data/episodes/my_way_home/episode_0005/rgb/000384.png) | 0.235 | 0.280 | 0.416 | no | [video](/root/autodl-tmp/outputs/wan/overfit_0016/current_frame_single_sample_0800/eval/videos/counter_02.mp4) |
| `my_way_home_episode_0050_clip_000037` | [PNG](/root/autodl-tmp/GameDynamics_starter/data/episodes/my_way_home/episode_0050/rgb/000296.png) | 0.368 | 0.203 | 0.344 | no | [video](/root/autodl-tmp/outputs/wan/overfit_0016/current_frame_single_sample_0800/eval/videos/counter_03.mp4) |
| `my_way_home_episode_0038_clip_000028` | [PNG](/root/autodl-tmp/GameDynamics_starter/data/episodes/my_way_home/episode_0038/rgb/000224.png) | 0.186 | 0.247 | 0.277 | no | [video](/root/autodl-tmp/outputs/wan/overfit_0016/current_frame_single_sample_0800/eval/videos/counter_04.mp4) |
| `my_way_home_episode_0005_clip_000053` | [PNG](/root/autodl-tmp/GameDynamics_starter/data/episodes/my_way_home/episode_0005/rgb/000424.png) | 0.252 | 0.175 | 0.314 | no | [video](/root/autodl-tmp/outputs/wan/overfit_0016/current_frame_single_sample_0800/eval/videos/counter_05.mp4) |

**Counterfactual: 0/5; FAIL** (target ≥4/5). Every B output changes visibly, but each remains closer by SSIM to training A than to the supplied B. The [contact sheet](/root/autodl-tmp/outputs/wan/overfit_0016/current_frame_single_sample_0800/eval/contact_sheet.png) shows the five-stage A trajectory at frames 0, 1, and 3 plus each B source/output pair. The step-800 A output largely removes the earlier human figure and captures the red-wall corridor, while geometry remains imperfect and the state swap does not generalize.

## Artifacts and code

- Machine-readable [metrics.json](/root/autodl-tmp/outputs/wan/overfit_0016/current_frame_single_sample_0800/eval/metrics.json), [generation_summary.json](/root/autodl-tmp/outputs/wan/overfit_0016/current_frame_single_sample_0800/eval/generation_summary.json), and [training summary](/root/autodl-tmp/outputs/wan/overfit_0016/current_frame_single_sample_0800/train_0800/summary.json).
- Implementation: [M6A training runner with single-sample option](/root/autodl-tmp/GameDynamics_starter/training/wan/finetrainers_current_frame_train.py), [M6A-2 evaluation runner](/root/autodl-tmp/GameDynamics_starter/training/wan/finetrainers_current_frame_single_eval.py), reused [M6A evaluation helpers](/root/autodl-tmp/GameDynamics_starter/training/wan/finetrainers_current_frame_eval.py) and [metrics](/root/autodl-tmp/GameDynamics_starter/training/wan/current_frame_metrics.py).
- No explicit action conditioning, action-turn evaluation, package upgrade, or 500-clip experiment was performed.

