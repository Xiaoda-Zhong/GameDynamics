# M6B-4: cross-attention action-token pilot

**Classification: A. CROSS-ATTENTION ACTION CONTROL PASS** (4/4 primary criteria passed).

The official run resumed exactly from the validated step-32 checkpoint and ended at step 800. O0 remained native VACE conditioning; all evaluation criteria use generated future transitions only. The action-token architecture, LR, data, text, mask behavior, and frozen pretrained backbone were unchanged.

## Primary decision

| Criterion | Threshold | Observed | Result |
|---|---:|---:|---|
| Turn direction accuracy | ≥0.70 | 66/70 = 0.943 | PASS |
| Median NOOP motion ratio | ≤0.50 | 0.353 | PASS |
| Median absolute action-motion correlation | ≥0.50 | 0.797 | PASS |
| Matched-seed LEFT/RIGHT flip | ≥4/5 | 5/5 | PASS |

The primary flow procedure is exactly `training.wan.vace_action_eval.analyze`, which reuses the validated M5C-5 Farneback ROI, source sign calibration, and O1→O2 through O15→O16 action alignment. O0→O1 is descriptive only. No criterion was changed after observing outputs.

## Seed-42 checkpoint trajectory

| Step | Turn accuracy | Mean abs(r), two runs | Expected first/final flip |
|---:|---:|---:|---|
| 100 | 0.929 | 0.539 | No |
| 200 | 0.929 | 0.779 | Yes |
| 400 | 0.786 | 0.575 | No |
| 800 | 1.000 | 0.824 | Yes |

## Run validity

The step-32 checkpoint contained adapter weights, all 34 AdamW parameter states, CPU/CUDA and noise-generator RNG states, and data-order generator state. The fixed 1,000-timestep flow schedule has no per-step training state. Reconstructed sample and timestep choices matched all 32 smoke-log entries, so training resumed at step 33. All 16 samples appeared 50 times over 800 steps. Adapter-only checkpoints at 100, 200, 400, and 800 were reloaded successfully; all pretrained VACE parameters remained frozen and unchanged. The full log records finite loss and adapter gradients, LR, per-block gradient norms from step 33 onward, runtime, and GPU memory.

Continuation runtime was 299.961 seconds, with peak allocated/reserved GPU memory of 4,625.1/5,010.0 MiB. Final and trajectory video generation took 430.616 seconds and observed 14,046 MiB peak GPU memory. All 15 final outputs and eight trajectory paths decode to exactly 17 frames. Every final output's sidecar verifies black O0 and white future masks, active O0 conditioning, and action-hook calls; no first frame was replaced after generation. The step-800 trajectory paths reuse the identical seed-42 final videos.

## Artifacts

Training summary: `/root/autodl-tmp/GameDynamics_starter/data/experiments/m6b4_action_token_800/training_summary.json`; full 800-step log: `/root/autodl-tmp/GameDynamics_starter/data/experiments/m6b4_action_token_800/steps.jsonl`. Final evaluation: `/root/autodl-tmp/GameDynamics_starter/data/experiments/m6b4_action_token_800/eval/metrics.json`; 15 videos: `/root/autodl-tmp/GameDynamics_starter/data/experiments/m6b4_action_token_800/eval/videos`. Trajectory: `/root/autodl-tmp/GameDynamics_starter/data/experiments/m6b4_action_token_800/eval/checkpoint_trajectory/metrics.json` and `/root/autodl-tmp/GameDynamics_starter/data/experiments/m6b4_action_token_800/eval/checkpoint_trajectory/videos`. Contact sheet: `/root/autodl-tmp/GameDynamics_starter/data/experiments/m6b4_action_token_800/eval/contact_sheet.png`. Metrics JSON lists all final video paths and hashes.
