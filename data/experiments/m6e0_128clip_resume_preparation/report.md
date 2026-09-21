# M6E-0: step-2000 continuation readiness

**Status:** exact resume is possible and prepared. No optimizer step or evaluation was run for M6E-0.

## M6D training-log audit

Source: `data/experiments/m6d_128clip_action_scaleup/train/steps.jsonl`. All 2,000 steps are present in order, with finite losses and gradient norms.

| Optimizer steps | Mean loss | Median loss | Final loss | Mean adapter grad norm |
|---|---:|---:|---:|---:|
| 1-500 | 0.162485 | 0.131792 | 0.064267 | 0.561683 |
| 501-1000 | 0.152774 | 0.122667 | 0.346819 | 0.364936 |
| 1001-1500 | 0.143500 | 0.114552 | 0.116706 | 0.330758 |
| 1501-2000 | 0.141876 | 0.111366 | 0.183742 | 0.330060 |

The broad 500-step mean loss declines from 0.162485 to 0.141876. **Near step 2000 it is not still trending downward:** the mean rose from 0.134261 in steps 1501–1750 to 0.149491 in steps 1751–2000, and the fitted slope over the last 500 steps is +0.00005011 loss/step. Individual final-step losses are noisy.

## Exact-resume state

- The [step-2000 adapter](../m6d_128clip_action_scaleup/checkpoints/step_2000/adapter.pt) has 3,684,864 parameters and matches its saved SHA-256 sidecar. The [training state](../m6d_128clip_action_scaleup/checkpoints/step_2000/training_state.pt) records current optimizer step 2000, all 34 AdamW parameter states at step 2000, LR 2e-4, CPU and CUDA RNG states, and the CUDA diffusion-noise generator state.
- The scheduler is a fixed 1,000-timestep UniPC flow schedule. It has no mutable per-training-step state to restore; the reconstructed sigma matches every logged M6D sigma.
- Data order is reconstructed from seed 42 and timestep order from seed 43. Both reconstructed streams match all 2,000 log records. The next sample index is 27; the next timestep index is 422. The diffusion-noise stream resumes from its saved generator state.
- The existing M6D runner starts from step 1 and refuses an occupied log. The new [continuation runner](../../../training/wan/m6e_resume_128.py) adds only the missing restore path and read-only validation. Its default mode audits; training requires the explicit `--execute` flag.

## Prepared continuation

Use the same checksum-verified 128-clip subset and cached native VACE O0 conditioning; unchanged M6B-4 action-token architecture and temporal mask; generic prompt; future-only latent loss; frozen pretrained VACE/VAE/text weights; AdamW at LR 2e-4; BF16 backbone; batch 1; gradient accumulation 1. Resume from optimizer step 2000 through step 6400, saving adapter-only checkpoints at **3200, 4800, 6400**. The existing VACE mask behavior is unchanged.

When continuation is authorized, run from the repository root:

```bash
python -m training.wan.m6e_resume_128 --execute
```

Read-only readiness check:

```bash
python -m training.wan.m6e_resume_128
```

The continuation command was **not** executed during M6E-0. No model weights, existing configuration, source manifests, or data were changed.
