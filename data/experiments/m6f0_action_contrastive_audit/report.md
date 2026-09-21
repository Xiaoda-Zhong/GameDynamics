# M6F-0: action-contrastive objective feasibility audit

**Status: engineering-feasible on one paired sample.** No optimizer step or evaluation run.

Real TRAIN sample: `my_way_home_episode_0016_clip_000036` (subset index 0, episode_0016). The deterministic left/right swap changes 10/16 action IDs at positions [6, 7, 8, 9, 10, 11, 12, 13, 14, 15].

## Paired objective

Both forwards reuse the exact same cached real target, native VACE O0 control, generic text embedding, diffusion timestep, BF16 noise tensor, and noisy latent. Only the 16 integer action IDs differ. Loss is the existing M6D MSE on future temporal latent positions 1–4.

- `L_correct = 0.13071337`; `L_wrong = 0.13137129`; `L_wrong - L_correct = 0.00065792`.
- Provisional `margin = 0.05` and `lambda = 1.0`: `L_rank = 0.04934208`; `L_total = 0.18005546`. These values are for gradient sanity only and were not chosen from held-out evaluation.
- Ranking hinge active: **True**. Adapter gradient norms: correct 0.231840, wrong 0.266290, contrastive difference 0.244362, total 0.394987. All adapter gradients were finite; total gradient was nonzero. The total gradient was assembled algebraically from separate correct/wrong backward passes with unchanged weights.

The existing adapter prefers the correct actions by only 0.00065792 MSE on this sample/timestep. This verifies that the objective can carry a signal; it does not establish robust action semantics or held-out generalization.

## Engineering validity

Timestep index 588 (timestep 678, sigma 0.678161); seed-44 noise. The pretrained VACE transformer was frozen and unchanged. VAE and text encoder were not loaded; their existing cached outputs were reused. Native O0 and mask behavior, including the previously observed all-one latent mask, stayed unchanged. No optimizer was constructed and adapter weights were unchanged.

Peak allocated/reserved VRAM: 4593.2/5012.0 MiB; paired audit runtime 1.441 s.

## Counterfactual eligibility

4 of 128 fixed TRAIN clips are unchanged by the prescribed swap. For those clips, `A_wrong == A_correct`, so the ranking term is the constant margin and its gradient is zero. A future objective should skip the contrastive term for those clips while retaining the standard denoising loss; do not invent a different action sequence for them. This audit made no training change.

Numerical record: `/root/autodl-tmp/GameDynamics_starter/data/experiments/m6f0_action_contrastive_audit/audit.json`. Frozen adapter: `/root/autodl-tmp/GameDynamics_starter/data/experiments/m6d_128clip_action_scaleup/checkpoints/step_6400/adapter.pt`.
