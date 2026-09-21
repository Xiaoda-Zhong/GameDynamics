# M6F-1: TRAIN-set action-contrastive gap audit

**Status:** complete. Frozen step-6400 model; no optimizer step, held-out evaluation, or model change.

## Deterministic paired protocol

The fixed 128-clip TRAIN subset contains **124 swappable** and **4 unchanged** clips. For each eligible clip, the prescribed LEFT/RIGHT swap was compared at three positions of the same 1,000-step M6D flow schedule: early index 100 (timestep 964, sigma 0.964324), middle index 500 (timestep 750, sigma 0.750375), late index 900 (timestep 251, sigma 0.251872).

Noise seed is `44 + 3 × zero-based subset index + region index` (early=0, middle=1, late=2). Within each pair, both forwards reuse the same real target latent, native VACE O0 control, generic text embedding, timestep, BF16 noise, and noisy latent; only integer action IDs differ. Loss is the existing MSE on future latent positions 1–4. The four unchanged clips were excluded from the contrastive comparison.

## Measured gap: delta = L_wrong − L_correct

| Region | Pairs | Delta > 0 | Mean delta | Median delta | Std delta | Mean correct loss | Mean wrong loss |
|---|---:|---:|---:|---:|---:|---:|---:|
| Global | 372 | 90.9% | 0.021510 | 0.010238 | 0.038003 | 0.139240 | 0.160750 |
| Early | 124 | 85.5% | 0.040076 | 0.024024 | 0.058294 | 0.166529 | 0.206605 |
| Middle | 124 | 96.8% | 0.018555 | 0.014487 | 0.016879 | 0.083921 | 0.102476 |
| Late | 124 | 90.3% | 0.005899 | 0.004177 | 0.007238 | 0.167270 | 0.173169 |

Standard deviation uses the population definition (`ddof=0`). A positive delta means the correct actions explain the real future better under this denoising loss.

| Region | p10 | p25 | p50 | p75 | p90 |
|---|---:|---:|---:|---:|---:|
| Global | 0.000107 | 0.003095 | 0.010238 | 0.024782 | 0.053049 |
| Early | -0.002342 | 0.007979 | 0.024024 | 0.052232 | 0.091337 |
| Middle | 0.002051 | 0.005383 | 0.014487 | 0.024782 | 0.042904 |
| Late | 0.000062 | 0.001362 | 0.004177 | 0.007646 | 0.014765 |

## Action-mix strata

These descriptive strata use one mean delta per clip across its three scheduler regions, so clips have equal weight.

### Swapped left/right actions

| Bin | Clips | Mean clip delta | Median clip delta | Positive clip means |
|---|---:|---:|---:|---:|
| 1-4 | 8 | 0.002612 | 0.002987 | 87.5% |
| 5-8 | 37 | 0.017806 | 0.011724 | 91.9% |
| 9-12 | 29 | 0.019424 | 0.017801 | 82.8% |
| 13-16 | 50 | 0.028484 | 0.025794 | 98.0% |

### NOOP fraction

| Bin | Clips | Mean clip delta | Median clip delta | Positive clip means |
|---|---:|---:|---:|---:|
| 0-25% | 90 | 0.022552 | 0.019989 | 92.2% |
| >25-50% | 19 | 0.018102 | 0.009057 | 84.2% |
| >50-75% | 14 | 0.020738 | 0.011513 | 100.0% |
| >75-100% | 1 | 0.003240 | 0.003240 | 100.0% |

## TRAIN-only recommendation

Recommend **margin = 0.010, lambda = 0.25** as conservative fixed starting values for a later contrastive training experiment. The margin is near the measured global median gap (0.010238) and activates the hinge in 49.5% of these TRAIN pairs (28.2% early, 40.3% middle, 79.8% late). The M6F-0 provisional margin 0.05 would activate 88.7%. At margin 0.01, the mean hinge value is 0.003635; lambda 0.25 contributes 0.000909 to the mean loss value. This loss-scale estimate does not guarantee a particular gradient balance. No held-out result was used to choose either number.

## Validity and limits

All 372 clip/region pairs and 744 forwards completed with finite losses. The frozen adapter and pretrained VACE weights remained unchanged; VAE/text weights were not loaded. Native O0/mask behavior was reused from the M6D cache. Peak allocated/reserved VRAM: 4365.5/4766.0 MiB; runtime: 111.1 s. This is a TRAIN-subset gap measurement with one fixed noise draw per clip/region; it does not estimate held-out action control.

Raw pairs: `data/experiments/m6f1_action_contrastive_gap_audit/raw_results.json`, `data/experiments/m6f1_action_contrastive_gap_audit/raw_results.csv`. Machine summary: `data/experiments/m6f1_action_contrastive_gap_audit/summary.json`. Runner: `training/wan/m6f1_action_gap_audit.py`.
