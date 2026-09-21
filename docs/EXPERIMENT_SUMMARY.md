# Experiment Summary

GameDynamics studies explicit action control for short ViZDoom video prediction:

```text
observed O0 + integer actions A0...A15 -> predicted O1...O16
```

All conclusions below use predeclared metrics and fixed checkpoints. The final showcase artifact is the M6F-2 step-6400 action-token cross-attention adapter.

## Progression

| Stage | Question | Result |
|---|---|---|
| T2V Control LoRA | Can an expanded patch input preserve the current frame across states? | Multi-state O0 conditioning failed. This path was closed. |
| VACE first-frame smoke | Does native masked conditioning retain the observed state? | Early-future state ID was 12/12, but decoded O0 missed the strict hard-anchor threshold. O0 became an observed condition rather than a prediction target. |
| Additive action adapter | Can a grouped temporal bias control gameplay motion? | All four action-dynamics criteria failed. An internal-effect audit showed strong, distinct action signals, ruling out weak amplitude and global feature collapse. |
| Action-token cross-attention | Does preserving 16 ordered action tokens improve control? | Four masked adapters at blocks 3/11/19/27 passed all four criteria on the 16-clip pilot. |
| Held-out generalization | Does tiny-set control transfer to unseen states? | Partial: direction and correlation passed; NOOP suppression and matched flips failed. |
| 128-clip scale-up | Does added TRAIN diversity close the gap? | Partial generalization at both 2,000 and 6,400 steps. More budget improved some metrics but did not pass all four. |
| Contrastive objective | Does correct-versus-swapped ranking improve action discrimination? | Direction accuracy and correlation improved over the standard-objective baseline. NOOP and matched flips still failed. |
| Flip decomposition and lag audit | Are remaining failures concentrated or merely delayed? | The official first-turn flip was 7/12. None of its five failures recovered within O2→O5, ruling out simple short temporal lag. |

## Final Model

- Frozen Wan2.1-VACE-1.3B backbone with native O0 conditioning
- Six-action integer vocabulary
- 16 separate 256-dimensional action tokens
- Learned action and temporal embeddings
- Temporally masked cross-attention at transformer blocks 3, 11, 19, and 27
- 3,684,864 trainable adapter parameters
- 128 TRAIN clips, 6,400 steps, AdamW at 2e-4
- Future-only denoising MSE plus TRAIN-selected swapped-action ranking loss
- Official checkpoint fixed at step 6400

## Final Held-Out Results

| Metric | M6E standard objective | M6F-2 contrastive | Threshold | Final outcome |
|---|---:|---:|---:|---|
| Turn direction accuracy | 0.857 | **0.887** | ≥ 0.70 | PASS |
| Median absolute action-motion correlation | 0.718 | **0.786** | ≥ 0.50 | PASS |
| Median NOOP motion ratio | 0.619 | **0.572** | ≤ 0.50 | FAIL |
| Matched-seed LEFT/RIGHT flip | 8/12 | **7/12** | ≥ 9/12 | FAIL |

The contrastive objective strengthened aggregate direction and correlation control and reduced the NOOP ratio, but it did not improve counterfactual flip consistency. The result is evidence of partial held-out action control, not a mature or general-purpose world model.

## Reviewer Entry Points

- Final adapter: [`step_6400/adapter.pt`](../data/experiments/m6f2_action_contrastive_128_6400/checkpoints/step_6400/adapter.pt)
- Training protocol: [`train/protocol.json`](../data/experiments/m6f2_action_contrastive_128_6400/train/protocol.json)
- Final metrics: [`eval/metrics.json`](../data/experiments/m6f2_action_contrastive_128_6400/eval/metrics.json)
- Final report: [`eval/report.md`](../data/experiments/m6f2_action_contrastive_128_6400/eval/report.md)
- Lag audit: [`m6g0b report`](../data/experiments/m6g0b_first_turn_temporal_lag/report.md)
- Demo: [`heldout_state_D_seed_43.mp4`](../demo/heldout_state_D_seed_43.mp4)

Earlier scripts under `training/wan/` are retained as historical experiment and debug entry points. They are not the recommended final workflow.
