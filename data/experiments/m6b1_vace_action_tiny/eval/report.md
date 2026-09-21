# M6B-1: explicit VACE action adapter tiny pilot

**Classification: C. EXPLICIT ACTION CONTROL FAIL**

The official M6A-4 classification remains C under its predeclared O0 hard-anchor rule. M6B-1 treats O0 as observed conditioning and evaluates future dynamics only.

## Fixed flow protocol

The M5C-5 OpenCV Farneback implementation and ROI are reused unchanged. The real source clip calibrates flow sign from its two left and six right turns. Primary M6B-1 scoring uses only the 15 generated-future pairs O1→O2 through O15→O16, aligned to A1…A15. O0→O1 is retained as descriptive flow and is excluded from every criterion. Consequently the first flip segment is A1 alone and the final segment is A10…A15. The NOOP ratio is computed within each mixed-action video and the median of 10 ratios is primary, as in M5C-5.

Source sign: positive raw u means left-turn camera motion; negative means right-turn; source left mean raw u=+3.753, right mean raw u=-3.242 px.

## Primary decision

| Criterion | Required | Observed | Result |
|---|---:|---:|---|
| Turn direction | ≥0.70 | 38/70 = 0.543 | FAIL |
| Median R_noop | ≤0.50 | 0.668 | FAIL |
| Median abs(r) | ≥0.50 | 0.382 | FAIL |
| Matched-seed flip | ≥4/5 | 0/5 | FAIL |

## Per-video future-only flow

| Condition | Seed | Turns correct | R_noop | Signed r | abs(r) | Matched all-NOOP ratio |
|---|---:|---:|---:|---:|---:|---:|
| original | 42 | 3/7 | 0.804 | 0.657 | 0.657 | — |
| original | 43 | 2/7 | 0.805 | 0.563 | 0.563 | — |
| original | 44 | 3/7 | 0.544 | 0.253 | 0.253 | — |
| original | 45 | 3/7 | 0.651 | 0.423 | 0.423 | — |
| original | 46 | 3/7 | 0.955 | 0.635 | 0.635 | — |
| reversed | 42 | 4/7 | 0.752 | -0.057 | 0.057 | — |
| reversed | 43 | 6/7 | 0.686 | 0.191 | 0.191 | — |
| reversed | 44 | 5/7 | 0.457 | 0.340 | 0.340 | — |
| reversed | 45 | 5/7 | 0.386 | 0.573 | 0.573 | — |
| reversed | 46 | 4/7 | 0.527 | 0.146 | 0.146 | — |
| all_noop | 42 | — | — | — | — | 0.885 |
| all_noop | 43 | — | — | — | — | 0.787 |
| all_noop | 44 | — | — | — | — | 0.565 |
| all_noop | 45 | — | — | — | — | 0.749 |
| all_noop | 46 | — | — | — | — | 0.857 |

## Counterfactual segments

| Seed | ORIGINAL A1 | REVERSED A1 | ORIGINAL A10…A15 | REVERSED A10…A15 | Flip |
|---:|---:|---:|---:|---:|---|
| 42 | 1.245 | 0.063 | 0.033 | 0.148 | No |
| 43 | 2.925 | 0.226 | 0.519 | 0.430 | No |
| 44 | 1.597 | 0.122 | 0.587 | 0.356 | No |
| 45 | 1.623 | 0.133 | 0.424 | 0.950 | No |
| 46 | 2.712 | 0.024 | 0.034 | 0.285 | No |

## Supporting future-frame similarity

ORIGINAL runs mean O1…O16 SSIM to the real source: 0.401; descriptive only, no threshold. Decoded O0 fidelity is not evaluated as a success criterion.

## Training and engineering validity

The existing 16 real TRAIN clips contain 256 transitions; fixed six-action counts (ID:transitions) are 0:46, 1:43, 2:22, 3:62, 4:32, 5:51. All IDs are present. The prompt was the same generic scene sentence for every sample, with no action text.

The pretraining zero-init check passed: ORIGINAL and ALL-NOOP actions produced pixel-identical 17-frame pipeline outputs at the same O0 and seed before video encoding (maximum difference 0). The action hook executed 60 times in each 30-step CFG generation. The saved check was reused; its videos were not regenerated.

Only `Embedding(6,32)` and `Linear(128,1536)` were trainable: **198,336 parameters**. The VACE transformer, both patch embeddings, all VACE blocks, VAE, and text encoder were frozen. The optimizer contained exactly the adapter parameters. The transformer parameter version counters were unchanged at steps 32 and 800, and it had no gradients. Native VACE pixel and latent-mask preparation was kept as installed, including the M6B-0 observed all-white compressed latent mask; no mask fix was made.

The corrected 32-step smoke gate passed: finite loss and gradients, nonzero adapter gradient norm, no NaN/Inf or OOM, each clip sampled twice, and pretrained transformer weights unchanged. Training then reached **800 optimizer steps** with AdamW, LR 1e-3, batch/accumulation 1/1, seed 42, BF16 VACE backbone with its protected FP32 components, and gradient checkpointing. Every clip appeared exactly 50 times; all 800 logged losses and adapter gradient norms were finite, with a nonzero adapter gradient norm each step. Adapter-only checkpoints at steps 100, 200, 400, and 800 all loaded strictly with the expected three state-dict keys. No pretrained weight is in those checkpoints.

Training runtime was **446.91 s**, with peak observed VRAM **5,444 MiB**. The 15 matched-seed videos took **326.47 s** and reached **14,040 MiB** observed VRAM. Each MP4 decodes to 17 frames at 448×256. Every sidecar confirms the same prompt, checkpoint, inference steps, O0 mask, nonzero O0 conditioning latent, active action hook, and no generated-frame replacement. Only action IDs and seed varied.

An initial 63-step implementation attempt was stopped after Diffusers warned that a device move had recast protected FP32 VACE components. Its logs and provisional gate are preserved under `train/aborted_bf16_cast/` and are excluded from these results. The official run restarted at step 1 after preserving the model's loaded dtypes. The adapter design, LR, data, metrics, and thresholds were unchanged.

## Interpretation

Loss on the tiny training set decreased over much of the pilot, but the future-only flow test does not show reliable action control. In all five matched seeds, the first-turn horizontal flow had the same sign under ORIGINAL and REVERSED, and none passed the combined first/final flip rule. The official decision is determined by the four criteria above.

## Artifacts

Metrics: `/root/autodl-tmp/GameDynamics_starter/data/experiments/m6b1_vace_action_tiny/eval/metrics.json`. Flow transitions: `/root/autodl-tmp/GameDynamics_starter/data/experiments/m6b1_vace_action_tiny/eval/transitions.jsonl`. Contact sheet: `/root/autodl-tmp/GameDynamics_starter/data/experiments/m6b1_vace_action_tiny/eval/contact_sheet.png`. The metrics JSON lists all 15 video paths and SHA-256 hashes. Training logs: `train/steps.jsonl` and `train/summary.json`; smoke gate: `train/smoke_gate.json`; checkpoints: `checkpoints/step_0100`, `step_0200`, `step_0400`, `step_0800`; generation details: `eval/generation_summary.json`; saved zero-init result: `zero_init/sanity.json`; protocol: `protocol.json`.
