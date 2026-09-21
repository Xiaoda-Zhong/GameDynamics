# M6B-4: action-token adapter 32-step smoke gate

**Status: PASS. Ready for a separately authorized 800-step continuation. Training stopped at optimizer step 32.** No evaluation videos were generated.

The [training runner](../../../training/wan/vace_action_token_smoke.py) reused the M6B-3 16-token cross-attention adapter and its temporal mask, with hooks after VACE main blocks 3, 11, 19, and 27. The action and temporal embeddings, four cross-attention sites, and zero-initialized output projections were unchanged. The native VACE O0 control and existing latent-mask behavior were unchanged. The prompt was `A first-person gameplay view in a retro 3D stone maze.` The real 17-frame clip was the diffusion target, with MSE loss only on future latent positions 1–4.

| Gate check | Result |
|---|---:|
| Adapter trainable parameters | **3,684,864**, exact |
| Pretrained VACE trainable parameters | 0 |
| Pretrained VAE/text parameters trained | 0; cached real latents and fixed prompt embedding were reused |
| Zero-init output vs unhooked VACE | Bit-identical for ORIGINAL and ALL-NOOP before training |
| Dataset | All 16 cached samples/actions valid; each sampled twice |
| Optimizer | AdamW, LR 2e-4; batch 1; accumulation 1; seed 42 |
| Backbone and loss | BF16 frozen VACE; future-only latent MSE |
| Steps and numerical checks | 32; all losses and gradients finite; nonzero adapter gradient at every step; no OOM |
| Frozen VACE integrity | No gradients and no parameter version changes after step 32 |
| Loss range | 0.0609588–0.3977790 |
| Runtime | 14.811 seconds, including setup and zero-init check |
| Peak GPU memory | 4,621.3 MiB allocated; 5,030.0 MiB reserved |

At step 1, eight output-projection gradient tensors were nonzero; upstream projection/embedding gradients became nonzero after the zero-initialized output projections updated. Each of the four action hooks ran once during forward and once during gradient-checkpoint recomputation on every step. The [32-step log](steps.jsonl) records loss, LR, gradient norm, sample, timestep, runtime, and allocated/reserved VRAM per step. The [smoke gate](smoke_gate.json) records the final checks and counts. The [step-32 checkpoint](step_0032.pt) contains adapter and optimizer state plus RNG state; it was reloaded successfully with all 34 adapter parameter states. No pretrained weights are stored in it.

This gate establishes numerical and engineering viability only. Action control has not been evaluated for this adapter.
