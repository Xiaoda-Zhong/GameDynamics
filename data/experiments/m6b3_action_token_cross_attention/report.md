# M6B-3: action-token cross-attention architecture audit

**Status:** architecture and forward-only shape audit complete. No training or generation was run. M6B-1 remains **C. EXPLICIT ACTION CONTROL FAIL**; M6B-2 remains **B. ACTION SIGNAL IS STRONG BUT SEMANTIC CONTROL FAILS**. This is a proposed replacement for the additive adapter, not a change to that adapter.

## Fixed context and measured shapes

Use Wan2.1-VACE-1.3B-diffusers, the native VACE O0 conditioning, the existing VACE latent-mask behavior, six integer action IDs, and the generic prompt `A first-person gameplay view in a retro 3D stone maze.` All pretrained transformer, VACE branch, patch-embedding, VAE, and text-encoder weights stay frozen. The familiar cached 17-frame sample has video target `[B,16,5,32,56]`, native O0 control `[B,96,5,32,56]`, and text `[B,512,4096]` for `B=1`. VACE has 30 main transformer blocks and 15 VACE hint blocks at even main-block indices 0–28. The real forward hook observed a video hidden tensor `[1,2240,1536]` at each proposed site.

The 17 RGB frames `[B,17,3,256,448]` become five VAE temporal latent positions. Wan's spatial patch embedding makes `16×28=448` tokens per latent position, ordered in time: latent 0 corresponds to observed O0, then latent positions 1–4 correspond to O1–O4, O5–O8, O9–O12, and O13–O16. The hidden sequence is `[B,5×448,1536]`.

## Action tokens and temporal mask

Use separate learned `ActionEmbedding(6,256)` and `TemporalEmbedding(16,256)`. Token `t` is their sum for action ID `A_t` and position `t`; input `[B,16]` becomes `[B,16,256]`. Repeated actions therefore retain distinct time positions. Four-head cross-attention uses video states as queries and the 16 separate action tokens as keys and values. Its projections are `Q:1536→256`, `K:256→256`, `V:256→256`, `O:256→1536`; head dimension is 64.

Cross-attention queries only the 1,792 future video tokens. A boolean SDPA allow-mask has shape `[1,1,1792,16]` (broadcast over batch and four heads); `True` means permitted. It contains 7,168 true entries:

| Latent | Global video-token indices | Future-query indices | Allowed separate action-token indices |
|---|---:|---:|---:|
| 0, observed O0 | 0–447 | omitted | none |
| 1, O1–O4 | 448–895 | 0–447 | 0–3 |
| 2, O5–O8 | 896–1343 | 448–895 | 4–7 |
| 3, O9–O12 | 1344–1791 | 896–1343 | 8–11 |
| 4, O13–O16 | 1792–2239 | 1344–1791 | 12–15 |

The resulting `Q` is `[B,4,1792,64]`, `K/V` each `[B,4,16,64]`, attention result `[B,1792,256]`, and output residual `[B,1792,1536]`. The first 448 hidden tokens pass through exactly unchanged at each injected site. Omitting observed-state queries also avoids an all-masked attention row. Later frozen self-attention can propagate information; the mask guarantees direct action access only to the aligned latent position at each injected site.

## Injection

Attach project-owned forward hooks to **main blocks 3, 11, 19, and 27**, after each full block returns its `[B,2240,1536]` tensor. These odd indices are evenly spaced by eight blocks and lie between the even-indexed VACE hint additions. Each block's self-attention, text cross-attention, and feed-forward operations complete before the action residual is added. No installed Diffusers code or native O0 path changes. The action output projection has zero-initialized weight **and** bias, so the untrained residual is exactly zero. The other projections and embeddings use ordinary initialization.

The existing VACE preprocessing and discovered latent-mask behavior remain untouched. O0 is a condition, not an evaluation target. A later training pilot would use the existing future-only loss; this audit does not define or run that pilot.

## Parameter and memory audit

| Trainable component | Parameters |
|---|---:|
| Action embedding, 6×256 | 1,536 |
| Temporal embedding, 16×256 | 4,096 |
| Per site: Q projection | 393,472 |
| Per site: K projection | 65,792 |
| Per site: V projection | 65,792 |
| Per site: zero-initialized output projection | 394,752 |
| Per site total | 919,808 |
| Four sites plus embeddings | **3,684,864** |

FP32 AdamW's two optimizer moment tensors would require about **28.1 MiB**; FP32 parameters, gradients, and moments together about **56.2 MiB**, excluding temporary optimizer buffers. At batch 1 in BF16, one future query projection is 0.875 MiB, one future residual is 5.25 MiB, the 4-head dense 1792×16 attention matrix is 0.219 MiB in BF16 or 0.438 MiB in FP32, and the boolean mask is 0.027 MiB. These are tensor-size estimates, not a measured training peak; saved activations at four sites and checkpoint recomputation add memory. The real forward-only check peaked at **4,569,509,376 allocated bytes**, versus **4,568,649,216** for baseline in the same loaded process. This small peak difference is only a forward observation and does not prove a backward pass fits 32GB.

## Zero-init prototype result

The [prototype](../../../training/wan/vace_action_token_prototype.py) loaded the real frozen transformer and the cached familiar sample. It used fixed seed 42 noise, the existing O0 control and prompt, and the actual 30-step scheduler's middle position (index 14, timestep 774, sigma 0.774521). Action-token tensors for ORIGINAL, REVERSED, and ALL-NOOP were distinct. Each of the four hooks ran once per denoiser call and observed `[1,2240,1536]`. With zero output projections, all three transformer outputs were **bit-identical to unhooked frozen VACE and to each other**; output shape was `[1,16,5,32,56]`. Frozen transformer trainable parameter count was zero. Full measured data: [prototype.json](prototype.json).

## Later implementation plan

Keep the adapter and mask in `training/wan/vace_action_token_prototype.py` or promote that small module to `vace_action_token_adapter.py`. A future pilot would add one training runner and one inference runner using the existing VACE data/O0 utilities. No M6B-1/M6B-2 code or artifacts, source manifests, dependencies, pretrained weights, or VACE mask logic need edits. During later training, hooks must remain installed through backward: gradient checkpointing can recompute selected blocks, and that recomputation must see the same action residual. The zero output projection initially gates gradients to upstream adapter layers until it begins learning; this is expected and was not tested by training here.
