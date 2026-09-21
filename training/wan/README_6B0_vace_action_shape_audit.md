# M6B-0: VACE explicit action conditioning design and shape audit

**Status:** architecture audit complete. No training, finetuning, adapter integration, or package change was performed.

M6A-4 retains its official **C. VACE CONDITIONING FAILS** classification under the predeclared decoded frame-0 threshold. Its 12/12 early-future state identification and 0.162 median separation margin support using Wan2.1-VACE-1.3B as the state-conditioned backbone. From M6B onward, O0 is an observed input; the intended world-model output is O1...O16 conditioned on O0 and A0...A15. O0 is excluded from future-prediction evaluation.

## One real sample and action IDs

The probe read the existing TRAIN sample `my_way_home_episode_0028_clip_000011` (M6A-3 state A): 17 consecutive, unmodified PNGs, O0...O16, and its 16 recorded transition actions. The source PNGs are 320x240; the existing M6A in-memory bicubic preprocessing produced 448x256 model inputs. No source manifest or frame was changed. In the episode metadata, action A_t is the transition from observation O_t to O_(t+1).

| Integer ID | Exact existing action string |
|---:|---|
| 0 | `MOVE_FORWARD` |
| 1 | `TURN_LEFT` |
| 2 | `TURN_RIGHT` |
| 3 | `NOOP` |
| 4 | `MOVE_FORWARD_LEFT` |
| 5 | `MOVE_FORWARD_RIGHT` |

For this sample, `action_ids` is `[4,4,3,3,3,3,3,3,3,3,5,5,5,5,5,5]`, shape `[1,16]`. The vocabulary order comes from `data_pipeline/collect_vizdoom.py`; the original strings remain authoritative metadata. There is no action text prompt.

## Measured shape trace

The probe loaded the local pinned `Wan-AI/Wan2.1-VACE-1.3B-diffusers` weights with installed Diffusers 0.40.0. `B=1`, channels are RGB or latent channels, and tensor order is `[B,C,T,H,W]` until tokenization.

| Stage | Observed shape |
|---|---:|
| Real RGB clip after in-memory resize/normalization | `[1,3,17,256,448]` |
| Wan VAE posterior parameters (mean + log variance) | `[1,32,5,32,56]` |
| Wan VAE posterior mode / real target latents | `[1,16,5,32,56]` |
| Native VACE conditioning video and pixel mask | `[1,3,17,256,448]`, `[1,1,17,256,448]` |
| VACE inactive + reactive video latents | `[1,32,5,32,56]` |
| VACE mask latents | `[1,64,5,32,56]` |
| VACE transformer control input (video + mask) | `[1,96,5,32,56]` |
| VACE transformer noisy-latent input | `[1,16,5,32,56]` |
| Frozen noisy and control `Conv3d` patch embedding outputs | each `[1,1536,5,16,28]` |
| Flattened temporal-spatial tokens on each branch | each `[1,2240,1536]` |

The VAE has temporal scale 4 and spatial scale 8. Its encoder handles O0 alone, then input chunks O1...O4, O5...O8, O9...O12, and O13...O16, yielding five temporal latent positions. The transformer patch size is `(1,2,2)` and width is `12 heads × 128 = 1536`; each temporal position has `16 × 28 = 448` spatial tokens. The patch embedding is followed immediately by temporal-spatial flattening in the installed VACE transformer.

### Native O0 conditioning caveat found by the probe

The supplied 17-frame pixel mask was exactly black at frame 0 and white at frames 1...16, and the inactive VACE conditioning latent carrying O0 was nonzero. However, the installed `prepare_masks` uses `nearest-exact` interpolation from 17 frames to 5 latent positions. It selects pixel-mask indices `[1,5,8,11,15]`, so **the resulting 64-channel latent mask is white at all five positions**, including latent position 0. This is an observed property of Diffusers 0.40.0 for this 17-frame protocol. It is consistent with M6A-4's failure of the hard frame-0 threshold, but does not prove the cause of that failure. The proposed action adapter preserves the native O0 video/mask preparation and does not change the M6A-4 result. Any correction of this mask behavior would be a separate, predeclared protocol change before training.

## Exact action-to-latent schedule

`raw_actions[t]` drives O_t → O_(t+1). Four ordered actions are associated with each future temporal latent position. The token ranges follow the transformer's time-major flattening.

| Latent `k` | Output observations represented | Directly injected actions | Token indices |
|---:|---|---|---:|
| 0 | O0, observed state | none; zero action bias | 0...447 |
| 1 | O1...O4 | A0...A3 | 448...895 |
| 2 | O5...O8 | A4...A7 | 896...1343 |
| 3 | O9...O12 | A8...A11 | 1344...1791 |
| 4 | O13...O16 | A12...A15 | 1792...2239 |

These are the encoder's four-frame chunks and the adapter's **direct injection locations**. Causal VAE context and transformer attention can carry information across positions; a latent is not an isolated four-frame receptive field. The four actions within each chunk retain order through separate positions in a concatenated feature vector, though the 4:1 temporal compression cannot guarantee exact sub-frame timing without empirical action tests.

## Proposed minimal adapter (design only)

Use a learned six-entry action table, then one shared projection for every four-action group. Initialize the projection weight and bias to zero so an untrained adapter reproduces the frozen VACE path. At each transformer invocation, add the projected action signal to the **noisy-latent** `patch_embedding` output, before `flatten(2).transpose(1,2)`. Keep the VACE control branch and its O0 conditioning unchanged. A per-call hook or thin wrapper around the existing patch embedding can do this without editing installed Diffusers or the pretrained checkpoint.

```text
action_ids:                                  [B,16]       int64
Embedding(6,32):                             [B,16,32]
ordered reshape (A0..A3, ..., A12..A15):     [B,4,128]
zero-initialized Linear(128,1536):            [B,4,1536]
prepend fixed zero for observed O0:           [B,5,1536]
transpose and broadcast:                      [B,1536,5,1,1]
add to frozen patch_embedding(noisy_latents): [B,1536,5,16,28]
flatten into transformer tokens:              [B,2240,1536]
```

The `Linear` has distinct input columns for all four action slots, so swapping A0 and A1 need not produce the same feature. Only tokens at temporal position `k=t//4+1` receive a direct signal from A_t; attention may propagate that information later. The generic scene prompt stays the same for all states and actions. When future training is authorized, construct the diffusion target by encoding the exact real O0...O16 clip, but apply the denoising loss only at temporal latent positions `[1,2,3,4]` (mask `[0,1,1,1,1]`). This implements the O0 + actions → future observations formulation at the latent resolution. Frame-level evaluation should likewise use O1...O16 only.

Parameter count: `Embedding = 6×32 = 192`; `Linear = 128×1536 + 1536 bias = 198,144`; **total = 198,336 trainable parameters** (about 0.76 MiB of FP32 weights, or about 3.03 MiB including FP32 gradients and two Adam moments). The adapter is small relative to the 32 GB GPU. The probe's 2,328.8 MiB peak allocation covered VAE encoding and the two patch layers, **not a full frozen-transformer backward pass**; future training must verify activation memory with BF16 and gradient checkpointing before claiming a 32 GB fit.

Initially freeze the VACE transformer's noisy/control patch embeddings, all 30 main blocks, all 15 VACE blocks, condition/time/text embeddings, and output head. Freeze the Wan VAE and text encoder; tokenizer has no trainable weights. Train only the action embedding and projection. The base model has no Control LoRA in this design.

## Smallest future implementation surface

No implementation file was changed in M6B-0. Once training is separately authorized, the minimal new files are:

1. `training/wan/vace_action_adapter.py`: fixed vocabulary, shape checks, two-layer parameter module, and per-call patch-output injection.
2. `training/wan/vace_action_train.py`: read existing real clips/actions, prepare native O0 VACE control, freeze base weights, compute the future-only latent loss, and save only adapter weights.
3. `training/wan/vace_action_infer.py`: load adapter with frozen VACE, pass O0 plus integer action IDs, and evaluate O1...O16.

Reuse the current selected-state metadata, source-frame preprocessing, and generic prompt. Do not alter source manifests, M6A-4 artifacts, installed packages, or pretrained weights. This audit does not establish action controllability; it establishes tensor compatibility and a temporally ordered injection design.

Probe output: `data/experiments/m6b0_vace_action_shape_audit/shape_audit.json`. Source of installed shape logic: `diffusers/models/autoencoders/autoencoder_kl_wan.py` (`_encode`), `diffusers/pipelines/wan/pipeline_wan_vace.py` (`prepare_video_latents`, `prepare_masks`), and `diffusers/models/transformers/transformer_wan_vace.py` (`patch_embedding` and `forward`).
