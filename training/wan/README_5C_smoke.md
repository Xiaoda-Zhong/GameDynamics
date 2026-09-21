# Milestone 5C Wan LoRA smoke run

This experiment trains **only** a Wan2.1-T2V-1.3B transformer LoRA on the 16-sample `overfit_0016` manifest. It uses [Hugging Face Finetrainers' Wan SFT implementation](https://github.com/huggingface/finetrainers/blob/main/docs/models/wan.md) at commit `7c238443b7f102d3cfb4425caabd629e2ee1c02b`. Finetrainers supplies the Wan flow objective, model loading, gradient checkpointing, optimizer loop, and latent/text precomputation. The [official Wan example](https://github.com/huggingface/finetrainers/blob/main/examples/training/sft/wan/crush_smol_lora/train.sh) supplies the attention projection target pattern and logit-normal timestep sampling.

## Inputs and derived data

Source: `data/wan_training/my_way_home/overfit_0016/manifest.jsonl`. Each record has 17 source RGB PNG observations, 16 unmodified action-label strings, and one deterministic action-derived prompt. `finetrainers_data.py` validates all 16 records and writes only to `/root/autodl-tmp/outputs/wan/overfit_0016/derived_dataset/`:

- `videos/0000.mp4` through `videos/0015.mp4`: 17 frames each, 8 fps, lossless RGB H.264 encoding. Every decoded pixel is compared with the corresponding PNG. The source files remain untouched.
- `prompt.txt`: one original manifest prompt per line, in video order.
- `videos.txt`: one relative video path per line, in the same order.
- `source_map.jsonl`: global clip ID, derived video path, all 17 source PNG paths, 16 raw action strings, and the exact prompt for every sample.

`finetrainers_dataset.json` configures Finetrainers' video bucket as `[17, 256, 448]` (`frames, height, width`) with bicubic spatial resizing. The source videos are 320×240. Finetrainers does no temporal padding or frame interpolation for these 17-frame samples.

## Trainer settings

Single CUDA GPU; BF16 transformer, text encoder and VAE; batch size 1; accumulation 1; seed 42; rank/alpha 8; `blocks.*(to_q|to_k|to_v|to_out.0)` (240 matched attention projections); AdamW at constant `5e-5`; gradient checkpointing; 16 optimizer steps; no validation or W&B reporting. The wrapper explicitly freezes the VAE and text encoder. Finetrainers freezes the original transformer weights before adding LoRA. The wrapper rejects any trainable non-LoRA parameter.

The wrapper's checkpoint hook saves only actual trainable LoRA tensors using `WanPipeline.save_lora_weights`. This avoids Finetrainers' normal Accelerate training-state checkpoint, which would include the frozen base model. PEFT 0.21 returns an empty adapter state when the transformer blocks have Finetrainers activation checkpoint wrappers, so the hook normalizes the reported parameter names before saving. It verifies that the saved tensor element count equals the trainable parameter count. The checkpoint is `lora_weights/000016/pytorch_lora_weights.safetensors` (480 tensors; about 23 MiB). It loads with `WanPipeline.load_lora_weights()`.

## Outputs and reproducibility

`steps.jsonl` has one row per optimizer step: `step`, `loss`, `learning_rate`, `lora_grad_norm`, `gpu_allocated_mib`, `gpu_reserved_mib`, and `elapsed_seconds`. `summary.json` records the setup, first and final loss, runtime, peak observed device VRAM, and checkpoint path. `steps_attempt1_save_failed.jsonl` preserves the first 16-step attempt, whose final checkpoint hook failed before the adapter-name normalization fix. The successful run is in `steps.jsonl`.

The source checkout is loaded directly from `/tmp/finetrainers-wan-5c` rather than installed as a Python package. To recreate the environment and run in a fresh output directory:

```bash
git clone https://github.com/huggingface/finetrainers.git /tmp/finetrainers-wan-5c
git -C /tmp/finetrainers-wan-5c checkout 7c238443b7f102d3cfb4425caabd629e2ee1c02b
python -m pip install datasets==3.6.0 peft==0.21.0 decord==0.6.0 wandb==0.30.0 pandas==3.0.6 kornia==0.8.3 torchdata==0.11.0 hf_transfer==0.1.9 av==16.1.0
python -m training.wan.finetrainers_smoke --steps 16 --output-dir /root/autodl-tmp/outputs/wan/overfit_0016_reproduction
```

The base model must exist at `/root/autodl-tmp/models/Wan2.1-T2V-1.3B-Diffusers`. The script refuses more than 20 steps or an existing step log/checkpoint. `--prepare-only` verifies the dataset without using the GPU. `environment_changes.json` in the completed output lists every package installed or changed on this machine, including transitive dependencies. `torchao` was removed after its selected version proved incompatible with the installed Torch; it is not needed for this Wan run.

This smoke run verifies correct optimizer steps and checkpoint compatibility. A single pass over 16 different clips with random training timesteps does not establish memorization; the longer overfit run is a separate decision.
