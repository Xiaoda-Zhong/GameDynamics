# Milestone 5C-3: single-sample overfit diagnostic

Only `my_way_home_episode_0028_clip_000011` was selected, in memory, from `data/wan_training/my_way_home/overfit_0016/manifest.jsonl`. The manifest was not edited; its SHA-256 was `7d56952b72cc7c1075319ee1aa038c04b6b250913826f111f3bf63c9c8def991` before and after the run. The derived file-list dataset contains one original prompt and one lossless RGB video made from the 17 referenced source PNGs. Decoding the source comparison MP4 reproduces every PNG pixel exactly. The source is 320×240 at 8 fps; the trainer applies its existing bicubic resize to 448×256.

The training settings match 5C-2: Wan2.1-T2V-1.3B, BF16, 17 frames, 448×256, rank/alpha 8, the same 240 `blocks.*(to_q|to_k|to_v|to_out.0)` linear projections, constant `5e-5` learning rate, AdamW, batch size 1, accumulation 1, seed 42, gradient checkpointing, frozen VAE, frozen text encoder, and frozen base transformer. The trainer's existing precomputation setting remains 16 items; each comes from the sole selected clip. Training stopped at 800 optimizer steps. Checkpoints at 100, 200, 400, and 800 contain only LoRA adapter tensors, and each was loaded with `WanPipeline.load_lora_weights()` during comparison generation.

All five comparison videos use the exact training prompt, seed 42, 448×256, 17 frames, 30 inference steps, guidance scale 5.0, flow shift 3.0, and no negative prompt. Step 0 uses the frozen pretrained model.

## Results

| Optimizer steps | Mean loss |
| --- | ---: |
| 1–100 | 0.163404 |
| 101–200 | 0.130483 |
| 201–400 | 0.118448 |
| 401–800 | 0.109555 |

Final step loss: **0.090171**. Training runtime: **479.21 s**. End-to-end runtime, including comparisons: **690.82 s**. Peak observed device VRAM: **14,502 MiB** (during inference). All 800 per-step losses and gradient norms were finite; the maximum logged LoRA gradient norm was 0.602224. The 401–500 mean loss temporarily rose to 0.126157, then the 701–800 mean fell to 0.096885. No numerical failure or checkpoint-load failure occurred.

Visual inspection of frames 0, 8, and 16 at every checkpoint found no progressive approach to the target clip. The generated videos show figures moving against a pale cyan scene; even step 800 does not show first-person gameplay, the stone corridor, or the large red wall. The source shows a camera turn that reveals more of the corridor as the red wall recedes. Generated motion remains character/object motion and does not approach that camera path. Under these fixed settings, the pipeline did not visibly memorize this one video by step 800, despite decreasing training loss.

## Files

Run root: `/root/autodl-tmp/outputs/wan/overfit_0016/single_sample_0800/`

- `train/steps.jsonl`: one JSON object per optimizer step with `step`, `loss`, `learning_rate`, `lora_grad_norm`, `gpu_allocated_mib`, `gpu_reserved_mib`, and `elapsed_seconds`.
- `train/loss_100_step_intervals.jsonl`: eight records with `steps` and `mean_loss`.
- `train/summary.json`: configuration, checkpoint paths, loss summaries, runtime, and peak observed VRAM.
- `train/lora_weights/{000100,000200,000400,000800}/pytorch_lora_weights.safetensors`: adapter-only checkpoints.
- `eval/source_17_frames_lossless.mp4`: exact source observations, viewable at their original resolution.
- `eval/step_000_base.mp4`, `step_100.mp4`, `step_200.mp4`, `step_400.mp4`, `step_800.mp4`: controlled comparisons.
- `eval/summary.json`: inference settings, video paths, checkpoint-load results, runtime, and peak observed VRAM.
- `summary.json`: combined result. `single_sample_0800_run.log` in the parent directory contains console output.

Reproduce in a fresh output directory by changing `ROOT` in `finetrainers_single_sample.py`, then running:

```bash
python -m training.wan.finetrainers_single_sample all
```

The exact source manifest, local model, and pinned Finetrainers checkout described in `README_5C_smoke.md` are required.
