# Milestone 5C-4: single-sample scene+action prompt diagnostic

**Conclusion B: Scene context improves the base-model prior, but the LoRA still does not memorize the source.** The base model changes from pale cyan scenes with moving people to a corridor with a prominent red wall when scene words are added. After 800 LoRA steps, the generated clip remains stylized, begins with a visible soldier, and shows little of the camera turn in the real ViZDoom clip. Lower training loss is not treated as visual success.

## Controlled input and settings

Sample: `my_way_home_episode_0028_clip_000011`. The new dataset contains one lossless RGB video with the same 17 source observations and the same 16 raw action-label strings as M5C-3. The source and derived video files are byte-identical to the M5C-3 files (SHA-256 `5feb092591c2ed748b4bf546091ef56772a704c941c5326c7fda432428acb655`). The original manifest was not edited; its SHA-256 before and after this run is `7d56952b72cc7c1075319ee1aa038c04b6b250913826f111f3bf63c9c8def991`.

- **Action-only prompt:** “The player moves forward while turning left for 2 steps, stays still for 8 steps, then moves forward while turning right for 6 steps.”
- **Scene+action prompt:** “A first-person gameplay view in a retro 3D stone corridor with a red wall. The player moves forward while turning left for 2 steps, stays still for 8 steps, then moves forward while turning right for 6 steps.” The requested line break is represented by one space because Finetrainers reads one prompt per line.

The new prompt is in this experiment's `train/derived_dataset/prompt.txt` and `source_map.jsonl`. The mapping also records the original prompt, the exact 16 action strings, and all 17 source PNG paths. M5C-0 and M5C-3 data and outputs were not changed.

All recorded training settings match M5C-3: Wan2.1-T2V-1.3B, BF16, 17 real frames, bicubic resize from 320×240 to 448×256, LoRA rank/alpha 8, the same 240 attention projection targets, AdamW with constant `5e-5` learning rate, batch size 1, accumulation 1, seed 42, gradient checkpointing, frozen VAE/text encoder/base transformer, and 800 optimizer steps. Both base controls were generated **before training** without LoRA. All generated comparisons used seed 42, 17 frames, 448×256, 30 inference steps, guidance scale 5.0, flow shift 3.0, and no negative prompt. The new action-only base video is pixel-identical across all 17 frames to the existing M5C-3 base video.

## Video comparison

Experiment root: `/root/autodl-tmp/outputs/wan/overfit_0016/single_sample_scene_action_0800/`

| Condition | Video path | Visual observation |
| --- | --- | --- |
| Base + action-only | `eval/base_action_only/step_000_base.mp4` | Pale cyan scene with stylized moving people; no first-person corridor or red wall. |
| Base + scene+action | `eval/base_scene_action/step_000_base.mp4` | Corridor and large red wall appear immediately, but the scene is neon/stylized and includes a character. The viewpoint barely turns. |
| M5C-3 action-only LoRA step 800 | `/root/autodl-tmp/outputs/wan/overfit_0016/single_sample_0800/eval/step_800.mp4` | Pale cyan scene and people persist; no meaningful approach to the source. |
| M5C-4 scene+action LoRA step 100 | `eval/step_100.mp4` | Blocky stone corridor and red wall persist. A drawn person appears in the first frame, then disappears. |
| M5C-4 scene+action LoRA step 200 | `eval/step_200.mp4` | Similar corridor and red wall; a soldier remains in the first frame. Little camera motion. |
| M5C-4 scene+action LoRA step 400 | `eval/step_400.mp4` | Stone blocks look somewhat more coherent, but the soldier and nearly fixed corridor remain. |
| M5C-4 scene+action LoRA step 800 | `eval/step_800.mp4` | Stone corridor and red wall are clear, yet still stylized. The visible soldier rules out a consistent first-person view; the source geometry, texture, and turn are not reproduced. |
| Real source | `eval/source_17_frames_lossless.mp4` | First-person ViZDoom corridor with brown stone textures and a large red wall. The camera turn reveals more of the corridor as the wall recedes. No visible person. |

The relative paths in the table are under the new experiment root. `eval/comparison_contact_sheet.jpg` shows frames 0, 8, and 16 for all eight rows. `eval/camera_motion_contact_sheet.jpg` shows frames 0, 4, 8, 12, and 16 for the scene base, step 800, and source. The generated scene+action walls remain nearly fixed through the clip; the source wall shifts substantially as the camera turns. This is a qualitative observation from the controlled videos, not an action-alignment score.

## Loss, runtime, VRAM, and checkpoints

| Optimizer steps | M5C-3 action-only mean loss | M5C-4 scene+action mean loss |
| --- | ---: | ---: |
| 1–100 | 0.163404 | 0.156232 |
| 101–200 | 0.130483 | 0.129152 |
| 201–400 | 0.118448 | 0.117493 |
| 401–800 | 0.109555 | 0.108835 |

M5C-4 initial/final step loss: `0.182911` / `0.085532`. All 800 losses and LoRA gradient norms were finite; the maximum logged gradient norm was `0.551957`. Training took `436.93 s`; base controls took `60.37 s`; the four checkpoint comparisons took `119.81 s`. Total active phase runtime was `617.11 s`. Peak observed device VRAM across the three phases was `11,882 MiB` (training). There was no numerical instability, malformed generated video, or checkpoint-load failure.

| Step | Adapter-only checkpoint directory | `WanPipeline.load_lora_weights()` |
| ---: | --- | --- |
| 100 | `train/lora_weights/000100/` | Passed |
| 200 | `train/lora_weights/000200/` | Passed |
| 400 | `train/lora_weights/000400/` | Passed |
| 800 | `train/lora_weights/000800/` | Passed |

Each directory contains `pytorch_lora_weights.safetensors` (23,649,160 bytes) and `scheduler/scheduler_config.json` (482 bytes). The only model weights saved are LoRA adapter weights; no base-model weights or optimizer-state tensors are present. `train/steps.jsonl` records each step's loss, learning rate, gradient norm, allocated/reserved GPU memory, and elapsed time. `train/summary.json`, `eval/base_controls.json`, `eval/summary.json`, and the experiment `summary.json` record the settings and phase results. Console logs are in the parent output directory as `single_sample_scene_action_0800_{base,train,eval}_run.log`.

The implementation is `training/wan/finetrainers_scene_action_single_sample.py`. In a fresh output directory, run `python -m training.wan.finetrainers_scene_action_single_sample all` after changing its `ROOT` constant. The fixed root prevents accidental reuse of this completed run. The experiment stopped after these single-sample controls and did not start a 500-clip run.
