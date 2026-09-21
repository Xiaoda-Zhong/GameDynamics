#!/usr/bin/env python3
"""M6A: pinned Wan Control LoRA smoke and 16-clip current-frame prototype."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import re
import time
from pathlib import Path

import torch

from training.wan.finetrainers_data import prepare_finetrainers_data
from training.wan.current_frame_metrics import source_frame_rgb
from training.wan.finetrainers_scene_action_single_sample import SCENE_CONTEXT
from training.wan.finetrainers_smoke import (
    FINETRAINERS_COMMIT, GPUMemorySampler, MANIFEST, MODEL, OUTPUT,
    _require_official_source,
)


SOURCE = Path("/tmp/finetrainers-wan-5c")
ROOT = OUTPUT / "current_frame_control_lora_0320"
SMOKE = ROOT / "smoke_0008"
TRAIN = ROOT / "train_0320"
SINGLE_SAMPLE_ID = "my_way_home_episode_0028_clip_000011"
SINGLE_ROOT = OUTPUT / "current_frame_single_sample_0800"
SINGLE_TRAIN = SINGLE_ROOT / "train_0800"
RANK = 128  # Pinned Finetrainers Wan image_condition example, not old T2V rank 8.
ALPHA = 128
TARGET_MODULES = r"blocks.*(to_q|to_k|to_v|to_out.0|ff.net.0.proj|ff.net.2)"
LR = 2e-5


def _hash_manifest() -> str:
    return hashlib.sha256(MANIFEST.read_bytes()).hexdigest()


def _check_derived_dataset(
    output: Path, sample_id: str | None = None,
    sample_ids: tuple[str, ...] | None = None, prompt_override: str | None = None,
) -> None:
    rows = [json.loads(line) for line in (output / "derived_dataset/source_map.jsonl").read_text().splitlines()]
    expected_size = len(sample_ids) if sample_ids is not None else (1 if sample_id is not None else 16)
    if len(rows) != expected_size or len({row["global_clip_id"] for row in rows}) != expected_size:
        raise ValueError(f"Control dataset must contain exactly {expected_size} distinct clips")
    if sample_id is not None and rows[0]["global_clip_id"] != sample_id:
        raise ValueError("Single-sample Control dataset selected the wrong clip")
    if sample_ids is not None and tuple(row["global_clip_id"] for row in rows) != sample_ids:
        raise ValueError("Four-state Control dataset selected the wrong clips or order")
    prompts = (output / "derived_dataset/prompt.txt").read_text().splitlines()
    if len(prompts) != expected_size:
        raise ValueError(f"Control dataset must contain exactly {expected_size} prompts")
    for index, row in enumerate(rows):
        if (
            prompts[index] != row["prompt"]
            or row["prompt"] != (prompt_override if prompt_override is not None
                                  else f"{SCENE_CONTEXT} {row['source_prompt']}")
            or len(row["source_rgb_frames"]) != 17
            or len(row["raw_actions"]) != 16
            or not all(isinstance(action, str) for action in row["raw_actions"])
        ):
            raise ValueError(f"Derived control dataset changed sample {index}")


def run_training(
    output: Path, steps: int, checkpoints: tuple[int, ...], *, sample_id: str | None = None,
    sample_ids: tuple[str, ...] | None = None, prompt_override: str | None = None,
) -> dict:
    if tuple(sorted(set(checkpoints))) != checkpoints or checkpoints[-1] != steps:
        raise ValueError("Checkpoints must be ordered, unique, and end at the final step")
    if sample_id is not None and sample_ids is not None:
        raise ValueError("Choose either one sample or a tuple of samples")
    if sample_ids is not None and (len(sample_ids) != 4 or steps != 1600 or prompt_override is None):
        raise ValueError("M6A-3 requires four states, 1600 steps, and a generic prompt")
    if (output / "steps.jsonl").exists() or (output / "lora_weights").exists():
        raise FileExistsError(f"Control LoRA training already started in {output}")
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1 or not torch.cuda.is_bf16_supported():
        raise RuntimeError("Exactly one BF16-capable CUDA GPU is required")
    _require_official_source(SOURCE)
    from finetrainers import BaseArgs
    from finetrainers.models.wan.control_specification import WanControlModelSpecification
    from finetrainers.parallel import ParallelBackendEnum
    from finetrainers.trainer.control_trainer.config import ControlLowRankConfig
    from finetrainers.trainer.control_trainer.trainer import ControlTrainer

    class AdapterOnlyCheckpointer:
        def __init__(self, trainer: ControlTrainer) -> None:
            self.trainer = trainer
            self.paths: dict[int, Path] = {}

        def save(self, step: int, force: bool = False, *, _device: torch.device,
                 _is_main_process: bool) -> str | None:
            if step not in checkpoints:
                if force:
                    raise RuntimeError(f"Unexpected final Control LoRA save at step {step}")
                return None
            if not _is_main_process:
                raise RuntimeError("Single-GPU Control LoRA checkpoint requires the main process")
            if step in self.paths:
                return str(self.paths[step])
            trainer = self.trainer
            state_dict = {}
            for name, parameter in trainer.transformer.named_parameters():
                if not parameter.requires_grad:
                    continue
                if ".lora_A.default." not in name and ".lora_B.default." not in name:
                    raise RuntimeError(f"Non-LoRA trainable parameter: {name}")
                key = name.replace("._checkpoint_wrapped_module", "").replace(".default.", ".")
                if key in state_dict:
                    raise RuntimeError(f"Duplicate Control LoRA key: {key}")
                state_dict[key] = parameter.detach().cpu().contiguous()
            if not state_dict or sum(t.numel() for t in state_dict.values()) != trainer.trainable_parameters:
                raise RuntimeError("Control LoRA state does not match trainable parameter count")
            path = output / "lora_weights" / f"{step:06d}"
            injection = trainer.model_specification.control_injection_layer_name
            injection_rank = trainer.model_specification._original_control_layer_out_features
            metadata = {"lora_config": json.dumps({
                "r": RANK,
                "lora_alpha": ALPHA,
                "init_lora_weights": True,
                "target_modules": trainer._get_lora_target_modules(),
                "rank_pattern": {injection: injection_rank},
                "alpha_pattern": {injection: injection_rank},
            })}
            trainer.model_specification._save_lora_weights(
                str(path), state_dict, None, trainer.scheduler, metadata,
            )
            self.paths[step] = path
            return str(path)

        def load(self, *_args: object, **_kwargs: object) -> None:
            raise RuntimeError("This adapter-only run does not resume optimizer state")

    class MeasuredControlTrainer(ControlTrainer):
        def __init__(self, args: BaseArgs, spec: WanControlModelSpecification) -> None:
            super().__init__(args, spec)
            self.started = time.monotonic()
            self.metric_path = output / "steps.jsonl"
            self.target_names: list[str] = []
            self.trainable_parameters = 0
            self.checkpointer: AdapterOnlyCheckpointer | None = None
            self.state_counts = {item: 0 for item in sample_ids or ()}

        def _set_components(self, components: dict) -> None:
            for name in ("text_encoder", "vae"):
                module = components.get(name)
                if module is not None:
                    module.requires_grad_(False)
                    module.eval()
            super()._set_components(components)

        def _prepare_trainable_parameters(self) -> None:
            expanded_in_channels = self.transformer.patch_embedding.in_channels
            self.target_names = [
                name for name, module in self.transformer.named_modules()
                if (
                    name == "patch_embedding" or re.fullmatch(TARGET_MODULES, name)
                ) and isinstance(module, (torch.nn.Linear, torch.nn.Conv3d))
            ]
            if "patch_embedding" not in self.target_names or len(self.target_names) < 181:
                raise RuntimeError("Official Control LoRA targets did not match Wan attention, FF, and injection")
            super()._prepare_trainable_parameters()
            trainable = [(name, p) for name, p in self.transformer.named_parameters() if p.requires_grad]
            if not trainable or any(".lora_A.default." not in name and ".lora_B.default." not in name for name, _ in trainable):
                raise RuntimeError("Only Control LoRA adapter parameters may be trainable")
            if not any("patch_embedding.lora_A.default." in name for name, _ in trainable):
                raise RuntimeError("Control injection LoRA is not trainable")
            self.trainable_parameters = sum(param.numel() for _, param in trainable)
            print("CONTROL_ARCHITECTURE " + json.dumps({
                "control_injection": "patch_embedding", "original_in_channels": 16,
                "expanded_in_channels": expanded_in_channels,
                "injection_rank": self.model_specification._original_control_layer_out_features,
                "rank": RANK, "alpha": ALPHA,
                "target_module_count": len(self.target_names),
                "target_examples": self.target_names[:10],
                "trainable_parameter_count": self.trainable_parameters,
                "trainable_tensor_count": len(trainable),
            }), flush=True)

        def _prepare_for_training(self) -> None:
            super()._prepare_for_training()
            if self.state.num_trainable_parameters != self.trainable_parameters:
                raise RuntimeError("Optimizer trainable count differs from inspected Control LoRA count")
            backend = self.state.parallel_backend
            original_log = backend.log

            def record_metrics(metrics: dict, step: int) -> None:
                loss = float(metrics["train/global_avg_loss"])
                grad_norm = float(metrics["train/grad_norm"])
                lr = float(self.optimizer.param_groups[0]["lr"])
                if not math.isfinite(loss) or not math.isfinite(grad_norm) or grad_norm <= 0:
                    raise FloatingPointError(f"Invalid Control LoRA loss/gradient at step {step}")
                if not math.isfinite(lr) or lr <= 0:
                    raise FloatingPointError(f"Invalid Control LoRA learning rate at step {step}")
                row = {
                    "step": step, "loss": loss, "learning_rate": lr,
                    "lora_grad_norm": grad_norm,
                    "gpu_allocated_mib": round(torch.cuda.memory_allocated() / 2**20, 2),
                    "gpu_reserved_mib": round(torch.cuda.memory_reserved() / 2**20, 2),
                    "elapsed_seconds": round(time.monotonic() - self.started, 2),
                }
                if sample_ids is not None:
                    current_id = getattr(self.model_specification, "current_sample_id", None)
                    if current_id not in self.state_counts:
                        raise RuntimeError(f"Unknown sampled state at step {step}: {current_id}")
                    self.state_counts[current_id] += 1
                    row["sample_id"] = current_id
                    row["per_state_sample_counts"] = dict(self.state_counts)
                with self.metric_path.open("a", encoding="utf-8") as file:
                    file.write(json.dumps(row) + "\n")
                print("CONTROL_STEP " + json.dumps(row), flush=True)
                original_log(metrics, step)

            backend.log = record_metrics

        def _prepare_checkpointing(self) -> None:
            self.checkpointer = AdapterOnlyCheckpointer(self)

    before = _hash_manifest()
    dataset_config = prepare_finetrainers_data(
        MANIFEST, output, sample_id=sample_id, sample_ids=sample_ids,
        prompt_override=prompt_override,
        prompt_prefix=None if prompt_override is not None else SCENE_CONTEXT,
    )
    _check_derived_dataset(output, sample_id, sample_ids, prompt_override)
    args = BaseArgs()
    control_config = ControlLowRankConfig()
    control_config.control_type = "none"
    control_config.rank = RANK
    control_config.lora_alpha = ALPHA
    control_config.target_modules = TARGET_MODULES
    control_config.frame_conditioning_type = "index"
    control_config.frame_conditioning_index = 0
    control_config.frame_conditioning_concatenate_mask = False
    control_config.train_qk_norm = False
    args.register_args(control_config)
    args.parallel_backend = ParallelBackendEnum.ACCELERATE
    args.model_name = "wan"
    args.training_type = "control-lora"
    args.pretrained_model_name_or_path = str(MODEL)
    args.dataset_config = str(dataset_config)
    args.output_dir = str(output)
    args.logging_dir = "logs"
    args.report_to = "none"
    args.seed = 42
    args.batch_size = 1
    args.gradient_accumulation_steps = 1
    args.gradient_checkpointing = True
    args.train_steps = steps
    args.control_type = "none"
    args.rank = RANK
    args.lora_alpha = ALPHA
    args.target_modules = TARGET_MODULES
    args.frame_conditioning_type = "index"
    args.frame_conditioning_index = 0
    args.frame_conditioning_concatenate_mask = False
    args.train_qk_norm = False
    args.lr = LR
    args.lr_scheduler = "constant_with_warmup"
    args.lr_warmup_steps = max(1, steps // 10)
    args.optimizer = "adamw"
    args.beta1 = 0.9
    args.beta2 = 0.99
    args.weight_decay = 1e-4
    args.epsilon = 1e-8
    args.max_grad_norm = 1.0
    args.flow_weighting_scheme = "logit_normal"
    args.enable_precomputation = True
    args.precomputation_items = len(sample_ids) if sample_ids is not None else (1 if sample_id is not None else 16)
    args.precomputation_once = True
    args.precomputation_dir = str(output / "precomputed")
    args.dataset_shuffle_buffer_size = 1 if sample_ids is not None else (1 if sample_id is not None else 16)
    args.validation_dataset_file = None
    args.checkpointing_steps = 0
    args.logging_steps = 1
    args.compile_modules = []
    args.compile_scopes = []
    args.enable_slicing = True
    args.enable_tiling = True
    args.transformer_dtype = torch.bfloat16
    args.text_encoder_dtype = torch.bfloat16
    args.vae_dtype = torch.bfloat16

    spec = WanControlModelSpecification(pretrained_model_name_or_path=str(MODEL))
    if sample_ids is not None:
        # Tag the four precomputed latent records, then remove the tag before the
        # official forward pass. The model receives exactly the original tensors.
        source_rows = [json.loads(line) for line in (output / "derived_dataset/source_map.jsonl").read_text().splitlines()]
        reference_o0 = [source_frame_rgb(row["source_rgb_frames"][0]) * 2 - 1 for row in source_rows]
        original_prepare_latents = spec.prepare_latents
        original_collate_latents = spec.collate_latents
        spec.precomputed_state_indices = []
        spec.current_sample_id = None

        def labeled_prepare_latents(*prepare_args, **prepare_kwargs):
            video = prepare_kwargs.get("video")
            if video is None or video.ndim != 5 or video.shape[0] != 1 or video.shape[1] != 17:
                raise ValueError("Expected one exact 17-frame source video for state labeling")
            first = video[0, 0].detach().float().cpu()
            errors = [float((first - reference).abs().mean()) for reference in reference_o0]
            index = min(range(4), key=lambda item: errors[item])
            if errors[index] > 0.03 or sorted(errors)[1] - errors[index] < 0.05:
                raise ValueError(f"Could not identify exact source O0 during precomputation: {errors}")
            spec.precomputed_state_indices.append(index)
            prepared = original_prepare_latents(*prepare_args, **prepare_kwargs)
            prepared["__state_index"] = torch.tensor([index], dtype=torch.int64)
            return prepared

        def labeled_collate_latents(items):
            indices = [int(item.pop("__state_index").item()) for item in items]
            if len(indices) != 1:
                raise ValueError("M6A-3 state tracking requires batch size one")
            spec.current_sample_id = sample_ids[indices[0]]
            return original_collate_latents(items)

        spec.prepare_latents = labeled_prepare_latents
        spec.collate_latents = labeled_collate_latents
    trainer = MeasuredControlTrainer(args, spec)
    with GPUMemorySampler() as sampler:
        trainer.run()
    if _hash_manifest() != before:
        raise RuntimeError("Source manifest changed during M6A training")
    if tuple(trainer.checkpointer.paths) != checkpoints:
        raise RuntimeError("Expected Control LoRA checkpoints were not all saved")
    rows = [json.loads(line) for line in trainer.metric_path.read_text().splitlines()]
    if len(rows) != steps or [row["step"] for row in rows] != list(range(1, steps + 1)):
        raise RuntimeError("Control LoRA optimizer-step log is incomplete")
    if sample_ids is not None:
        expected_counts = {item: steps // len(sample_ids) for item in sample_ids}
        if (steps % len(sample_ids) != 0 or trainer.state_counts != expected_counts
                or sorted(spec.precomputed_state_indices) != list(range(len(sample_ids)))
                or rows[-1]["per_state_sample_counts"] != expected_counts):
            raise RuntimeError("Four states were not sampled exactly 400 times each")
    checkpoint_info = {}
    for step, path in trainer.checkpointer.paths.items():
        weights = path / "pytorch_lora_weights.safetensors"
        if not weights.is_file():
            raise FileNotFoundError(weights)
        checkpoint_info[str(step)] = {"path": str(path), "adapter_bytes": weights.stat().st_size}
    result = {
        "framework": "Pinned Finetrainers Wan ControlTrainer Control LoRA",
        "finetrainers_commit": FINETRAINERS_COMMIT,
        "model": str(MODEL), "source_manifest": str(MANIFEST),
        "source_manifest_sha256": before,
        "dataset_size": len(sample_ids) if sample_ids is not None else (1 if sample_id is not None else 16),
        "training_sample_id": sample_id, "training_sample_ids": sample_ids,
        "prompt": prompt_override, "per_state_sample_counts": trainer.state_counts,
        "precomputed_state_order": [sample_ids[index] for index in spec.precomputed_state_indices] if sample_ids is not None else None,
        "steps": steps, "seed": 42,
        "frames": 17, "height": 256, "width": 448,
        "batch_size": 1, "gradient_accumulation_steps": 1,
        "dtype": "bf16", "rank": RANK, "lora_alpha": ALPHA,
        "control_type": "none", "frame_conditioning_type": "index", "frame_conditioning_index": 0,
        "control_injection": "patch_embedding", "original_in_channels": 16, "expanded_in_channels": 32,
        "injection_rank": trainer.model_specification._original_control_layer_out_features,
        "target_module_pattern": trainer._get_lora_target_modules(),
        "target_module_count": len(trainer.target_names),
        "target_module_examples": trainer.target_names[:10],
        "trainable_parameters": trainer.trainable_parameters,
        "learning_rate": LR, "lr_scheduler": "constant_with_warmup",
        "lr_warmup_steps": args.lr_warmup_steps,
        "initial_loss": rows[0]["loss"], "final_loss": rows[-1]["loss"],
        "mean_loss_first_half": sum(row["loss"] for row in rows[:steps // 2]) / (steps // 2),
        "mean_loss_second_half": sum(row["loss"] for row in rows[steps // 2:]) / (steps - steps // 2),
        "max_lora_grad_norm": max(row["lora_grad_norm"] for row in rows),
        "runtime_seconds": round(time.monotonic() - trainer.started, 2),
        "peak_gpu_vram_mib_observed": sampler.peak_mib,
        "checkpoint_paths": checkpoint_info,
        "packages": {name: importlib.metadata.version(name) for name in (
            "torch", "diffusers", "transformers", "accelerate", "peft", "datasets", "decord")},
    }
    (output / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("smoke", "train", "single_0800"))
    phase = parser.parse_args().phase
    if phase == "smoke":
        output, steps, checkpoints, sample_id = SMOKE, 8, (8,), None
    elif phase == "train":
        output, steps, checkpoints, sample_id = TRAIN, 320, (80, 160, 320), None
    else:
        output, steps, checkpoints, sample_id = (
            SINGLE_TRAIN, 800, (100, 200, 400, 800), SINGLE_SAMPLE_ID,
        )
    print(json.dumps(run_training(output, steps, checkpoints, sample_id=sample_id), indent=2))


if __name__ == "__main__":
    main()
