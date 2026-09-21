#!/usr/bin/env python3
"""Run at most 20 Wan LoRA smoke steps using Hugging Face Finetrainers.

Finetrainers owns the Wan model specification, flow objective, backward pass,
optimizer, and gradient checkpointing. This wrapper enforces frozen components,
records local JSONL metrics, and saves only the LoRA adapter.
"""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import math
import re
import subprocess
import sys
import threading
import time
from pathlib import Path

import torch

from training.wan.finetrainers_data import prepare_finetrainers_data


REPO_ROOT = Path(__file__).resolve().parents[2]
MANIFEST = REPO_ROOT / "data/wan_training/my_way_home/overfit_0016/manifest.jsonl"
MODEL = Path("/root/autodl-tmp/models/Wan2.1-T2V-1.3B-Diffusers")
OUTPUT = Path("/root/autodl-tmp/outputs/wan/overfit_0016")
FINETRAINERS_COMMIT = "7c238443b7f102d3cfb4425caabd629e2ee1c02b"
TARGET_MODULES = r"blocks.*(to_q|to_k|to_v|to_out.0)"  # Official Wan example.
RANK = 8
LR = 5e-5


def _require_official_source(source: Path) -> None:
    result = subprocess.run(
        ["git", "-C", str(source), "rev-parse", "HEAD"],
        capture_output=True, text=True, check=True,
    )
    if result.stdout.strip() != FINETRAINERS_COMMIT:
        raise ValueError(f"Finetrainers must be at {FINETRAINERS_COMMIT}, found {result.stdout.strip()}")
    sys.path.insert(0, str(source))


class GPUMemorySampler:
    """Track observed device VRAM throughout precomputation and training."""

    def __init__(self) -> None:
        self.peak_mib = 0
        self.error: Exception | None = None
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._sample, daemon=True)

    def _sample(self) -> None:
        while not self._stop.is_set():
            try:
                result = subprocess.run(
                    ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                    capture_output=True, text=True, check=True,
                )
                used = [int(line.strip()) for line in result.stdout.splitlines() if line.strip()]
                if len(used) != 1:
                    raise ValueError(f"Expected one GPU, found {len(used)}")
                self.peak_mib = max(self.peak_mib, used[0])
            except Exception as error:
                self.error = error
                return
            self._stop.wait(0.25)

    def __enter__(self) -> "GPUMemorySampler":
        self._thread.start()
        return self

    def __exit__(self, *_exc: object) -> None:
        self._stop.set()
        self._thread.join()
        if self.error is not None:
            raise RuntimeError("GPU memory sampling failed") from self.error


def run_training(
    source: Path,
    output: Path,
    steps: int,
    checkpoint_steps: tuple[int, ...],
    *,
    step_label: str = "TRAIN_STEP",
    sample_id: str | None = None,
    prompt_override: str | None = None,
) -> dict:
    """Use the same pinned Wan SFT configuration for a bounded LoRA run."""
    if steps < 1 or tuple(sorted(set(checkpoint_steps))) != checkpoint_steps or checkpoint_steps[-1] != steps:
        raise ValueError("Checkpoint steps must be unique, sorted, and end at the final step")
    if checkpoint_steps[0] < 1:
        raise ValueError("Checkpoint steps must be positive")
    _require_official_source(source)
    from finetrainers import BaseArgs
    from finetrainers.models.wan import WanModelSpecification
    from finetrainers.parallel import ParallelBackendEnum
    from finetrainers.trainer.sft_trainer.config import SFTLowRankConfig
    from finetrainers.trainer.sft_trainer.trainer import SFTTrainer

    class LoraOnlyCheckpointer:
        def __init__(self, trainer: SFTTrainer) -> None:
            self.trainer = trainer
            self.paths: dict[int, Path] = {}

        def save(self, step: int, force: bool = False, *, _device: torch.device,
                 _is_main_process: bool) -> str | None:
            if step not in checkpoint_steps:
                if force:
                    raise RuntimeError(f"Unexpected final LoRA save at step {step}")
                return None
            if not _is_main_process:
                raise RuntimeError("Single-GPU LoRA save requires the main process")
            if step in self.paths:
                return str(self.paths[step])  # Final forced save repeats the last scheduled step.
            trainer = self.trainer
            # Finetrainers wraps each block for activation checkpointing. PEFT 0.21
            # returns an empty dict for that wrapped model, so normalize the
            # actual trainable parameter names before the official Wan save API.
            state_dict = {}
            for name, parameter in trainer.transformer.named_parameters():
                if not parameter.requires_grad:
                    continue
                if ".lora_A.default." not in name and ".lora_B.default." not in name:
                    raise RuntimeError(f"Non-LoRA trainable parameter: {name}")
                key = name.replace("._checkpoint_wrapped_module", "").replace(".default.", ".")
                if key in state_dict:
                    raise RuntimeError(f"Duplicate LoRA key after unwrapping: {key}")
                state_dict[key] = parameter.detach().cpu().contiguous()
            if not state_dict or sum(t.numel() for t in state_dict.values()) != trainer.trainable_parameters:
                raise RuntimeError("LoRA checkpoint does not match trainable parameter count")
            path = output / "lora_weights" / f"{step:06d}"
            metadata = {"lora_config": json.dumps({
                "r": RANK, "lora_alpha": RANK, "init_lora_weights": True,
                "target_modules": TARGET_MODULES,
            })}
            trainer.model_specification._save_lora_weights(
                str(path), state_dict, trainer.scheduler, metadata,
            )
            self.paths[step] = path
            return str(path)

        def load(self, *_args: object, **_kwargs: object) -> None:
            raise RuntimeError("This adapter-only run does not resume optimizer state")

    class MeasuredWanTrainer(SFTTrainer):
        def __init__(self, args: BaseArgs, spec: WanModelSpecification) -> None:
            super().__init__(args, spec)
            self.started = time.monotonic()
            self.metric_path = output / "steps.jsonl"
            self.target_names: list[str] = []
            self.checkpointer: LoraOnlyCheckpointer | None = None

        def _set_components(self, components: dict) -> None:
            for name in ("text_encoder", "vae"):
                module = components.get(name)
                if module is not None:
                    module.requires_grad_(False)
                    module.eval()
            super()._set_components(components)

        def _prepare_trainable_parameters(self) -> None:
            self.target_names = [
                name for name, module in self.transformer.named_modules()
                if re.fullmatch(TARGET_MODULES, name) and isinstance(module, torch.nn.Linear)
            ]
            if not self.target_names:
                raise RuntimeError("The official Wan target pattern matched no linear projections")
            super()._prepare_trainable_parameters()
            trainable = [(name, p) for name, p in self.transformer.named_parameters() if p.requires_grad]
            if not trainable or any("lora_" not in name for name, _ in trainable):
                raise RuntimeError("Only transformer LoRA parameters may be trainable")
            self.trainable_parameters = sum(param.numel() for _, param in trainable)
            print(json.dumps({
                "trainable_parameters": self.trainable_parameters,
                "target_module_count": len(self.target_names),
                "target_modules": sorted(set(name.rsplit(".", 1)[-1] for name in self.target_names)),
                "target_examples": self.target_names[:8],
            }), flush=True)

        def _prepare_for_training(self) -> None:
            super()._prepare_for_training()
            backend = self.state.parallel_backend
            original_log = backend.log

            def record_metrics(metrics: dict, step: int) -> None:
                loss = float(metrics["train/global_avg_loss"])
                grad_norm = float(metrics["train/grad_norm"])
                lr = float(self.optimizer.param_groups[0]["lr"])
                if not math.isfinite(loss) or not math.isfinite(grad_norm) or grad_norm <= 0:
                    raise FloatingPointError(f"Invalid loss or LoRA gradient at step {step}")
                if not math.isfinite(lr) or lr <= 0:
                    raise FloatingPointError(f"Invalid learning rate at step {step}")
                row = {
                    "step": step, "loss": loss, "learning_rate": lr,
                    "lora_grad_norm": grad_norm,
                    "gpu_allocated_mib": round(torch.cuda.memory_allocated() / 2**20, 2),
                    "gpu_reserved_mib": round(torch.cuda.memory_reserved() / 2**20, 2),
                    "elapsed_seconds": round(time.monotonic() - self.started, 2),
                }
                with self.metric_path.open("a", encoding="utf-8") as file:
                    file.write(json.dumps(row) + "\n")
                print(step_label + " " + json.dumps(row), flush=True)
                original_log(metrics, step)

            backend.log = record_metrics

        def _prepare_checkpointing(self) -> None:
            self.checkpointer = LoraOnlyCheckpointer(self)

    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("Exactly one CUDA GPU is required for Wan LoRA training")
    if not torch.cuda.is_bf16_supported():
        raise RuntimeError("CUDA BF16 support is required")
    if not MODEL.is_dir():
        raise FileNotFoundError(MODEL)
    if (output / "steps.jsonl").exists() or (output / "lora_weights").exists():
        raise FileExistsError("Wan LoRA training has already started in this output directory")

    dataset_config = prepare_finetrainers_data(
        MANIFEST, output, sample_id=sample_id, prompt_override=prompt_override,
    )
    args = BaseArgs()
    lora_config = SFTLowRankConfig()
    lora_config.rank = RANK
    lora_config.lora_alpha = RANK
    lora_config.target_modules = TARGET_MODULES
    args.register_args(lora_config)
    args.parallel_backend = ParallelBackendEnum.ACCELERATE
    args.model_name = "wan"
    args.training_type = "lora"
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
    args.rank = RANK
    args.lora_alpha = RANK
    args.target_modules = TARGET_MODULES
    args.lr = LR
    args.lr_scheduler = "constant"
    args.lr_warmup_steps = 0
    args.optimizer = "adamw"
    args.flow_weighting_scheme = "logit_normal"  # Official Wan example.
    args.enable_precomputation = True
    args.precomputation_items = 16
    args.precomputation_once = True
    args.precomputation_dir = str(output / "precomputed")
    args.validation_dataset_file = None
    args.checkpointing_steps = 0
    args.logging_steps = 1
    args.compile_modules = []
    args.compile_scopes = []
    args.transformer_dtype = torch.bfloat16
    args.text_encoder_dtype = torch.bfloat16
    args.vae_dtype = torch.bfloat16

    specification = WanModelSpecification(pretrained_model_name_or_path=str(MODEL))
    trainer = MeasuredWanTrainer(args, specification)
    with GPUMemorySampler() as sampler:
        trainer.run()
    duration = time.monotonic() - trainer.started
    saved = trainer.checkpointer.paths
    if tuple(saved) != checkpoint_steps:
        raise RuntimeError(f"Expected checkpoints {checkpoint_steps}, found {tuple(saved)}")
    for path in saved.values():
        if not (path / "pytorch_lora_weights.safetensors").is_file():
            raise RuntimeError(f"LoRA checkpoint was not saved: {path}")
    path = saved[steps]
    rows = [json.loads(line) for line in trainer.metric_path.read_text().splitlines()]
    if len(rows) != steps or [row["step"] for row in rows] != list(range(1, steps + 1)):
        raise RuntimeError(f"Expected {steps} logged optimizer steps, found {len(rows)}")
    summary = {
        "framework": "Hugging Face Finetrainers Wan SFT LoRA",
        "finetrainers_commit": FINETRAINERS_COMMIT,
        "manifest": str(MANIFEST), "model": str(MODEL),
        "training_sample_id": sample_id,
        "dataset_size": 1 if sample_id is not None else 16,
        "steps": steps, "batch_size": 1, "gradient_accumulation_steps": 1,
        "seed": 42, "dtype": "bf16", "rank": RANK, "lora_alpha": RANK,
        "target_module_pattern": TARGET_MODULES,
        "target_module_count": len(trainer.target_names),
        "trainable_parameters": trainer.trainable_parameters,
        "learning_rate": LR, "scheduler": "constant",
        "initial_loss": rows[0]["loss"], "final_loss": rows[-1]["loss"],
        "peak_gpu_vram_mib_observed": sampler.peak_mib,
        "peak_gpu_reserved_mib_logged": max(row["gpu_reserved_mib"] for row in rows),
        "runtime_seconds": round(duration, 2), "checkpoint": str(path),
        "checkpoints": {str(step): {
            "path": str(saved[step]),
            "adapter_bytes": (saved[step] / "pytorch_lora_weights.safetensors").stat().st_size,
        } for step in checkpoint_steps},
        "packages": {name: importlib.metadata.version(name) for name in (
            "torch", "diffusers", "transformers", "accelerate", "peft", "datasets", "decord")},
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary


def run_smoke(source: Path, output: Path, steps: int) -> dict:
    """Keep the original smoke CLI limited to one final adapter checkpoint."""
    return run_training(source, output, steps, (steps,), step_label="SMOKE_STEP")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--finetrainers-source", type=Path, default=Path("/tmp/finetrainers-wan-5c"))
    parser.add_argument("--output-dir", type=Path, default=OUTPUT)
    parser.add_argument("--steps", type=int, default=16)
    parser.add_argument("--prepare-only", action="store_true")
    options = parser.parse_args()
    if not 1 <= options.steps <= 20:
        parser.error("Smoke training must be limited to 1–20 optimizer steps")
    if options.prepare_only:
        print(prepare_finetrainers_data(MANIFEST, options.output_dir))
    else:
        print(json.dumps(run_smoke(options.finetrainers_source, options.output_dir, options.steps), indent=2))


if __name__ == "__main__":
    main()
