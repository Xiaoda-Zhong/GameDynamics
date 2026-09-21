#!/usr/bin/env python3
"""Run one reproducible pretrained Wan2.1 text-to-video baseline inference."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

try:
    from .gpu_check import require_cuda
except ImportError:
    from gpu_check import require_cuda


DEFAULT_CONFIG = Path(__file__).parent / "configs" / "baseline.yaml"
PRECISIONS = {"bf16", "fp16", "fp32"}


def load_config(path: str | Path) -> dict[str, Any]:
    try:
        import yaml
    except ImportError as error:
        raise RuntimeError(
            "PyYAML is required for Wan inference; install the GPU requirements"
        ) from error
    config_path = Path(path).expanduser().resolve()
    if not config_path.is_file():
        raise FileNotFoundError(f"inference config does not exist: {config_path}")
    with config_path.open("r", encoding="utf-8") as config_file:
        config = yaml.safe_load(config_file)
    if not isinstance(config, dict):
        raise ValueError(f"inference config must be a YAML mapping: {config_path}")
    return config


def validate_config(config: dict[str, Any]) -> None:
    required = {
        "model_path_or_id": str,
        "prompt": str,
        "output_path": str,
        "seed": int,
        "num_frames": int,
        "width": int,
        "height": int,
        "fps": (int, float),
        "inference_steps": int,
        "precision": str,
    }
    for field, expected_type in required.items():
        if field not in config or isinstance(config[field], bool) or not isinstance(
            config[field], expected_type
        ):
            raise ValueError(f"config field {field!r} is missing or has the wrong type")
    for field in ("model_path_or_id", "prompt", "output_path"):
        if not config[field].strip():
            raise ValueError(f"config field {field!r} must not be empty")
    if config["num_frames"] <= 0 or config["num_frames"] % 4 != 1:
        raise ValueError("Wan num_frames must be positive and satisfy num_frames = 4*k + 1")
    if config["width"] <= 0 or config["height"] <= 0:
        raise ValueError("width and height must be positive")
    if config["width"] % 16 or config["height"] % 16:
        raise ValueError("width and height must be divisible by 16")
    if config["fps"] <= 0 or config["inference_steps"] <= 0:
        raise ValueError("fps and inference_steps must be positive")
    if config["precision"] not in PRECISIONS:
        raise ValueError(f"precision must be one of {sorted(PRECISIONS)}")


def apply_overrides(config: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    result = dict(config)
    names = {
        "model": "model_path_or_id",
        "prompt": "prompt",
        "seed": "seed",
        "output": "output_path",
        "num_frames": "num_frames",
        "width": "width",
        "height": "height",
        "fps": "fps",
        "inference_steps": "inference_steps",
        "precision": "precision",
    }
    for argument_name, config_name in names.items():
        value = getattr(args, argument_name)
        if value is not None:
            result[config_name] = str(value) if config_name == "output_path" else value
    return result


def run_inference(config: dict[str, Any]) -> Path:
    """Load local/cached weights, generate frames, and write an MP4."""
    validate_config(config)
    gpu_report = require_cuda()
    if config["precision"] == "bf16" and not gpu_report["bf16_supported"]:
        raise RuntimeError(
            "precision is bf16, but CUDA device 0 does not report bf16 support; "
            "select fp16 or use a bf16-capable GPU"
        )

    try:
        import torch
        from diffusers import AutoencoderKLWan, WanPipeline
        from diffusers.schedulers import UniPCMultistepScheduler
        from diffusers.utils import export_to_video
    except ImportError as error:
        raise RuntimeError(
            "Wan inference dependencies are incomplete; install "
            "training/wan/requirements-gpu.txt"
        ) from error

    dtype = {
        "bf16": torch.bfloat16,
        "fp16": torch.float16,
        "fp32": torch.float32,
    }[config["precision"]]
    model = config["model_path_or_id"]
    load_options = {"local_files_only": True}
    try:
        vae = AutoencoderKLWan.from_pretrained(
            model,
            subfolder="vae",
            torch_dtype=torch.float32,
            **load_options,
        )
        pipeline = WanPipeline.from_pretrained(
            model,
            vae=vae,
            torch_dtype=dtype,
            **load_options,
        )
    except (OSError, ValueError) as error:
        raise RuntimeError(
            f"could not load Wan weights from {model!r} without downloading. "
            "Download the model explicitly or pass --model /path/to/local/weights."
        ) from error

    pipeline.scheduler = UniPCMultistepScheduler.from_config(
        pipeline.scheduler.config,
        flow_shift=float(config.get("flow_shift", 3.0)),
    )
    if bool(config.get("enable_model_cpu_offload", True)):
        pipeline.enable_model_cpu_offload()
    else:
        pipeline.to("cuda")

    generator = torch.Generator(device="cuda").manual_seed(config["seed"])
    result = pipeline(
        prompt=config["prompt"],
        negative_prompt=config.get("negative_prompt") or None,
        height=config["height"],
        width=config["width"],
        num_frames=config["num_frames"],
        num_inference_steps=config["inference_steps"],
        guidance_scale=float(config.get("guidance_scale", 5.0)),
        generator=generator,
    )
    frames = result.frames[0]
    if len(frames) != config["num_frames"]:
        raise RuntimeError(
            f"Wan returned {len(frames)} frames; expected {config['num_frames']}"
        )

    output_path = Path(config["output_path"]).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    export_to_video(frames, str(output_path), fps=config["fps"])
    run_metadata = {
        **config,
        "output_path": str(output_path),
        "gpu_environment": gpu_report,
    }
    metadata_path = output_path.with_suffix(output_path.suffix + ".json")
    metadata_path.write_text(
        json.dumps(run_metadata, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return output_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--model", help="local model path or cached model identifier")
    parser.add_argument("--prompt")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--num-frames", type=int)
    parser.add_argument("--width", type=int)
    parser.add_argument("--height", type=int)
    parser.add_argument("--fps", type=float)
    parser.add_argument("--inference-steps", type=int)
    parser.add_argument("--precision", choices=sorted(PRECISIONS))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    try:
        # Check hardware before importing YAML or any model packages so a local
        # CPU invocation fails for the actual prerequisite first.
        require_cuda()
        config = apply_overrides(load_config(args.config), args)
        output_path = run_inference(config)
    except (FileNotFoundError, RuntimeError, ValueError) as error:
        raise SystemExit(f"Wan inference failed: {error}") from error
    print(f"Wrote Wan baseline video: {output_path}")


if __name__ == "__main__":
    main()
