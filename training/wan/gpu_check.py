#!/usr/bin/env python3
"""Report whether the current Python environment can run Wan inference."""

from __future__ import annotations

import argparse
import json
from typing import Any


def get_gpu_environment() -> dict[str, Any]:
    report: dict[str, Any] = {
        "torch_version": None,
        "cuda_available": False,
        "cuda_version": None,
        "gpu_name": None,
        "vram_gib": None,
        "bf16_supported": False,
    }
    try:
        import torch
    except ImportError as error:
        report["error"] = f"PyTorch is not installed: {error}"
        return report

    report["torch_version"] = torch.__version__
    report["cuda_version"] = torch.version.cuda
    report["cuda_available"] = bool(torch.cuda.is_available())
    if not report["cuda_available"]:
        report["error"] = "CUDA is not available to PyTorch"
        return report

    try:
        properties = torch.cuda.get_device_properties(0)
        report["gpu_name"] = torch.cuda.get_device_name(0)
        report["vram_gib"] = round(properties.total_memory / (1024**3), 2)
        report["bf16_supported"] = bool(torch.cuda.is_bf16_supported())
    except (RuntimeError, AssertionError) as error:
        report["error"] = f"could not query CUDA device 0: {error}"
    return report


def require_cuda() -> dict[str, Any]:
    report = get_gpu_environment()
    if not report["cuda_available"] or report.get("error"):
        reason = report.get("error", "CUDA device 0 is unavailable")
        raise RuntimeError(
            f"Wan inference requires an NVIDIA CUDA GPU: {reason}. "
            "Run training/wan/gpu_check.py in the GPU environment for details."
        )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--require-cuda",
        action="store_true",
        help="return a nonzero exit code unless CUDA is usable",
    )
    args = parser.parse_args()
    report = get_gpu_environment()
    print(json.dumps(report, indent=2))
    if args.require_cuda and (not report["cuda_available"] or report.get("error")):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
