"""Deterministic source-frame preprocessing and RGB identity metrics for M6A."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from decord import VideoReader, cpu
from PIL import Image


HEIGHT = 256
WIDTH = 448
FRAMES = 17


def source_frame_rgb(path: str | Path) -> torch.Tensor:
    """Return RGB in [0, 1] after the pinned trainer's bicubic preprocessing."""
    with Image.open(path) as image:
        rgb = np.asarray(image.convert("RGB"), dtype=np.float32)
    tensor = torch.from_numpy(rgb).permute(2, 0, 1).contiguous() / 127.5 - 1.0
    resized = F.interpolate(
        tensor.unsqueeze(0), size=(HEIGHT, WIDTH), mode="bicubic", align_corners=False,
    )[0]
    return ((resized + 1.0) / 2.0).clamp(0.0, 1.0).contiguous()


def generated_frames_rgb(path: str | Path) -> list[torch.Tensor]:
    video = VideoReader(str(path), ctx=cpu(0))
    if len(video) != FRAMES or tuple(video[0].shape) != (HEIGHT, WIDTH, 3):
        raise ValueError(f"Expected a {FRAMES}-frame {WIDTH}x{HEIGHT} video: {path}")
    frames = []
    for index in range(FRAMES):
        frame = video[index]
        array = frame.asnumpy() if hasattr(frame, "asnumpy") else frame.numpy()
        frames.append(torch.from_numpy(array.copy()).permute(2, 0, 1).float() / 255.0)
    return frames


def ssim_rgb(first: torch.Tensor, second: torch.Tensor) -> float:
    """Channel-averaged 11x11 Gaussian SSIM, sigma 1.5, valid pixels, range 1."""
    if first.shape != second.shape or tuple(first.shape) != (3, HEIGHT, WIDTH):
        raise ValueError("SSIM inputs must be matching 3x256x448 RGB tensors")
    positions = torch.arange(11, dtype=torch.float64) - 5
    kernel_1d = torch.exp(-(positions ** 2) / (2 * 1.5 ** 2))
    kernel_1d /= kernel_1d.sum()
    kernel = torch.outer(kernel_1d, kernel_1d).to(torch.float32)
    weight = kernel.expand(3, 1, 11, 11).contiguous()
    x, y = first.unsqueeze(0).float(), second.unsqueeze(0).float()
    mu_x = F.conv2d(x, weight, groups=3)
    mu_y = F.conv2d(y, weight, groups=3)
    mu_x2, mu_y2, mu_xy = mu_x.square(), mu_y.square(), mu_x * mu_y
    sigma_x2 = F.conv2d(x * x, weight, groups=3) - mu_x2
    sigma_y2 = F.conv2d(y * y, weight, groups=3) - mu_y2
    sigma_xy = F.conv2d(x * y, weight, groups=3) - mu_xy
    c1, c2 = 0.01 ** 2, 0.03 ** 2
    value = ((2 * mu_xy + c1) * (2 * sigma_xy + c2)) / (
        (mu_x2 + mu_y2 + c1) * (sigma_x2 + sigma_y2 + c2)
    )
    result = float(value.mean())
    if not np.isfinite(result):
        raise FloatingPointError("SSIM is nonfinite")
    return result


def mean_absolute_rgb_error(first: torch.Tensor, second: torch.Tensor) -> float:
    if first.shape != second.shape:
        raise ValueError("RGB error inputs must have matching dimensions")
    return float((first.float() - second.float()).abs().mean() * 255.0)


def select_distinct_pairs(source_maps: list[dict], count: int = 5) -> list[dict]:
    """Greedily choose disjoint O0 pairs by largest mean absolute RGB difference."""
    if len(source_maps) != 16 or count < 1 or count > 8:
        raise ValueError("Expected 16 sources and one to eight disjoint pairs")
    frames = [source_frame_rgb(row["source_rgb_frames"][0]) for row in source_maps]
    candidates = [
        (mean_absolute_rgb_error(frames[i], frames[j]), i, j)
        for i in range(16) for j in range(i + 1, 16)
    ]
    candidates.sort(key=lambda item: (-item[0], item[1], item[2]))
    used = set()
    pairs = []
    for mae, i, j in candidates:
        if i in used or j in used:
            continue
        pairs.append({
            "a_index": i,
            "b_index": j,
            "a_sample_id": source_maps[i]["global_clip_id"],
            "b_sample_id": source_maps[j]["global_clip_id"],
            "source_o0_pair_mae_rgb_0_255": mae,
            "source_o0_pair_ssim": ssim_rgb(frames[i], frames[j]),
        })
        used.update((i, j))
        if len(pairs) == count:
            break
    if len(pairs) != count or any(pair["source_o0_pair_ssim"] >= 0.9 for pair in pairs):
        raise ValueError("Could not find five distinct initial-observation pairs")
    return pairs
