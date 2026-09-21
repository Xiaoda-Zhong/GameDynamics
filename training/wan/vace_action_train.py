#!/usr/bin/env python3
"""M6B-1: train only the 198,336-parameter action adapter on 16 real clips."""

from __future__ import annotations

import argparse
import collections
import gc
import hashlib
import json
import math
import random
import time
from pathlib import Path

import torch
import torch.nn.functional as F

from training.wan.current_frame_metrics import source_frame_rgb
from training.wan.finetrainers_smoke import GPUMemorySampler
from training.wan.vace_action_adapter import ACTION_NAMES, ACTION_TO_ID, VACEActionAdapter
from training.wan.vace_hard_first_frame_smoke import (
    FRAMES, HEIGHT, WIDTH, MODEL, PROMPT, _conditioning_images, _model_ready,
)

ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / "data/experiments/m6b1_vace_action_tiny"
MANIFEST = ROOT / "data/wan_training/my_way_home/overfit_0016/manifest.jsonl"
CHECKPOINT_STEPS = (100, 200, 400, 800)
STEPS, SMOKE_STEPS, SEED, LR = 800, 32, 42, 1e-3


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def _records() -> list[dict]:
    rows = [json.loads(line) for line in MANIFEST.read_text().splitlines() if line.strip()]
    if len(rows) != 16 or len({r["global_clip_id"] for r in rows}) != 16:
        raise ValueError("Expected exactly 16 distinct existing samples")
    for row in rows:
        if (row.get("dataset_split") != "train" or row.get("num_frames") != 17
                or row.get("num_actions") != 16 or len(row["rgb_frames"]) != 17
                or len(row["raw_actions"]) != 16 or set(row["raw_actions"]) - set(ACTION_NAMES)
                or not all(Path(path).is_file() for path in row["rgb_frames"])):
            raise ValueError(f"Invalid real TRAIN sample: {row.get('global_clip_id')}")
    counts = collections.Counter(action for row in rows for action in row["raw_actions"])
    if any(counts[name] == 0 for name in ACTION_NAMES):
        raise RuntimeError(f"Absent action ID; stop before training: {counts}")
    protocol = json.loads((OUTPUT / "protocol.json").read_text())
    expected = {str(i): counts[name] for i, name in enumerate(ACTION_NAMES)}
    if (protocol["manifest_sha256"] != _sha256(MANIFEST)
            or protocol["transition_counts_by_id"] != expected
            or protocol["generic_prompt"] != PROMPT):
        raise RuntimeError("Predeclared protocol, action counts, or manifest changed")
    return rows


def preflight() -> dict:
    rows = _records()
    sanity_path = OUTPUT / "zero_init/sanity.json"
    if not sanity_path.is_file():
        raise FileNotFoundError("Run zero-init generation sanity before training")
    sanity = json.loads(sanity_path.read_text())
    if (not sanity.get("passed") or sanity.get("max_absolute_pixel_difference") != 0
            or not sanity.get("identical_sha256")):
        raise RuntimeError("Saved zero-init sanity failed")
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("BF16 CUDA is required")
    if not _model_ready():
        raise FileNotFoundError("Pinned local VACE weights are incomplete")
    adapter = VACEActionAdapter()
    count = sum(p.numel() for p in adapter.parameters() if p.requires_grad)
    if count != 198_336:
        raise RuntimeError(f"Adapter has {count} trainable parameters, expected 198336")
    return {"samples": len(rows), "transitions": sum(len(r["raw_actions"]) for r in rows),
            "action_counts_by_id": json.loads((OUTPUT / "protocol.json").read_text())["transition_counts_by_id"],
            "adapter_trainable_parameters": count, "zero_init_sanity": str(sanity_path)}


def _cache_latents(pipe, rows: list[dict], device: torch.device) -> tuple[torch.Tensor, list[dict]]:
    """Derive normalized real targets and unchanged native VACE control once per clip."""
    directory = OUTPUT / "train/cache"
    directory.mkdir(parents=True, exist_ok=True)
    index_path = directory / "index.json"
    cache_index = {"manifest_sha256": _sha256(MANIFEST), "sample_ids": [r["global_clip_id"] for r in rows],
             "prompt": PROMPT, "vae_dtype": "float32", "cached_dtype": "bfloat16",
             "native_vace_mask_unchanged": True}
    if index_path.exists() and json.loads(index_path.read_text()) != cache_index:
        raise RuntimeError("Existing latent cache has different sources or settings")
    prompt_path = directory / "prompt.pt"
    if not prompt_path.exists():
        pipe.text_encoder.to(device).eval()
        with torch.no_grad():
            prompt, negative = pipe.encode_prompt(
                prompt=PROMPT, negative_prompt=None, do_classifier_free_guidance=False,
                num_videos_per_prompt=1, max_sequence_length=512, device=device,
            )
        if negative is not None or tuple(prompt.shape) != (1, 512, 4096):
            raise RuntimeError(f"Unexpected generic-prompt embedding shape: {tuple(prompt.shape)}")
        torch.save(prompt.detach().to("cpu", dtype=torch.bfloat16), prompt_path)
        pipe.text_encoder.to("cpu")
        torch.cuda.empty_cache()
    prompt = torch.load(prompt_path, map_location="cpu", weights_only=True)
    if tuple(prompt.shape) != (1, 512, 4096):
        raise RuntimeError("Invalid cached generic-prompt embedding")

    pipe.vae.to(device).eval()
    mean = torch.tensor(pipe.vae.config.latents_mean, device=device).view(1, 16, 1, 1, 1)
    inv_std = 1 / torch.tensor(pipe.vae.config.latents_std, device=device).view(1, 16, 1, 1, 1)
    samples = []
    for index, row in enumerate(rows):
        path = directory / f"sample_{index:02d}.pt"
        if not path.exists():
            real = (torch.stack([source_frame_rgb(p) for p in row["rgb_frames"]], dim=1).unsqueeze(0) * 2 - 1).to(device)
            video, masks = _conditioning_images({"source_o0": row["rgb_frames"][0]})
            conditioned, mask, references = pipe.preprocess_conditions(
                video, masks, None, 1, HEIGHT, WIDTH, FRAMES, torch.float32, device,
            )
            if (references != [[]] or tuple(conditioned.shape) != (1, 3, 17, 256, 448)
                    or bool(torch.count_nonzero(mask[:, :, 0]))
                    or not bool(torch.all(mask[:, :, 1:] == 1))):
                raise RuntimeError("Native M6A-4 O0 conditioning changed")
            with torch.no_grad():
                target = ((pipe.vae.encode(real).latent_dist.mode().float() - mean) * inv_std).to(torch.bfloat16)
                video_latents = pipe.prepare_video_latents(conditioned, mask, references, generator=None, device=device)
                mask_latents = pipe.prepare_masks(mask, references)
                if not bool(torch.all(mask_latents == 1)):
                    raise RuntimeError("Installed VACE latent-mask behavior changed; stop without fixing it")
                control = torch.cat((video_latents, mask_latents), dim=1).to(torch.bfloat16)
            if (tuple(target.shape) != (1, 16, 5, 32, 56)
                    or tuple(control.shape) != (1, 96, 5, 32, 56)
                    or not bool(torch.isfinite(target).all())
                    or not bool(torch.isfinite(control).all())
                    or float(video_latents[:, :16, 0].float().norm()) <= 0):
                raise RuntimeError(f"Invalid real/control latents: {row['global_clip_id']}")
            payload = {"target": target.cpu(), "control": control.cpu(),
                       "actions": torch.tensor([[ACTION_TO_ID[a] for a in row["raw_actions"]]], dtype=torch.long)}
            temporary = path.with_suffix(".partial.pt")
            torch.save(payload, temporary)
            temporary.replace(path)
            del real, target, video_latents, mask_latents, control, conditioned, mask
            torch.cuda.empty_cache()
        sample = torch.load(path, map_location="cpu", weights_only=True)
        if (tuple(sample["target"].shape) != (1, 16, 5, 32, 56)
                or tuple(sample["control"].shape) != (1, 96, 5, 32, 56)
                or sample["actions"].tolist() != [[ACTION_TO_ID[a] for a in row["raw_actions"]]]):
            raise RuntimeError(f"Cached sample differs: {row['global_clip_id']}")
        samples.append(sample)
        print(f"M6B1_CACHE {index + 1}/16 {row['global_clip_id']}", flush=True)
    if not index_path.exists():
        _json(index_path, cache_index)
    pipe.vae.to("cpu")
    torch.cuda.empty_cache()
    return prompt.to(device), samples


def _order() -> list[int]:
    generator = torch.Generator().manual_seed(SEED)
    order = [i for _ in range(50) for i in torch.randperm(16, generator=generator).tolist()]
    if len(order) != STEPS or collections.Counter(order) != {i: 50 for i in range(16)}:
        raise RuntimeError("Sample order must be exactly balanced")
    return order


def _gradient_norms(adapter: VACEActionAdapter) -> tuple[float, dict[str, float]]:
    parts = {}
    for name, parameter in adapter.named_parameters():
        gradient = parameter.grad
        if gradient is None or not bool(torch.isfinite(gradient).all()):
            raise FloatingPointError(f"Missing or nonfinite adapter gradient: {name}")
        parts[name] = float(torch.linalg.vector_norm(gradient.float()))
    total = math.sqrt(sum(value * value for value in parts.values()))
    if not math.isfinite(total) or total <= 0:
        raise FloatingPointError("Adapter gradient is zero or nonfinite")
    return total, parts


def _save_adapter(adapter: VACEActionAdapter, step: int, last_step: dict) -> None:
    path = OUTPUT / "checkpoints" / f"step_{step:04d}" / "adapter.pt"
    path.parent.mkdir(parents=True, exist_ok=True)
    state = {name: tensor.detach().cpu().contiguous() for name, tensor in adapter.state_dict().items()}
    if set(state) != {"embedding.weight", "projection.weight", "projection.bias"}:
        raise RuntimeError("Checkpoint contains a pretrained weight")
    temporary = path.with_suffix(".partial.pt")
    torch.save(state, temporary)
    temporary.replace(path)
    _json(path.with_suffix(".json"), {"step": step, "adapter_parameters": 198_336,
          "keys": sorted(state), "sha256": _sha256(path), "last_step": last_step})


def run() -> dict:
    check = preflight()
    rows = _records()
    train_dir = OUTPUT / "train"
    train_dir.mkdir(parents=True, exist_ok=True)
    log_path = train_dir / "steps.jsonl"
    if log_path.exists() and log_path.stat().st_size:
        raise FileExistsError("Training log exists; refusing to overwrite or silently resume")
    _json(train_dir / "preflight.json", check)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    random.seed(SEED)
    device = torch.device("cuda")
    started = time.monotonic()
    from diffusers import AutoencoderKLWan, UniPCMultistepScheduler, WanVACEPipeline
    vae = AutoencoderKLWan.from_pretrained(str(MODEL), subfolder="vae", torch_dtype=torch.float32, local_files_only=True)
    pipe = WanVACEPipeline.from_pretrained(str(MODEL), vae=vae, torch_dtype=torch.bfloat16, local_files_only=True)
    if (pipe.__class__.__name__ != "WanVACEPipeline"
            or pipe.transformer.__class__.__name__ != "WanVACETransformer3DModel"
            or hasattr(pipe.transformer, "peft_config")):
        raise RuntimeError("Expected unfinetuned native VACE")
    pretrained = (pipe.transformer, pipe.vae, pipe.text_encoder)
    for module in pretrained:
        module.requires_grad_(False)
        module.eval()
    prompt, samples = _cache_latents(pipe, rows, device)
    for module in pretrained:
        if any(p.requires_grad or p.grad is not None for p in module.parameters()):
            raise RuntimeError("A pretrained parameter is trainable")
    loaded_dtypes = {name: parameter.dtype for name, parameter in pipe.transformer.named_parameters()}
    transformer = pipe.transformer.to(device=device).eval()
    if any(parameter.dtype != loaded_dtypes[name] for name, parameter in transformer.named_parameters()):
        raise RuntimeError("Moving VACE to CUDA changed a pretrained parameter dtype")
    if not any(dtype == torch.float32 for dtype in loaded_dtypes.values()):
        raise RuntimeError("Expected native protected FP32 VACE components")
    transformer.enable_gradient_checkpointing()
    base_versions = {name: p._version for name, p in transformer.named_parameters()}
    del pipe, vae
    gc.collect()
    torch.cuda.empty_cache()
    adapter = VACEActionAdapter().to(device=device, dtype=torch.float32).train()
    if sum(p.numel() for p in adapter.parameters() if p.requires_grad) != 198_336:
        raise RuntimeError("Adapter-only trainable parameter count changed")
    optimizer = torch.optim.AdamW(adapter.parameters(), lr=LR)
    if {id(p) for group in optimizer.param_groups for p in group["params"]} != {id(p) for p in adapter.parameters()}:
        raise RuntimeError("Optimizer includes non-adapter parameters")
    scheduler = UniPCMultistepScheduler.from_pretrained(str(MODEL), subfolder="scheduler", local_files_only=True)
    scheduler.set_timesteps(1000)
    if scheduler.config.prediction_type != "flow_prediction" or scheduler.config.flow_shift != 3.0:
        raise RuntimeError("Expected pinned flow-prediction scheduler with shift 3.0")
    timesteps = scheduler.timesteps.to(device)
    sigmas = scheduler.sigmas[:1000].to(device)
    order = _order()
    timestep_indices = torch.randint(0, 1000, (STEPS,), generator=torch.Generator().manual_seed(SEED + 1)).tolist()
    noise_generator = torch.Generator(device=device).manual_seed(SEED + 2)
    control_scale = torch.ones(len(transformer.config.vace_layers), device=device, dtype=torch.bfloat16)
    counts = collections.Counter()
    last = None
    with GPUMemorySampler() as sampler, log_path.open("w", encoding="utf-8") as log:
        for step, index in enumerate(order, start=1):
            step_started = time.monotonic()
            sample = samples[index]
            target = sample["target"].to(device)
            control = sample["control"].to(device)
            actions = sample["actions"].to(device)
            timestep_index = timestep_indices[step - 1]
            sigma = sigmas[timestep_index].float()
            timestep = timesteps[timestep_index].expand(1)
            noise = torch.randn(target.shape, device=device, dtype=torch.bfloat16, generator=noise_generator)
            noisy = ((1 - sigma) * target.float() + sigma * noise.float()).to(torch.bfloat16)
            velocity = noise.float() - target.float()
            optimizer.zero_grad(set_to_none=True)
            with adapter.inject(transformer, actions):
                prediction = transformer(
                    hidden_states=noisy, timestep=timestep,
                    encoder_hidden_states=prompt, control_hidden_states=control,
                    control_hidden_states_scale=control_scale, return_dict=False,
                )[0]
            if tuple(prediction.shape) != (1, 16, 5, 32, 56):
                raise RuntimeError(f"Unexpected VACE prediction shape: {tuple(prediction.shape)}")
            loss = F.mse_loss(prediction.float()[:, :, 1:], velocity[:, :, 1:])
            if not bool(torch.isfinite(loss)):
                raise FloatingPointError(f"Nonfinite future-only loss at step {step}")
            loss.backward()
            grad_norm, components = _gradient_norms(adapter)
            optimizer.step()
            torch.cuda.synchronize()
            counts[index] += 1
            last = {"step": step, "sample_index": index, "sample_id": rows[index]["global_clip_id"],
                    "loss": float(loss.detach()), "lr": optimizer.param_groups[0]["lr"],
                    "adapter_grad_norm": grad_norm, "gradient_norms": components,
                    "timestep_index": timestep_index, "sigma": float(sigma),
                    "step_runtime_seconds": round(time.monotonic() - step_started, 3),
                    "elapsed_seconds": round(time.monotonic() - started, 3),
                    "gpu_allocated_mib": round(torch.cuda.memory_allocated() / 2**20, 1),
                    "gpu_reserved_mib": round(torch.cuda.memory_reserved() / 2**20, 1),
                    "gpu_peak_allocated_mib": round(torch.cuda.max_memory_allocated() / 2**20, 1)}
            log.write(json.dumps(last) + "\n")
            log.flush()
            print("M6B1_STEP " + json.dumps(last), flush=True)
            if step == SMOKE_STEPS:
                if counts != {i: 2 for i in range(16)}:
                    raise RuntimeError("Smoke gate sample schedule was not balanced")
                if (any(p.grad is not None for p in transformer.parameters())
                        or any(p._version != base_versions[name] for name, p in transformer.named_parameters())):
                    raise RuntimeError("A frozen VACE parameter changed during smoke gate")
                _json(train_dir / "smoke_gate.json", {
                    "passed": True, "steps": 32, "finite_losses": True,
                    "finite_gradients": True, "nonzero_adapter_gradients": True,
                    "no_oom": True, "pretrained_weights_unchanged": True,
                    "sample_counts": {str(i): counts[i] for i in range(16)}, "last_step": last})
            if step in CHECKPOINT_STEPS:
                _save_adapter(adapter, step, last)
            del target, control, actions, noise, noisy, velocity, prediction, loss
        if counts != {i: 50 for i in range(16)}:
            raise RuntimeError(f"800-step sample counts are unbalanced: {counts}")
        if (any(p.grad is not None for p in transformer.parameters())
                or any(p._version != base_versions[name] for name, p in transformer.named_parameters())):
            raise RuntimeError("A frozen VACE parameter changed during training")
        summary = {"optimizer_steps": STEPS, "smoke_gate_passed": True,
                   "adapter_trainable_parameters": 198_336,
                   "pretrained_transformer_weights_unchanged": True,
                   "pretrained_vae_text_frozen": True,
                   "sample_counts": {str(i): counts[i] for i in range(16)},
                   "runtime_seconds": round(time.monotonic() - started, 3),
                   "peak_gpu_allocated_mib": round(torch.cuda.max_memory_allocated() / 2**20, 1),
                   "peak_gpu_vram_mib_observed": sampler.peak_mib,
                   "checkpoint_steps": list(CHECKPOINT_STEPS), "last_step": last}
    _json(train_dir / "summary.json", summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args()
    if args.preflight_only:
        print(json.dumps(preflight(), indent=2))
    else:
        print(json.dumps(run(), indent=2), flush=True)


if __name__ == "__main__":
    main()
