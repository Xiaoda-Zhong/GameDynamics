#!/usr/bin/env python3
"""M6B-2 read-only action-feature and frozen-VACE denoiser effect audit."""

from __future__ import annotations

import hashlib
import json
import math
import time
from pathlib import Path

import torch

from training.wan.vace_action_adapter import VACEActionAdapter

REPO = Path(__file__).resolve().parents[2]
M6B1 = REPO / "data/experiments/m6b1_vace_action_tiny"
OUTPUT = REPO / "data/experiments/m6b2_action_signal_effect"
MODEL = REPO / "data/models/Wan2.1-VACE-1.3B-diffusers"
PAIRS = (("original", "reversed"), ("original", "all_noop"), ("reversed", "all_noop"))
STEPS = (100, 200, 400, 800)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _read_protocol() -> dict:
    protocol = json.loads((OUTPUT / "protocol.json").read_text())
    if (protocol["source_sample_id"] != "my_way_home_episode_0028_clip_000011"
            or protocol["m6b1_official_classification"] != "C. EXPLICIT ACTION CONTROL FAIL"
            or protocol["generic_prompt"] != "A first-person gameplay view in a retro 3D stone maze."
            or protocol["seed"] != 42):
        raise RuntimeError("M6B-2 fixed protocol changed")
    for key in ("cache_index", "target_and_o0_control_cache", "text_embedding_cache"):
        path = Path(protocol[f"{key}_path"])
        if not path.is_file() or _sha256(path) != protocol[f"{key}_sha256"]:
            raise RuntimeError(f"Cached fixed input changed: {key}")
    for step in STEPS:
        path = Path(protocol["adapter_checkpoint_paths"][str(step)])
        if not path.is_file() or _sha256(path) != protocol["adapter_checkpoint_sha256"][str(step)]:
            raise RuntimeError(f"Adapter checkpoint changed: step {step}")
    if json.loads((M6B1 / "eval/metrics.json").read_text())["classification"] != protocol["m6b1_official_classification"]:
        raise RuntimeError("M6B-1 official result changed")
    return protocol


def _stats(value: torch.Tensor) -> dict:
    flat = value.float().reshape(-1)
    if not bool(torch.isfinite(flat).all()):
        raise FloatingPointError("Nonfinite audit vector")
    return {"l2": float(torch.linalg.vector_norm(flat)),
            "mean_absolute": float(flat.abs().mean()),
            "max_absolute": float(flat.abs().max())}


def _pair(a: torch.Tensor, b: torch.Tensor) -> dict:
    x, y = a.float().reshape(-1), b.float().reshape(-1)
    delta = x - y
    xnorm, ynorm = torch.linalg.vector_norm(x), torch.linalg.vector_norm(y)
    denominator = (xnorm + ynorm) / 2
    dot = torch.dot(x, y)
    return {"absolute_l2": float(torch.linalg.vector_norm(delta)),
            "relative_l2": float(torch.linalg.vector_norm(delta) / denominator) if denominator > 0 else None,
            "cosine_similarity": float(dot / (xnorm * ynorm)) if xnorm > 0 and ynorm > 0 else None,
            "mean_absolute_difference": float(delta.abs().mean()),
            "max_absolute_difference": float(delta.abs().max())}


def _features(adapter: VACEActionAdapter, actions: dict[str, torch.Tensor]) -> tuple[dict, dict]:
    projected = {name: adapter(ids).float() for name, ids in actions.items()}
    for feature in projected.values():
        if tuple(feature.shape) != (1, 5, 1536) or not bool(torch.all(feature[:, 0] == 0)):
            raise RuntimeError("Adapter feature shape or zero observed-state slot changed")
    features = {}
    for name, feature in projected.items():
        features[name] = {"future_positions": {str(k): _stats(feature[:, k]) for k in range(1, 5)},
                          "global_future": _stats(feature[:, 1:])}
    comparisons = {}
    for left, right in PAIRS:
        key = f"{left}_vs_{right}"
        comparisons[key] = {"future_positions": {str(k): _pair(projected[left][:, k], projected[right][:, k]) for k in range(1, 5)},
                            "global_future": _pair(projected[left][:, 1:], projected[right][:, 1:])}
    return {"by_condition": features, "pairwise": comparisons}, projected


def _strength(base_patch: torch.Tensor, projected: dict[str, torch.Tensor]) -> dict:
    if tuple(base_patch.shape) != (1, 1536, 5, 16, 28):
        raise RuntimeError("Unexpected frozen noisy patch-embedding shape")
    result = {}
    for name, feature in projected.items():
        ratios = []
        for k in range(1, 5):
            # Match the actual broadcast addition across every spatial patch.
            action_bias = feature[:, k].reshape(1, 1536, 1, 1).expand(1, 1536, 16, 28)
            numerator = torch.linalg.vector_norm(action_bias.float())
            denominator = torch.linalg.vector_norm(base_patch[:, :, k].float())
            if denominator <= 0:
                raise RuntimeError("Zero frozen patch-embedding norm")
            ratios.append(float(numerator / denominator))
        result[name] = {"future_positions": {str(k): ratios[k - 1] for k in range(1, 5)},
                        "mean": sum(ratios) / 4, "range": [min(ratios), max(ratios)]}
    return result


def _output_pair(a: torch.Tensor, b: torch.Tensor) -> dict:
    if tuple(a.shape) != (1, 16, 5, 32, 56) or a.shape != b.shape:
        raise RuntimeError("Unexpected frozen VACE output shape")
    return {"global_all_positions": _pair(a, b),
            "global_future_positions": _pair(a[:, :, 1:], b[:, :, 1:]),
            "temporal_positions": {str(k): _pair(a[:, :, k], b[:, :, k]) for k in range(5)}}


def _load_adapter(step: int, device: torch.device) -> VACEActionAdapter:
    path = M6B1 / "checkpoints" / f"step_{step:04d}" / "adapter.pt"
    adapter = VACEActionAdapter()
    state = torch.load(path, map_location="cpu", weights_only=True)
    adapter.load_state_dict(state, strict=True)
    if sum(p.numel() for p in adapter.parameters()) != 198_336:
        raise RuntimeError("Checkpoint adapter parameter count changed")
    adapter.requires_grad_(False)
    return adapter.to(device=device, dtype=torch.bfloat16).eval()


def run() -> dict:
    protocol = _read_protocol()
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("The existing BF16 CUDA GPU is required")
    from diffusers import UniPCMultistepScheduler, WanVACETransformer3DModel
    device = torch.device("cuda")
    started = time.monotonic()
    sample = torch.load(protocol["target_and_o0_control_cache_path"], map_location="cpu", weights_only=True)
    prompt = torch.load(protocol["text_embedding_cache_path"], map_location="cpu", weights_only=True).to(device)
    target = sample["target"].to(device)
    control = sample["control"].to(device)
    if (tuple(target.shape) != (1, 16, 5, 32, 56)
            or tuple(control.shape) != (1, 96, 5, 32, 56)
            or tuple(prompt.shape) != (1, 512, 4096)
            or sample["actions"].tolist() != [protocol["action_ids"]["original"]]
            or not bool(torch.all(control[:, 32:] == 1))
            or float(control[:, :16, 0].float().norm()) <= 0):
        raise RuntimeError("Cached real target, generic text, or native O0 conditioning changed")
    actions = {name: torch.tensor([ids], device=device, dtype=torch.long) for name, ids in protocol["action_ids"].items()}
    scheduler = UniPCMultistepScheduler.from_pretrained(str(MODEL), subfolder="scheduler", local_files_only=True)
    scheduler.set_timesteps(30)
    for row in protocol["selected_scheduler_positions"]:
        i = row["index"]
        if int(scheduler.timesteps[i]) != row["timestep"] or abs(float(scheduler.sigmas[i]) - row["sigma"]) > 1e-8:
            raise RuntimeError("Actual scheduler differs from the fixed audit positions")
    noise = torch.randn(target.shape, generator=torch.Generator(device=device).manual_seed(42), device=device, dtype=torch.float32)
    transformer = WanVACETransformer3DModel.from_pretrained(str(MODEL), subfolder="transformer", torch_dtype=torch.bfloat16, local_files_only=True)
    original_dtypes = {name: parameter.dtype for name, parameter in transformer.named_parameters()}
    transformer.to(device).eval().requires_grad_(False)
    if any(parameter.dtype != original_dtypes[name] for name, parameter in transformer.named_parameters()):
        raise RuntimeError("Moving frozen VACE to CUDA changed a protected parameter dtype")
    base_versions = {name: parameter._version for name, parameter in transformer.named_parameters()}
    control_scale = torch.ones(len(transformer.config.vace_layers), device=device, dtype=torch.bfloat16)
    torch.cuda.reset_peak_memory_stats()
    metrics = {"protocol_path": str(OUTPUT / "protocol.json"),
               "fixed_input_checks": {"target_shape": list(target.shape), "control_shape": list(control.shape),
                    "prompt_shape": list(prompt.shape), "native_latent_mask_all_ones_unchanged": True,
                    "o0_inactive_control_latent_l2": float(torch.linalg.vector_norm(control[:, :16, 0].float())),
                    "initial_noise_sha256": hashlib.sha256(noise.cpu().numpy().tobytes()).hexdigest(),
                    "transformer_parameter_count": sum(p.numel() for p in transformer.parameters()),
                    "transformer_requires_grad_parameters": sum(p.numel() for p in transformer.parameters() if p.requires_grad)},
               "step_800": {}, "checkpoint_trajectory_middle": {}}
    outputs_middle_800 = None
    with torch.inference_mode():
        adapter800 = _load_adapter(800, device)
        feature800, projected800 = _features(adapter800, actions)
        metrics["step_800"]["action_features"] = feature800
        for row in protocol["selected_scheduler_positions"]:
            label, index = row["label"], row["index"]
            sigma = scheduler.sigmas[index].to(device=device, dtype=torch.float32)
            noisy = ((1 - sigma) * target.float() + sigma * noise).to(dtype=torch.bfloat16)
            timestep = scheduler.timesteps[index].to(device).expand(1)
            base_patch = transformer.patch_embedding(noisy)
            strength = _strength(base_patch, projected800)
            output = {}
            for name, ids in actions.items():
                with adapter800.inject(transformer, ids):
                    output[name] = transformer(
                        hidden_states=noisy, timestep=timestep,
                        encoder_hidden_states=prompt, control_hidden_states=control,
                        control_hidden_states_scale=control_scale, return_dict=False,
                    )[0].float()
                if transformer.patch_embedding._forward_hooks:
                    raise RuntimeError("Action hook remained installed after a transformer call")
            baseline = transformer(
                hidden_states=noisy, timestep=timestep,
                encoder_hidden_states=prompt, control_hidden_states=control,
                control_hidden_states_scale=control_scale, return_dict=False,
            )[0].float()
            pairwise = {f"{left}_vs_{right}": _output_pair(output[left], output[right]) for left, right in PAIRS}
            versus_baseline = {name: _output_pair(value, baseline) for name, value in output.items()}
            if label == "middle":
                with adapter800.inject(transformer, actions["original"]):
                    repeat = transformer(
                        hidden_states=noisy, timestep=timestep,
                        encoder_hidden_states=prompt, control_hidden_states=control,
                        control_hidden_states_scale=control_scale, return_dict=False,
                    )[0].float()
                repeat_max = float((output["original"] - repeat).abs().max())
                outputs_middle_800 = pairwise
            else:
                repeat_max = None
            metrics["step_800"][label] = {
                "scheduler_index": index, "timestep": row["timestep"], "sigma": row["sigma"],
                "same_noisy_latent_all_conditions": True,
                "noisy_latent_shape": list(noisy.shape),
                "base_patch_embedding_global_future_l2": _stats(base_patch[:, :, 1:])["l2"],
                "relative_conditioning_strength": strength,
                "relative_conditioning_strength_mean": sum(item["mean"] for item in strength.values()) / 3,
                "relative_conditioning_strength_range": [
                    min(item["range"][0] for item in strength.values()),
                    max(item["range"][1] for item in strength.values())],
                "denoiser_pairwise": pairwise,
                "denoiser_vs_no_adapter": versus_baseline,
                "repeat_original_max_absolute_difference": repeat_max,
            }
        if outputs_middle_800 is None:
            raise RuntimeError("Missing step-800 middle-timestep comparison")
        middle = next(row for row in protocol["selected_scheduler_positions"] if row["label"] == "middle")
        sigma = scheduler.sigmas[middle["index"]].to(device=device, dtype=torch.float32)
        noisy = ((1 - sigma) * target.float() + sigma * noise).to(dtype=torch.bfloat16)
        timestep = scheduler.timesteps[middle["index"]].to(device).expand(1)
        for step in STEPS:
            if step == 800:
                feature, pairwise = feature800, outputs_middle_800
            else:
                adapter = _load_adapter(step, device)
                feature, _ = _features(adapter, actions)
                output = {}
                for name, ids in actions.items():
                    with adapter.inject(transformer, ids):
                        output[name] = transformer(
                            hidden_states=noisy, timestep=timestep,
                            encoder_hidden_states=prompt, control_hidden_states=control,
                            control_hidden_states_scale=control_scale, return_dict=False,
                        )[0].float()
                pairwise = {f"{left}_vs_{right}": _output_pair(output[left], output[right]) for left, right in PAIRS}
                del adapter
            metrics["checkpoint_trajectory_middle"][str(step)] = {
                "action_features": feature,
                "denoiser_pairwise": pairwise,
            }
    if (any(parameter.grad is not None for parameter in transformer.parameters())
            or any(parameter._version != base_versions[name] for name, parameter in transformer.named_parameters())):
        raise RuntimeError("Frozen VACE transformer changed during audit")
    metrics["validity"] = {"frozen_transformer_parameters_unchanged": True,
                           "no_gradients_or_optimizer": True, "no_training_or_video_generation": True,
                           "identical_action_comparison_inputs": True,
                           "runtime_seconds": round(time.monotonic() - started, 3),
                           "peak_gpu_allocated_mib": round(torch.cuda.max_memory_allocated() / 2**20, 1)}
    output_path = OUTPUT / "metrics.json"
    output_path.write_text(json.dumps(metrics, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({"metrics_path": str(output_path),
                      "feature_global_cosines": {pair: data["global_future"]["cosine_similarity"] for pair, data in feature800["pairwise"].items()},
                      "denoiser_future_relative_l2": {label: {pair: data["global_future_positions"]["relative_l2"] for pair, data in result["denoiser_pairwise"].items()} for label, result in metrics["step_800"].items() if label != "action_features"},
                      "validity": metrics["validity"]}, indent=2), flush=True)
    return metrics


if __name__ == "__main__":
    run()
