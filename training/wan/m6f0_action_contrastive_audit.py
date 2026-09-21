"""One-sample, no-update feasibility audit for an action-ranking diffusion loss."""

from __future__ import annotations

import collections
import json
import math
import time
from pathlib import Path

import torch
import torch.nn.functional as F

from training.wan.m6d_select_128 import OUT as M6D, ROOT, digest
from training.wan.m6d_train_128 import PROMPT, load_cache
from training.wan.vace_action_adapter import ACTION_TO_ID
from training.wan.vace_action_token_prototype import ActionTokenAdapter, BLOCKS, MODEL, temporal_mask

OUT = ROOT / "data/experiments/m6f0_action_contrastive_audit"
CHECKPOINT = M6D / "checkpoints/step_6400/adapter.pt"
MARGIN = 0.05  # Provisional gradient-sanity value, not selected by held-out evaluation.
LAMBDA = 1.0  # Provisional gradient-sanity value, not selected by held-out evaluation.
SWAP = {
    ACTION_TO_ID["TURN_LEFT"]: ACTION_TO_ID["TURN_RIGHT"],
    ACTION_TO_ID["TURN_RIGHT"]: ACTION_TO_ID["TURN_LEFT"],
    ACTION_TO_ID["MOVE_FORWARD_LEFT"]: ACTION_TO_ID["MOVE_FORWARD_RIGHT"],
    ACTION_TO_ID["MOVE_FORWARD_RIGHT"]: ACTION_TO_ID["MOVE_FORWARD_LEFT"],
}


def _grad_snapshot(adapter: ActionTokenAdapter) -> dict[str, torch.Tensor]:
    result = {}
    for name, parameter in adapter.named_parameters():
        if parameter.grad is None or not bool(torch.isfinite(parameter.grad).all()):
            raise FloatingPointError(f"Missing or nonfinite action-adapter gradient: {name}")
        result[name] = parameter.grad.detach().float().cpu().clone()
    return result


def _norm(parts: dict[str, torch.Tensor]) -> float:
    return math.sqrt(sum(float(value.square().sum()) for value in parts.values()))


def run() -> dict:
    from diffusers import UniPCMultistepScheduler, WanVACETransformer3DModel

    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("BF16 CUDA is required for the M6F-0 audit")
    sidecar = json.loads((CHECKPOINT.parent / "adapter.json").read_text())
    if sidecar["step"] != 6400 or sidecar["sha256"] != digest(CHECKPOINT):
        raise RuntimeError("The frozen official step-6400 adapter changed")
    prompt_cpu, samples, rows = load_cache()
    unchanged = [i for i, row in enumerate(rows)
                 if all(ACTION_TO_ID[action] not in SWAP for action in row["raw_actions"])]
    candidates = [i for i in range(len(rows)) if i not in unchanged]
    if len(rows) != 128 or not candidates:
        raise RuntimeError("No left/right counterfactual is available in the fixed subset")
    sample_index = candidates[0]
    sample = samples[sample_index]
    correct_cpu = sample["actions"]
    wrong_cpu = correct_cpu.clone()
    for source, destination in SWAP.items():
        wrong_cpu[correct_cpu == source] = destination
    changed = (correct_cpu != wrong_cpu).nonzero(as_tuple=False)[:, 1].tolist()
    if not changed or correct_cpu.shape != (1, 16) or wrong_cpu.shape != (1, 16):
        raise RuntimeError("Counterfactual action IDs did not change")
    if any(wrong_cpu[correct_cpu == fixed].ne(fixed).any()
           for fixed in (ACTION_TO_ID["MOVE_FORWARD"], ACTION_TO_ID["NOOP"])):
        raise RuntimeError("MOVE_FORWARD or NOOP was altered")

    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    adapter = ActionTokenAdapter().to("cuda", dtype=torch.float32).train()
    adapter.load_state_dict(torch.load(CHECKPOINT, map_location="cpu", weights_only=True), strict=True)
    adapter_versions = {name: parameter._version for name, parameter in adapter.named_parameters()}
    if sum(p.numel() for p in adapter.parameters() if p.requires_grad) != 3_684_864:
        raise RuntimeError("Action adapter parameter count changed")
    transformer = WanVACETransformer3DModel.from_pretrained(
        str(MODEL), subfolder="transformer", torch_dtype=torch.bfloat16, local_files_only=True
    ).to("cuda").eval().requires_grad_(False)
    transformer.enable_gradient_checkpointing()
    if any(p.requires_grad or p.grad is not None for p in transformer.parameters()):
        raise RuntimeError("A pretrained VACE parameter is trainable")
    base_versions = {name: parameter._version for name, parameter in transformer.named_parameters()}
    scheduler = UniPCMultistepScheduler.from_pretrained(
        str(MODEL), subfolder="scheduler", local_files_only=True)
    scheduler.set_timesteps(1000)
    if scheduler.config.prediction_type != "flow_prediction" or scheduler.config.flow_shift != 3.0:
        raise RuntimeError("The M6D fixed flow scheduler changed")

    device = torch.device("cuda")
    target = sample["target"].to(device)
    control = sample["control"].to(device)
    prompt = prompt_cpu.to(device)
    correct = correct_cpu.to(device)
    wrong = wrong_cpu.to(device)
    if (target.shape != (1, 16, 5, 32, 56) or control.shape != (1, 96, 5, 32, 56)
            or prompt.shape != (1, 512, 4096)
            or not bool(torch.all(control[:, 32:] == 1))
            or float(control[:, :16, 0].float().norm()) <= 0):
        raise RuntimeError("Cached target, native O0 control, or generic text changed")
    timestep_index = int(torch.randint(
        0, 1000, (1,), generator=torch.Generator().manual_seed(43)).item())
    sigma = scheduler.sigmas[timestep_index].to(device).float()
    timestep = scheduler.timesteps[timestep_index].to(device).expand(1)
    noise = torch.randn(target.shape, device=device, dtype=torch.bfloat16,
                        generator=torch.Generator(device="cuda").manual_seed(44))
    noisy = ((1 - sigma) * target.float() + sigma * noise.float()).to(torch.bfloat16)
    velocity = noise.float() - target.float()
    mask = temporal_mask(device)
    control_scale = torch.ones(len(transformer.config.vace_layers), device=device, dtype=torch.bfloat16)

    def loss_for(actions: torch.Tensor, *, backward: bool) -> tuple[float, dict[int, int]]:
        # Every tensor other than actions is captured once above and reused by reference.
        tokens = adapter.tokens(actions)
        calls = collections.Counter()
        handles = []
        try:
            for block in BLOCKS:
                def hook(_module, _inputs, hidden, *, site=block, action_tokens=tokens):
                    calls[site] += 1
                    with torch.autocast("cuda", dtype=torch.bfloat16):
                        return adapter.sites[str(site)](hidden, action_tokens, mask)
                handles.append(transformer.blocks[block].register_forward_hook(hook))
            prediction = transformer(hidden_states=noisy, timestep=timestep,
                encoder_hidden_states=prompt, control_hidden_states=control,
                control_hidden_states_scale=control_scale, return_dict=False)[0]
            if prediction.shape != target.shape:
                raise RuntimeError("The VACE denoiser output shape changed")
            loss = F.mse_loss(prediction.float()[:, :, 1:], velocity[:, :, 1:])
            if not bool(torch.isfinite(loss)):
                raise FloatingPointError("Nonfinite future-only denoising loss")
            value = float(loss.detach())
            if backward:
                loss.backward()
        finally:
            for handle in handles:
                handle.remove()
        required = 2 if backward else 1
        if any(calls[block] != required for block in BLOCKS):
            raise RuntimeError(f"Action cross-attention hook missed a pass: {calls}")
        return value, dict(calls)

    started = time.monotonic()
    torch.cuda.reset_peak_memory_stats()
    with torch.no_grad():
        l_correct, correct_forward_calls = loss_for(correct, backward=False)
        l_wrong, wrong_forward_calls = loss_for(wrong, backward=False)
    rank = max(0.0, MARGIN + l_correct - l_wrong)
    total = l_correct + LAMBDA * rank
    hinge_active = rank > 0

    adapter.zero_grad(set_to_none=True)
    l_correct_grad, correct_backward_calls = loss_for(correct, backward=True)
    grad_correct = _grad_snapshot(adapter)
    adapter.zero_grad(set_to_none=True)
    l_wrong_grad, wrong_backward_calls = loss_for(wrong, backward=True)
    grad_wrong = _grad_snapshot(adapter)
    if abs(l_correct_grad - l_correct) > 1e-6 or abs(l_wrong_grad - l_wrong) > 1e-6:
        raise RuntimeError("The paired measurement and gradient forwards disagree")
    difference = {name: grad_correct[name] - grad_wrong[name] for name in grad_correct}
    combined = {name: grad_correct[name] + (LAMBDA * difference[name] if hinge_active else 0)
                for name in grad_correct}
    norms = {"correct_loss": _norm(grad_correct), "wrong_loss": _norm(grad_wrong),
             "contrastive_difference": _norm(difference),
             "ranking_term": LAMBDA * _norm(difference) if hinge_active else 0.0,
             "total_loss": _norm(combined)}
    if any(not math.isfinite(value) for value in norms.values()) or norms["total_loss"] <= 0:
        raise FloatingPointError("Nonfinite or zero action-adapter gradient")
    torch.cuda.synchronize()
    if (any(p.requires_grad or p.grad is not None or p._version != base_versions[name]
            for name, p in transformer.named_parameters())
            or any(p._version != adapter_versions[name] for name, p in adapter.named_parameters())):
        raise RuntimeError("Pretrained VACE or adapter weights changed during the audit")
    output = {
        "milestone": "M6F-0", "status": "one-sample paired-forward gradient audit; zero optimizer steps",
        "sample_index": sample_index, "sample_id": rows[sample_index]["global_clip_id"],
        "episode_id": rows[sample_index]["episode_name"], "subset_manifest_sha256": digest(M6D / "subset_manifest.json"),
        "adapter_checkpoint": str(CHECKPOINT), "adapter_checkpoint_sha256": digest(CHECKPOINT),
        "adapter_parameters": 3_684_864,
        "correct_action_ids": correct_cpu[0].tolist(), "wrong_action_ids": wrong_cpu[0].tolist(),
        "changed_action_positions": changed, "unchanged_counterfactual_sample_indices": unchanged,
        "unchanged_counterfactual_sample_count": len(unchanged),
        "shared_inputs": {"real_future_latent_shape": list(target.shape),
            "native_o0_control_shape": list(control.shape), "text_shape": list(prompt.shape),
            "noise_shape": list(noise.shape), "noisy_latent_shape": list(noisy.shape),
            "same_tensor_objects_for_both_action_forwards": True,
            "future_only_loss_latent_positions": [1, 2, 3, 4],
            "native_vace_mask_behavior_unchanged": True,
            "o0_control_latent_norm": float(control[:, :16, 0].float().norm()),
            "timestep_index": timestep_index, "timestep": int(timestep.item()),
            "sigma": float(sigma), "noise_seed": 44,
            "prompt": PROMPT},
        "provisional_margin": MARGIN, "provisional_lambda": LAMBDA,
        "loss_correct": l_correct, "loss_wrong": l_wrong,
        "loss_wrong_minus_correct": l_wrong - l_correct,
        "ranking_loss": rank, "total_loss": total, "ranking_hinge_active": hinge_active,
        "adapter_gradient_norms": norms,
        "gradient_computation": "Separate correct/wrong backward passes with unchanged weights; total gradient is the exact algebraic combination gc + lambda*(gc-gw) while the hinge is active",
        "adapter_gradient_all_finite": True, "adapter_total_gradient_nonzero": True,
        "action_hook_calls": {"correct_value": correct_forward_calls,
            "wrong_value": wrong_forward_calls,
            "correct_gradient": correct_backward_calls,
            "wrong_gradient": wrong_backward_calls},
        "pretrained_vace_frozen_and_unchanged": True,
        "vae_and_text_encoder_not_loaded_or_trained": True,
        "adapter_weights_unchanged": True, "optimizer_steps": 0,
        "runtime_seconds": round(time.monotonic() - started, 3),
        "peak_gpu_allocated_mib": round(torch.cuda.max_memory_allocated() / 2**20, 1),
        "peak_gpu_reserved_mib": round(torch.cuda.max_memory_reserved() / 2**20, 1),
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "audit.json").write_text(json.dumps(output, indent=2, allow_nan=False) + "\n")
    report = ["# M6F-0: action-contrastive objective feasibility audit", "",
        "**Status: engineering-feasible on one paired sample.** No optimizer step or evaluation run.", "",
        f"Real TRAIN sample: `{output['sample_id']}` (subset index {sample_index}, {output['episode_id']}). "
        f"The deterministic left/right swap changes {len(changed)}/16 action IDs at positions {changed}.", "",
        "## Paired objective", "",
        "Both forwards reuse the exact same cached real target, native VACE O0 control, generic text embedding, "
        "diffusion timestep, BF16 noise tensor, and noisy latent. Only the 16 integer action IDs differ. "
        "Loss is the existing M6D MSE on future temporal latent positions 1–4.", "",
        f"- `L_correct = {l_correct:.8f}`; `L_wrong = {l_wrong:.8f}`; `L_wrong - L_correct = {l_wrong-l_correct:.8f}`.",
        f"- Provisional `margin = {MARGIN}` and `lambda = {LAMBDA}`: `L_rank = {rank:.8f}`; `L_total = {total:.8f}`. "
        "These values are for gradient sanity only and were not chosen from held-out evaluation.",
        f"- Ranking hinge active: **{hinge_active}**. Adapter gradient norms: correct {norms['correct_loss']:.6f}, "
        f"wrong {norms['wrong_loss']:.6f}, contrastive difference {norms['contrastive_difference']:.6f}, "
        f"total {norms['total_loss']:.6f}. All adapter gradients were finite; total gradient was nonzero. "
        "The total gradient was assembled algebraically from separate correct/wrong backward passes with unchanged weights.", "",
        "The existing adapter prefers the correct actions by only "
        f"{l_wrong-l_correct:.8f} MSE on this sample/timestep. This verifies that the objective can carry a signal; "
        "it does not establish robust action semantics or held-out generalization.", "",
        "## Engineering validity", "",
        f"Timestep index {timestep_index} (timestep {int(timestep.item())}, sigma {float(sigma):.6f}); "
        "seed-44 noise. The pretrained VACE transformer was frozen and unchanged. "
        "VAE and text encoder were not loaded; their existing cached outputs were reused. "
        "Native O0 and mask behavior, including the previously observed all-one latent mask, stayed unchanged. "
        "No optimizer was constructed and adapter weights were unchanged.", "",
        f"Peak allocated/reserved VRAM: {output['peak_gpu_allocated_mib']:.1f}/{output['peak_gpu_reserved_mib']:.1f} MiB; "
        f"paired audit runtime {output['runtime_seconds']:.3f} s.", "",
        "## Counterfactual eligibility", "",
        f"{len(unchanged)} of 128 fixed TRAIN clips are unchanged by the prescribed swap. "
        "For those clips, `A_wrong == A_correct`, so the ranking term is the constant margin and its gradient is zero. "
        "A future objective should skip the contrastive term for those clips while retaining the standard denoising loss; "
        "do not invent a different action sequence for them. This audit made no training change.", "",
        f"Numerical record: `{OUT / 'audit.json'}`. Frozen adapter: `{CHECKPOINT}`.", ""]
    (OUT / "report.md").write_text("\n".join(report))
    return output


if __name__ == "__main__":
    print(json.dumps(run(), indent=2, allow_nan=False), flush=True)
