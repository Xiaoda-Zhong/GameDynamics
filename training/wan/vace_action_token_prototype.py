"""M6B-3 forward-only prototype of temporally masked action-token attention."""

from __future__ import annotations

import json
import time
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

REPO = Path(__file__).resolve().parents[2]
MODEL = REPO / "data/models/Wan2.1-VACE-1.3B-diffusers"
OUT = REPO / "data/experiments/m6b3_action_token_cross_attention"
M6B2 = REPO / "data/experiments/m6b2_action_signal_effect/protocol.json"
BLOCKS = (3, 11, 19, 27)
TOKENS_PER_LATENT = 16 * 28
FUTURE_TOKENS = 4 * TOKENS_PER_LATENT


def temporal_mask(device: torch.device) -> torch.Tensor:
    """True permits attention; observed latent 0 is omitted from queries."""
    mask = torch.zeros(1, 1, FUTURE_TOKENS, 16, dtype=torch.bool, device=device)
    for group in range(4):
        mask[:, :, group * TOKENS_PER_LATENT:(group + 1) * TOKENS_PER_LATENT,
             group * 4:(group + 1) * 4] = True
    return mask


class ActionCrossAttention(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.q_proj = nn.Linear(1536, 256)
        self.k_proj = nn.Linear(256, 256)
        self.v_proj = nn.Linear(256, 256)
        self.out_proj = nn.Linear(256, 1536)
        nn.init.zeros_(self.out_proj.weight)
        nn.init.zeros_(self.out_proj.bias)

    def forward(self, hidden: torch.Tensor, actions: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        if hidden.ndim != 3 or hidden.shape[1:] != (5 * TOKENS_PER_LATENT, 1536):
            raise ValueError(f"Unexpected video hidden shape: {tuple(hidden.shape)}")
        if actions.shape != (hidden.shape[0], 16, 256):
            raise ValueError(f"Unexpected action-token shape: {tuple(actions.shape)}")
        batch = hidden.shape[0]
        future = hidden[:, TOKENS_PER_LATENT:]
        q = self.q_proj(future).reshape(batch, FUTURE_TOKENS, 4, 64).transpose(1, 2)
        k = self.k_proj(actions).reshape(batch, 16, 4, 64).transpose(1, 2)
        v = self.v_proj(actions).reshape(batch, 16, 4, 64).transpose(1, 2)
        result = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, dropout_p=0.0)
        result = result.transpose(1, 2).reshape(batch, FUTURE_TOKENS, 256)
        residual = self.out_proj(result)
        return torch.cat((hidden[:, :TOKENS_PER_LATENT], future + residual), dim=1)


class ActionTokenAdapter(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.action_embedding = nn.Embedding(6, 256)
        self.temporal_embedding = nn.Embedding(16, 256)
        self.sites = nn.ModuleDict({str(i): ActionCrossAttention() for i in BLOCKS})

    def tokens(self, action_ids: torch.Tensor) -> torch.Tensor:
        if action_ids.ndim != 2 or action_ids.shape[1] != 16:
            raise ValueError("Expected action_ids [B,16]")
        positions = torch.arange(16, device=action_ids.device)
        return self.action_embedding(action_ids) + self.temporal_embedding(positions)[None]


def run() -> dict:
    from diffusers import UniPCMultistepScheduler, WanVACETransformer3DModel

    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("BF16 CUDA is required for the real VACE forward check")
    protocol = json.loads(M6B2.read_text())
    sample = torch.load(protocol["target_and_o0_control_cache_path"], map_location="cpu", weights_only=True)
    target = sample["target"].to("cuda")
    control = sample["control"].to("cuda")
    prompt = torch.load(protocol["text_embedding_cache_path"], map_location="cpu", weights_only=True).to("cuda")
    assert target.shape == (1, 16, 5, 32, 56) and control.shape == (1, 96, 5, 32, 56)
    adapter = ActionTokenAdapter().to("cuda", dtype=torch.bfloat16).eval()
    params = sum(p.numel() for p in adapter.parameters())
    assert params == 3_684_864, params
    mask = temporal_mask(torch.device("cuda"))
    assert mask.shape == (1, 1, 1792, 16) and int(mask.sum()) == 7168
    ids = {name: torch.tensor([values], dtype=torch.long, device="cuda")
           for name, values in protocol["action_ids"].items()}
    tokens = {name: adapter.tokens(value) for name, value in ids.items()}
    assert all(value.shape == (1, 16, 256) for value in tokens.values())
    assert not torch.equal(tokens["original"], tokens["reversed"])
    assert not torch.equal(tokens["original"], tokens["all_noop"])
    transformer = WanVACETransformer3DModel.from_pretrained(
        str(MODEL), subfolder="transformer", torch_dtype=torch.bfloat16, local_files_only=True
    ).to("cuda").eval().requires_grad_(False)
    assert len(transformer.blocks) == 30 and transformer.config.vace_layers == list(range(0, 30, 2))
    scheduler = UniPCMultistepScheduler.from_pretrained(str(MODEL), subfolder="scheduler", local_files_only=True)
    scheduler.set_timesteps(30)
    middle = next(row for row in protocol["selected_scheduler_positions"] if row["label"] == "middle")
    index = middle["index"]
    assert int(scheduler.timesteps[index]) == middle["timestep"]
    sigma = scheduler.sigmas[index].to(device="cuda", dtype=torch.float32)
    noise = torch.randn(target.shape, generator=torch.Generator(device="cuda").manual_seed(42),
                        device="cuda", dtype=torch.float32)
    noisy = ((1 - sigma) * target.float() + sigma * noise).to(torch.bfloat16)
    timestep = scheduler.timesteps[index].to("cuda").expand(1)
    scales = torch.ones(len(transformer.config.vace_layers), device="cuda", dtype=torch.bfloat16)

    def denoise() -> torch.Tensor:
        return transformer(hidden_states=noisy, timestep=timestep, encoder_hidden_states=prompt,
                           control_hidden_states=control, control_hidden_states_scale=scales,
                           return_dict=False)[0]

    started = time.monotonic()
    torch.cuda.reset_peak_memory_stats()
    with torch.inference_mode():
        baseline = denoise()
        baseline_peak = torch.cuda.max_memory_allocated()
        outputs = {}
        hook_shapes = {}
        for name, action_tokens in tokens.items():
            handles = []
            calls = {i: 0 for i in BLOCKS}
            try:
                for block_index in BLOCKS:
                    def hook(_module, _inputs, output, *, site=block_index):
                        calls[site] += 1
                        hook_shapes[str(site)] = list(output.shape)
                        return adapter.sites[str(site)](output, action_tokens, mask)
                    handles.append(transformer.blocks[block_index].register_forward_hook(hook))
                outputs[name] = denoise()
            finally:
                for handle in handles:
                    handle.remove()
            assert all(count == 1 for count in calls.values()), calls
            assert torch.equal(outputs[name], baseline), f"Zero-init output changed: {name}"
        assert torch.equal(outputs["original"], outputs["reversed"])
        assert torch.equal(outputs["original"], outputs["all_noop"])
    result = {
        "milestone": "M6B-3", "purpose": "forward-only architecture and shape prototype",
        "model": str(MODEL), "selected_main_block_indices": list(BLOCKS),
        "model_main_blocks": len(transformer.blocks), "vace_hint_block_indices": list(transformer.config.vace_layers),
        "input_shapes": {"target": list(target.shape), "o0_control": list(control.shape),
                         "prompt": list(prompt.shape), "action_ids": [1, 16],
                         "action_tokens": list(tokens["original"].shape)},
        "hook_video_shapes": hook_shapes, "attention_mask_shape": list(mask.shape),
        "attention_mask_true_entries": int(mask.sum()),
        "allowed_actions_by_latent": {str(i + 1): list(range(i * 4, i * 4 + 4)) for i in range(4)},
        "adapter_parameters": params, "per_site_parameters": sum(p.numel() for p in adapter.sites["3"].parameters()),
        "all_action_token_sequences_distinct": True, "zero_init_exact_equal_to_unhooked_vace": True,
        "zero_init_exact_equal_across_all_three_actions": True,
        "output_shape": list(baseline.shape), "scheduler_index": index,
        "timestep": int(scheduler.timesteps[index]), "sigma": float(sigma),
        "frozen_vace_trainable_parameters": sum(p.numel() for p in transformer.parameters() if p.requires_grad),
        "runtime_seconds": time.monotonic() - started,
        "baseline_peak_allocated_bytes": baseline_peak,
        "all_passes_peak_allocated_bytes": torch.cuda.max_memory_allocated(),
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "prototype.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


if __name__ == "__main__":
    print(json.dumps(run(), indent=2))
