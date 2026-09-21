"""Cache exact 17-frame targets and unchanged native VACE O0 controls for M6D."""

from __future__ import annotations

import json
import time
from pathlib import Path
from types import SimpleNamespace

import torch

from training.wan.current_frame_metrics import source_frame_rgb
from training.wan.finetrainers_smoke import GPUMemorySampler
from training.wan.m6d_select_128 import OUT, digest
from training.wan.vace_action_adapter import ACTION_TO_ID
from training.wan.vace_action_train import OUTPUT as M6B1
from training.wan.vace_action_token_prototype import MODEL
from training.wan.vace_action_token_smoke import save_json
from training.wan.vace_hard_first_frame_smoke import (
    FRAMES, HEIGHT, WIDTH, PROMPT, _conditioning_images,
)


def run() -> dict:
    from diffusers import AutoencoderKLWan, WanVACEPipeline

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA VAE cache device is required")
    manifest_path = OUT / "subset_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    rows = manifest["records"]
    if len(rows) != 128 or not manifest["all_six_actions_present"]:
        raise RuntimeError("M6D subset has not passed selection")
    cache_dir = OUT / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    prompt_path = M6B1 / "train/cache/prompt.pt"
    prompt = torch.load(prompt_path, map_location="cpu", weights_only=True)
    if prompt.shape != (1, 512, 4096) or prompt.dtype != torch.bfloat16:
        raise RuntimeError("Existing generic-prompt embedding changed")
    index = {"subset_manifest_sha256": digest(manifest_path),
        "sample_ids": manifest["clip_ids"], "prompt": PROMPT,
        "prompt_embedding_path": str(prompt_path), "prompt_embedding_sha256": digest(prompt_path),
        "vae_dtype": "float32", "cached_dtype": "bfloat16",
        "native_vace_pixel_and_latent_mask_behavior_unchanged": True}
    index_path = cache_dir / "index.json"
    if index_path.exists() and json.loads(index_path.read_text()) != index:
        raise RuntimeError("Existing M6D cache index differs")
    save_json(index_path, index)
    started = time.monotonic()
    vae = AutoencoderKLWan.from_pretrained(
        str(MODEL), subfolder="vae", torch_dtype=torch.float32, local_files_only=True)
    pipe = WanVACEPipeline.from_pretrained(
        str(MODEL), vae=vae, text_encoder=None, transformer=None,
        torch_dtype=torch.bfloat16, local_files_only=True)
    if pipe.transformer is not None or pipe.text_encoder is not None:
        raise RuntimeError("Unexpected pretrained module loaded for latent cache")
    # Native helpers read only transformer.config.patch_size for spatial sizing.
    # Supply the exact pinned config field without loading or changing model weights.
    transformer_config = json.loads((MODEL / "transformer/config.json").read_text())
    if transformer_config["patch_size"] != [1, 2, 2]:
        raise RuntimeError("Pinned VACE patch size changed")
    pipe.transformer_2 = SimpleNamespace(config=SimpleNamespace(patch_size=transformer_config["patch_size"]))
    pipe.vae.to("cuda").eval().requires_grad_(False)
    mean = torch.tensor(pipe.vae.config.latents_mean, device="cuda").view(1, 16, 1, 1, 1)
    inv_std = 1 / torch.tensor(pipe.vae.config.latents_std, device="cuda").view(1, 16, 1, 1, 1)
    torch.cuda.reset_peak_memory_stats()
    encoded, reused = 0, 0
    with GPUMemorySampler() as sampler:
        for i, row in enumerate(rows):
            path = cache_dir / f"sample_{i:03d}.pt"
            expected_actions = [[ACTION_TO_ID[name] for name in row["raw_actions"]]]
            if not path.exists():
                if not all(Path(frame).is_file() for frame in row["rgb_frames"]):
                    raise FileNotFoundError(f"Missing exact source frames for {row['global_clip_id']}")
                real = (torch.stack([source_frame_rgb(frame) for frame in row["rgb_frames"]], dim=1)
                        .unsqueeze(0) * 2 - 1).to("cuda")
                video, masks = _conditioning_images({"source_o0": row["rgb_frames"][0]})
                conditioned, mask, references = pipe.preprocess_conditions(
                    video, masks, None, 1, HEIGHT, WIDTH, FRAMES, torch.float32, torch.device("cuda"))
                if (references != [[]] or conditioned.shape != (1, 3, 17, 256, 448)
                        or bool(torch.count_nonzero(mask[:, :, 0]))
                        or not bool(torch.all(mask[:, :, 1:] == 1))):
                    raise RuntimeError(f"Native O0 pixel mask changed: {row['global_clip_id']}")
                with torch.no_grad():
                    target = ((pipe.vae.encode(real).latent_dist.mode().float() - mean) * inv_std).to(torch.bfloat16)
                    video_latents = pipe.prepare_video_latents(
                        conditioned, mask, references, generator=None, device=torch.device("cuda"))
                    mask_latents = pipe.prepare_masks(mask, references)
                    if not bool(torch.all(mask_latents == 1)):
                        raise RuntimeError("Installed VACE latent-mask behavior changed; do not fix here")
                    control = torch.cat((video_latents, mask_latents), dim=1).to(torch.bfloat16)
                payload = {"target": target.cpu(), "control": control.cpu(),
                    "actions": torch.tensor(expected_actions, dtype=torch.long)}
                temporary = path.with_suffix(".partial.pt")
                torch.save(payload, temporary)
                temporary.replace(path)
                del real, target, conditioned, mask, video_latents, mask_latents, control, payload
                encoded += 1
            else:
                reused += 1
            sample = torch.load(path, map_location="cpu", weights_only=True)
            if (sample["target"].shape != (1, 16, 5, 32, 56)
                    or sample["control"].shape != (1, 96, 5, 32, 56)
                    or sample["actions"].tolist() != expected_actions
                    or not bool(torch.isfinite(sample["target"]).all())
                    or not bool(torch.isfinite(sample["control"]).all())
                    or not bool(torch.all(sample["control"][:, 32:] == 1))
                    or float(sample["control"][:, :16, 0].float().norm()) <= 0):
                raise RuntimeError(f"Invalid cached target/control/actions: {row['global_clip_id']}")
            print(f"M6D_CACHE {i+1}/128 {row['global_clip_id']}", flush=True)
            torch.cuda.empty_cache()
        summary = {"status": "complete", "samples": len(rows), "encoded": encoded, "reused": reused,
            "subset_manifest_sha256": digest(manifest_path),
            "cache_index_sha256": digest(index_path),
            "prompt_embedding_sha256": digest(prompt_path),
            "runtime_seconds": round(time.monotonic() - started, 3),
            "peak_gpu_allocated_mib": round(torch.cuda.max_memory_allocated() / 2**20, 1),
            "peak_gpu_reserved_mib": round(torch.cuda.max_memory_reserved() / 2**20, 1),
            "peak_gpu_vram_mib_observed": sampler.peak_mib,
            "frozen_vae": all(not p.requires_grad for p in pipe.vae.parameters()),
            "native_vace_mask_unchanged": True}
    save_json(cache_dir / "summary.json", summary)
    return summary


if __name__ == "__main__":
    print(json.dumps(run(), indent=2), flush=True)
