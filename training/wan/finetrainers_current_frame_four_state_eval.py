#!/usr/bin/env python3
"""M6A-3: four-state current-frame Control LoRA evaluation."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw

from training.wan.current_frame_metrics import (
    FRAMES, HEIGHT, WIDTH, generated_frames_rgb, mean_absolute_rgb_error,
    source_frame_rgb, ssim_rgb,
)
from training.wan.finetrainers_current_frame_eval import (
    FLOW_SHIFT, FPS, GUIDANCE, SOURCE, STEPS, _control_latent, _load_pipeline,
)
from training.wan.finetrainers_current_frame_train import ALPHA, LR, RANK
from training.wan.finetrainers_smoke import GPUMemorySampler, MODEL, _require_official_source


ROOT = Path("/root/autodl-tmp/GameDynamics_starter/data/experiments/m6a3_four_state_1600")
PROMPT = "A first-person gameplay view in a retro 3D stone maze."
CHECKPOINTS = (200, 400, 800, 1600)
FINAL_SEEDS = (42, 43, 44)
IDENTIFICATION_MINIMUM = 10
OWN_SSIM_MINIMUM = 0.60
MARGIN_MINIMUM = 0.05
A_ID = "my_way_home_episode_0028_clip_000011"


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _source_rows(root: Path) -> list[dict]:
    selected = _read_json(root / "selected_states.json")
    states = selected["states"]
    if selected["generic_prompt"] != PROMPT:
        raise ValueError("Selected-state protocol changed the generic prompt")
    if len(states) != 4 or states[0]["sample_id"] != A_ID:
        raise ValueError("Selected states must contain A followed by three distinct states")
    if [state["state"] for state in states] != list("ABCD"):
        raise ValueError("Selected-state labels must be A through D")
    selected_ids = [state["sample_id"] for state in states]
    if len(set(selected_ids)) != 4:
        raise ValueError("Selected state IDs are duplicated")
    source_map = root / "train_1600/derived_dataset/source_map.jsonl"
    mapped = [json.loads(line) for line in source_map.read_text(encoding="utf-8").splitlines()]
    by_id = {row["global_clip_id"]: row for row in mapped}
    if len(mapped) != 4 or len(by_id) != 4 or set(by_id) != set(selected_ids):
        raise ValueError("Training data differs from the four preselected states")
    ordered = [by_id[sample_id] for sample_id in selected_ids]
    prompts = (root / "train_1600/derived_dataset/prompt.txt").read_text(encoding="utf-8").splitlines()
    if len(prompts) != 4 or any(prompt != PROMPT for prompt in prompts):
        raise ValueError("Training prompts must all be the exact generic M6A-3 prompt")
    for state, row in zip(states, ordered):
        if (row["prompt"] != PROMPT or row["source_rgb_frames"][0] != state["source_o0"]
                or row["source_rgb_frames"] != state["source_rgb_frames"]
                or row["raw_actions"] != state["raw_actions"]
                or len(row["source_rgb_frames"]) != FRAMES or len(row["raw_actions"]) != FRAMES - 1
                or not all(isinstance(action, str) for action in row["raw_actions"])):
            raise ValueError(f"State data differs from selection or source format: {state['sample_id']}")
    return ordered


def _protocol(root: Path) -> dict:
    train = root / "train_1600"
    summary = _read_json(train / "summary.json")
    if (summary["steps"] != 1600 or summary["dataset_size"] != 4
            or summary["model"] != str(MODEL) or summary["dtype"] != "bf16"
            or summary["batch_size"] != 1 or summary["gradient_accumulation_steps"] != 1
            or summary["rank"] != RANK or summary["lora_alpha"] != ALPHA
            or summary["target_module_count"] != 241 or summary["learning_rate"] != LR
            or summary["frame_conditioning_type"] != "index" or summary["frame_conditioning_index"] != 0
            or summary["frames"] != FRAMES or summary["height"] != HEIGHT or summary["width"] != WIDTH
            or summary["seed"] != 42 or set(map(int, summary["checkpoint_paths"])) != set(CHECKPOINTS)):
        raise ValueError("M6A-3 training summary differs from the declared model and schedule")
    manifest = Path(summary["source_manifest"])
    if hashlib.sha256(manifest.read_bytes()).hexdigest() != summary["source_manifest_sha256"]:
        raise RuntimeError("Source manifest changed since training")
    rows = _source_rows(root)
    if summary["training_sample_ids"] != [row["global_clip_id"] for row in rows]:
        raise ValueError("Training summary state order differs from selected states")
    o0s = [source_frame_rgb(row["source_rgb_frames"][0]) for row in rows]
    matrix = [[ssim_rgb(a, b) for b in o0s] for a in o0s]
    selected_matrix = _read_json(root / "selected_states.json")["pairwise_source_o0_ssim"]
    if len(selected_matrix) != 4 or any(len(row) != 4 for row in selected_matrix):
        raise ValueError("Selected-state pairwise SSIM matrix has the wrong shape")
    if any(abs(matrix[i][j] - selected_matrix[i][j]) > 1e-6
           for i in range(4) for j in range(4)):
        raise ValueError("Selected-state O0 SSIM matrix differs from real source frames")
    return {
        "experiment": "M6A-3 four-state current-state conditioning",
        "selection_path": str(root / "selected_states.json"),
        "source_manifest": str(manifest),
        "source_manifest_sha256": summary["source_manifest_sha256"],
        "states": [{
            "label": chr(ord("A") + index), "sample_id": row["global_clip_id"],
            "source_o0": row["source_rgb_frames"][0], "source_video": row["video_path"],
        } for index, row in enumerate(rows)],
        "source_o0_pairwise_ssim": matrix,
        "prompt": PROMPT,
        "checkpoints": {str(step): summary["checkpoint_paths"][str(step)]["path"] for step in CHECKPOINTS},
        "untrained_control_definition": (
            "Expanded patch embedding and real O0 control hook, with all Control LoRA adapters "
            "disabled; added base patch channels are zero initialized"
        ),
        "seed_42_trajectory": [None, *CHECKPOINTS],
        "final_checkpoint_seeds": list(FINAL_SEEDS),
        "height": HEIGHT, "width": WIDTH, "frames": FRAMES, "fps": FPS,
        "inference_steps": STEPS, "guidance_scale": GUIDANCE, "flow_shift": FLOW_SHIFT,
        "negative_prompt": None, "lora_scale": 1.0,
        "model_dtype": "bfloat16", "inference_vae_dtype": "float32",
        "source_preprocessing": "RGB PNG; /127.5-1; bicubic 256x448 align_corners=False",
        "metric_input": "saved video frames decoded from MP4",
        "ssim": "11x11 Gaussian sigma=1.5, valid pixels, RGB-channel mean, range=1",
        "mae": "Mean absolute RGB error in 0-255 units",
        "classification_thresholds": {
            "state_identification_correct_of_12": IDENTIFICATION_MINIMUM,
            "mean_own_state_ssim": OWN_SSIM_MINIMUM,
            "median_separation_margin": MARGIN_MINIMUM,
        },
    }


def _conditions() -> list[tuple[int | None, int, int]]:
    conditions = [(step, state, 42) for step in (None, *CHECKPOINTS) for state in range(4)]
    conditions += [(1600, state, seed) for seed in (43, 44) for state in range(4)]
    assert len(conditions) == 28
    return conditions


def _video_name(step: int | None, state: int, seed: int) -> str:
    checkpoint = "untrained" if step is None else f"step_{step:04d}"
    return f"{checkpoint}_state_{chr(ord('A') + state)}_seed_{seed}.mp4"


def run_generate(root: Path = ROOT) -> dict:
    _require_official_source(SOURCE)
    from diffusers.utils import export_to_video
    from finetrainers.patches.dependencies.diffusers.control import control_channel_concat

    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("BF16 CUDA GPU required")
    eval_dir = root / "eval"
    eval_dir.mkdir(parents=True, exist_ok=True)
    protocol = _protocol(root)
    protocol_path = eval_dir / "protocol.json"
    if protocol_path.exists() and _read_json(protocol_path) != protocol:
        raise ValueError("Existing M6A-3 evaluation protocol differs")
    protocol_path.write_text(json.dumps(protocol, indent=2) + "\n", encoding="utf-8")
    started = time.monotonic()
    with GPUMemorySampler() as sampler:
        pipe, loaded = _load_pipeline(root / "train_1600", CHECKPOINTS, active_step=1600)
        layer = pipe.transformer.patch_embedding
        if torch.count_nonzero(layer.base_layer.weight[:, 16:]).item() != 0:
            raise RuntimeError("Untrained control patch channels are not zero initialized")
        pipe.vae.to("cuda")
        controls = [
            _control_latent(pipe, source_frame_rgb(state["source_o0"]), (FRAMES - 1) // 4 + 1)
            for state in protocol["states"]
        ]
        pipe.vae.to("cpu")
        torch.cuda.empty_cache()
        pipe.enable_model_cpu_offload()
        fixed_latents = {
            seed: pipe.prepare_latents(
                1, 16, HEIGHT, WIDTH, FRAMES, torch.bfloat16, torch.device("cuda"),
                torch.Generator(device="cuda").manual_seed(seed),
            ) for seed in FINAL_SEEDS
        }
        videos = []
        for step, state_index, seed in _conditions():
            if step is None:
                pipe.disable_lora()
                if not layer.disable_adapters:
                    raise RuntimeError("Untrained control baseline has active LoRA weights")
            else:
                pipe.enable_lora()
                adapter = f"step_{step}"
                pipe.set_adapters([adapter], [1.0])
                if layer.disable_adapters or layer.active_adapters != [adapter]:
                    raise RuntimeError(f"Wrong active Control LoRA adapter at step {step}")
            control = controls[state_index].to("cuda")
            path = eval_dir / "videos" / _video_name(step, state_index, seed)
            metadata_path = path.with_suffix(".json")
            metadata = {
                "checkpoint_step": step, "state_label": protocol["states"][state_index]["label"],
                "sample_id": protocol["states"][state_index]["sample_id"],
                "source_o0": protocol["states"][state_index]["source_o0"],
                "prompt": PROMPT, "seed": seed,
                "control_latent_norm": float(control.float().norm()),
                "control_nonzero_temporal_indices": [0], "control_hook_active": True,
                "adapter_disabled": step is None,
                "video_path": str(path), "frame_count_verified": FRAMES,
            }
            path.parent.mkdir(parents=True, exist_ok=True)
            if path.exists():
                if not metadata_path.is_file() or _read_json(metadata_path) != metadata:
                    raise ValueError(f"Existing video metadata differs: {path}")
                generated_frames_rgb(path)
            else:
                with torch.no_grad(), control_channel_concat(
                    pipe.transformer, ["hidden_states"], [control], dims=[1],
                ):
                    result = pipe(
                        prompt=PROMPT, negative_prompt=None,
                        height=HEIGHT, width=WIDTH, num_frames=FRAMES,
                        num_inference_steps=STEPS, guidance_scale=GUIDANCE,
                        generator=torch.Generator(device="cuda").manual_seed(seed),
                        latents=fixed_latents[seed].clone(),
                    )
                if len(result.frames[0]) != FRAMES:
                    raise RuntimeError(f"Wrong generated frame count: {path}")
                temporary = path.with_name(path.stem + ".partial.mp4")
                export_to_video(result.frames[0], str(temporary), fps=FPS)
                generated_frames_rgb(temporary)
                temporary.replace(path)
                metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
                del result
                torch.cuda.empty_cache()
            videos.append(metadata)
            print("GENERATED " + json.dumps(metadata), flush=True)
        output = {
            "protocol": str(protocol_path), "checkpoints_loaded": loaded,
            "videos": videos, "video_count": len(videos),
            "runtime_seconds": round(time.monotonic() - started, 2),
            "peak_gpu_vram_mib_observed": sampler.peak_mib,
        }
    (eval_dir / "generation_summary.json").write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    return output


def _measure(video_path: str, true_index: int, sources: list[torch.Tensor]) -> dict:
    frame0 = generated_frames_rgb(video_path)[0]
    similarities = [ssim_rgb(frame0, source) for source in sources]
    predicted = max(range(4), key=lambda index: similarities[index])
    other_best = max(value for index, value in enumerate(similarities) if index != true_index)
    return {
        "ssim_to_A_B_C_D": similarities,
        "predicted_state": chr(ord("A") + predicted),
        "true_state": chr(ord("A") + true_index),
        "correct": predicted == true_index,
        "own_state_ssim": similarities[true_index],
        "own_state_rgb_mae_0_255": mean_absolute_rgb_error(frame0, sources[true_index]),
        "separation_margin": similarities[true_index] - other_best,
    }


def _contact_sheet(eval_dir: Path, protocol: dict, by_key: dict, sources: list[torch.Tensor]) -> Path:
    tile_w, tile_h, header = 224, 128, 24
    columns = [("Real O0", None, 42), ("Untrained", None, 42)] + [
        (f"Step {step}", step, 42) for step in CHECKPOINTS
    ] + [(f"Step 1600 seed {seed}", 1600, seed) for seed in (43, 44)]
    sheet = Image.new("RGB", (len(columns) * tile_w, 4 * (tile_h + header)), "#202020")
    draw = ImageDraw.Draw(sheet)
    for state in range(4):
        for column, (label, step, seed) in enumerate(columns):
            frame = (sources[state] if column == 0 else
                     generated_frames_rgb(by_key[step, state, seed]["video_path"])[0])
            array = (frame.clamp(0, 1).permute(1, 2, 0).numpy() * 255).round().astype(np.uint8)
            x, y = column * tile_w, state * (tile_h + header)
            sheet.paste(Image.fromarray(array).resize((tile_w, tile_h)), (x, y + header))
            draw.text((x + 3, y + 3), f"{chr(ord('A') + state)} {label}", fill="white")
    path = eval_dir / "contact_sheet.png"
    sheet.save(path)
    return path


def _training_validity(root: Path, ids: list[str]) -> dict:
    train = root / "train_1600"
    summary = _read_json(train / "summary.json")
    steps = [json.loads(line) for line in (train / "steps.jsonl").read_text(encoding="utf-8").splitlines()]
    finite_steps = (
        len(steps) == 1600 and [row["step"] for row in steps] == list(range(1, 1601))
        and all(math.isfinite(row["loss"]) and math.isfinite(row["lora_grad_norm"])
                and row["lora_grad_norm"] > 0 and math.isfinite(row["gpu_allocated_mib"])
                and math.isfinite(row["elapsed_seconds"]) for row in steps)
    )
    counts = summary.get("per_state_sample_counts")
    counts_valid = isinstance(counts, dict) and counts == {sample_id: 400 for sample_id in ids}
    logged_ids = [row.get("sample_id") for row in steps]
    if any(sample_id is not None for sample_id in logged_ids):
        from collections import Counter
        counts_valid = counts_valid and dict(Counter(logged_ids)) == counts
    return {
        "finite_losses_and_gradients": finite_steps,
        "per_state_sample_counts": counts,
        "balanced_400_each": counts_valid,
        "no_oom_observed": finite_steps,
        "training_runtime_seconds": summary.get("runtime_seconds"),
        "training_peak_gpu_vram_mib_observed": summary.get("peak_gpu_vram_mib_observed"),
    }



def _write_report(eval_dir: Path, protocol: dict, result: dict) -> Path:
    aggregate = result["final_aggregate"]
    lines = [
        "# M6A-3: four-state current-state conditioning",
        "",
        f"**Classification: {result['classification']}**",
        f"Engineering validity: {'PASS' if result['engineering_validity_pass'] else 'FAIL'}.",
        "",
        f"Generic prompt for every state and seed: {protocol['prompt']}",
        "",
        "## Real source states",
        "",
        "| State | TRAIN sample ID | Exact source O0 |",
        "|---|---|---|",
    ]
    for state in protocol["states"]:
        lines.append(f"| {state['label']} | `{state['sample_id']}` | `{state['source_o0']}` |")
    lines += [
        "", "Source O0 pairwise SSIM (same metric as evaluation):", "",
        "| | A | B | C | D |",
        "|---|---:|---:|---:|---:|",
    ]
    for index, row in enumerate(protocol["source_o0_pairwise_ssim"]):
        values = " | ".join(f"{value:.3f}" for value in row)
        lines.append(f"| {chr(ord('A') + index)} | {values} |")
    lines += [
        "", "## Step-1600 result",
        "",
        f"State ID: {aggregate['state_identification_correct_of_12']}/12 "
        f"(target ≥{IDENTIFICATION_MINIMUM}/12). "
        f"Mean own-state SSIM: {aggregate['mean_own_state_ssim']:.3f} "
        f"(target ≥{OWN_SSIM_MINIMUM:.2f}). "
        f"Median separation margin: {aggregate['median_separation_margin']:.3f} "
        f"(target ≥{MARGIN_MINIMUM:.2f}).",
        f"Median own-state SSIM: {aggregate['median_own_state_ssim']:.3f}; "
        f"RGB MAE mean/median: {aggregate['mean_own_state_rgb_mae_0_255']:.2f}/"
        f"{aggregate['median_own_state_rgb_mae_0_255']:.2f} (0–255 units).",
        "",
        "| True | Seed | SSIM A | SSIM B | SSIM C | SSIM D | Predicted | Own MAE | Margin |",
        "|---|---:|---:|---:|---:|---:|---|---:|---:|",
    ]
    for row in result["final_12x4_ssim_table"]:
        values = " | ".join(f"{value:.3f}" for value in row["ssim_to_A_B_C_D"])
        lines.append(
            f"| {row['true_state']} | {row['seed']} | {values} | {row['predicted_state']} | "
            f"{row['own_state_rgb_mae_0_255']:.2f} | {row['separation_margin']:.3f} |"
        )
    lines += [
        "", "## Per-state fidelity", "",
        "| State | ID | Mean SSIM | Median SSIM | Mean RGB MAE | Median RGB MAE | Median margin |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for state, row in result["per_state"].items():
        lines.append(
            f"| {state} | {row['correct_of_3']}/3 | {row['mean_own_state_ssim']:.3f} | "
            f"{row['median_own_state_ssim']:.3f} | "
            f"{row['mean_own_state_rgb_mae_0_255']:.2f} | "
            f"{row['median_own_state_rgb_mae_0_255']:.2f} | "
            f"{row['median_separation_margin']:.3f} |"
        )
    lines += [
        "", "## Checkpoint trajectory (seed 42)", "",
        "| Checkpoint | State ID | Mean own-state SSIM | A | B | C | D |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in result["checkpoint_trajectory"]:
        per = row["per_state"]
        checkpoint = "Untrained control" if row["checkpoint_step"] is None else str(row["checkpoint_step"])
        lines.append(
            f"| {checkpoint} | {row['correct_of_4']}/4 | {row['mean_own_state_ssim']:.3f} | "
            + " | ".join(f"{per[state]['own_state_ssim']:.3f}" for state in "ABCD") + " |"
        )
    training = result["engineering_validity"]
    evaluation = result["evaluation_runtime_and_memory"]
    lines += [
        "", "## Engineering validity and runtime", "",
        f"Finite 1600 loss/gradient records: {training['finite_losses_and_gradients']}. "
        f"Balanced samples (400/state): {training['balanced_400_each']}. "
        f"Observed counts: {training['per_state_sample_counts']}.",
        f"All four checkpoints load into 241 targets: {training['checkpoints_load']}. "
        f"All 28 videos decode to 17 frames with active O0: "
        f"{training['all_videos_17_frames_and_o0_active']}.",
        f"Training runtime: {training['training_runtime_seconds']} s; "
        f"training peak GPU memory: {training['training_peak_gpu_vram_mib_observed']} MiB. "
        f"Evaluation runtime: {evaluation['runtime_seconds']} s; "
        f"evaluation peak GPU memory: {evaluation['peak_gpu_vram_mib_observed']} MiB.",
        "", "## Artifacts", "",
        f"Selected states: `{protocol['selection_path']}`. "
        f"Pairwise source O0 SSIM: `{eval_dir.parent / 'pairwise_source_o0_ssim.json'}`. "
        f"All 28 videos: `{eval_dir / 'videos'}`. "
        f"Metrics: `{eval_dir / 'metrics.json'}`. "
        f"Contact sheet: `{eval_dir / 'contact_sheet.png'}`.",
        f"Per-step training log: `{eval_dir.parent / 'train_1600/steps.jsonl'}`.",
        "Checkpoints: " + ", ".join(
            f"step {step} `{path}`" for step, path in protocol["checkpoints"].items()
        ) + ".",
        "",
    ]
    path = eval_dir / "report.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def run_analyze(root: Path = ROOT) -> dict:
    eval_dir = root / "eval"
    protocol = _read_json(eval_dir / "protocol.json")
    generation = _read_json(eval_dir / "generation_summary.json")
    if generation["video_count"] != 28 or len(generation["videos"]) != 28:
        raise ValueError("Expected 28 M6A-3 evaluation videos")
    if protocol != _protocol(root):
        raise ValueError("M6A-3 protocol or selected data changed after generation")
    by_key = {
        (row["checkpoint_step"], ord(row["state_label"]) - ord("A"), row["seed"]): row
        for row in generation["videos"]
    }
    if len(by_key) != 28 or set(by_key) != set(_conditions()):
        raise ValueError("Evaluation videos do not cover the declared conditions")
    sources = [source_frame_rgb(state["source_o0"]) for state in protocol["states"]]
    measured = {}
    for step, state, seed in _conditions():
        row = by_key[step, state, seed]
        measured[step, state, seed] = {
            "checkpoint_step": step, "seed": seed, "video_path": row["video_path"],
            **_measure(row["video_path"], state, sources),
        }
    final = [measured[1600, state, seed] for state in range(4) for seed in FINAL_SEEDS]
    correct = sum(row["correct"] for row in final)
    mean_own = statistics.mean(row["own_state_ssim"] for row in final)
    median_own = statistics.median(row["own_state_ssim"] for row in final)
    mean_mae = statistics.mean(row["own_state_rgb_mae_0_255"] for row in final)
    median_mae = statistics.median(row["own_state_rgb_mae_0_255"] for row in final)
    median_margin = statistics.median(row["separation_margin"] for row in final)
    per_state = {}
    for state in range(4):
        subset = [measured[1600, state, seed] for seed in FINAL_SEEDS]
        per_state[chr(ord("A") + state)] = {
            "sample_id": protocol["states"][state]["sample_id"],
            "correct_of_3": sum(row["correct"] for row in subset),
            "mean_own_state_ssim": statistics.mean(row["own_state_ssim"] for row in subset),
            "median_own_state_ssim": statistics.median(row["own_state_ssim"] for row in subset),
            "mean_own_state_rgb_mae_0_255": statistics.mean(row["own_state_rgb_mae_0_255"] for row in subset),
            "median_own_state_rgb_mae_0_255": statistics.median(row["own_state_rgb_mae_0_255"] for row in subset),
            "median_separation_margin": statistics.median(row["separation_margin"] for row in subset),
        }
    trajectory = []
    for step in (None, *CHECKPOINTS):
        rows = [measured[step, state, 42] for state in range(4)]
        trajectory.append({
            "checkpoint_step": step, "seed": 42,
            "correct_of_4": sum(row["correct"] for row in rows),
            "accuracy": sum(row["correct"] for row in rows) / 4,
            "mean_own_state_ssim": statistics.mean(row["own_state_ssim"] for row in rows),
            "per_state": {row["true_state"]: row for row in rows},
        })
    training = _training_validity(root, [state["sample_id"] for state in protocol["states"]])
    checkpoints_valid = (
        set(generation["checkpoints_loaded"]) == {str(step) for step in CHECKPOINTS}
        and all(row["lora_layer_count"] == 241 and row["control_injection_b_norm"] > 0
                for row in generation["checkpoints_loaded"].values())
    )
    videos_valid = all(
        row["frame_count_verified"] == FRAMES and row["control_hook_active"]
        and row["control_nonzero_temporal_indices"] == [0] and row["control_latent_norm"] > 0
        and row["prompt"] == PROMPT and row["adapter_disabled"] == (row["checkpoint_step"] is None)
        for row in generation["videos"]
    )
    engineering_pass = (
        training["finite_losses_and_gradients"] and training["balanced_400_each"]
        and checkpoints_valid and videos_valid
    )
    fidelity_pass = mean_own >= OWN_SSIM_MINIMUM
    margin_pass = median_margin >= MARGIN_MINIMUM
    classification = (
        "C. MULTI-STATE CONDITIONING FAIL" if correct < IDENTIFICATION_MINIMUM else
        "A. MULTI-STATE CONDITIONING PASS" if fidelity_pass and margin_pass else
        "B. STATE DISCRIMINATION PASS, FIDELITY WEAK"
    )
    sheet = _contact_sheet(eval_dir, protocol, by_key, sources)
    output = {
        "classification": classification,
        "engineering_validity_pass": engineering_pass,
        "engineering_validity": {**training, "checkpoints_load": checkpoints_valid,
                                 "all_videos_17_frames_and_o0_active": videos_valid},
        "final_12x4_ssim_table": final,
        "final_aggregate": {
            "state_identification_correct_of_12": correct,
            "state_identification_accuracy": correct / 12,
            "mean_own_state_ssim": mean_own,
            "median_own_state_ssim": median_own,
            "mean_own_state_rgb_mae_0_255": mean_mae,
            "median_own_state_rgb_mae_0_255": median_mae,
            "median_separation_margin": median_margin,
        },
        "per_state": per_state,
        "checkpoint_trajectory": trajectory,
        "criteria": {
            "state_identification": {"pass": correct >= IDENTIFICATION_MINIMUM,
                                     "required_correct_of_12": IDENTIFICATION_MINIMUM},
            "own_state_fidelity": {"pass": fidelity_pass, "required_mean_ssim": OWN_SSIM_MINIMUM},
            "state_separation": {"pass": margin_pass, "required_median_margin": MARGIN_MINIMUM},
        },
        "source_o0_pairwise_ssim": protocol["source_o0_pairwise_ssim"],
        "evaluation_runtime_and_memory": {
            "runtime_seconds": generation["runtime_seconds"],
            "peak_gpu_vram_mib_observed": generation["peak_gpu_vram_mib_observed"],
        },
        "contact_sheet": str(sheet),
        "protocol": str(eval_dir / "protocol.json"),
        "generation_summary": str(eval_dir / "generation_summary.json"),
    }
    output["report"] = str(_write_report(eval_dir, protocol, output))
    (eval_dir / "metrics.json").write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("generate", "analyze"))
    parser.add_argument("--root", type=Path, default=ROOT)
    args = parser.parse_args()
    result = run_generate(args.root) if args.phase == "generate" else run_analyze(args.root)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
