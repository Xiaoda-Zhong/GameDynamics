# GameDynamics Codex Instructions

## Project goal
Build a research-oriented controllable gameplay video generation pipeline:
ViZDoom trajectories -> video/data curation -> Wan2.1 LoRA -> controllable generation -> evaluation.

## Current milestone
Milestone 1 only: collect ViZDoom trajectories with RGB, depth, action labels, and metadata.

## Rules
- Do not implement later milestones unless explicitly requested.
- Prefer small, readable Python modules over abstractions.
- Before changing code, explain the intended change briefly.
- After changing code, run the smallest relevant test.
- Do not silently swallow exceptions.
- Keep data formats explicit and documented.
- Preserve action labels exactly as strings in metadata.
- Code should run on Apple Silicon macOS for local data collection.
- Avoid GPU-only dependencies in Milestone 1.
