# Showcase Demo

[`heldout_state_D_seed_42.mp4`](heldout_state_D_seed_42.mp4) places the saved M6F-2 ORIGINAL, REVERSED, and ALL-NOOP generations side by side.

- held-out test state: `my_way_home_episode_0007_clip_000006`
- seed: 42
- identical observed O0, prompt, initial noise, model, and inference settings
- only the action sequence differs
- presentation labels: `ORIGINAL | REVERSED | ALL-NOOP`
- subtitle: “Same O0 · same seed/noise · only action sequence changed”
- both predeclared first-turn and final-turn flip checks pass for this pair

The compositor pastes all 17 decoded source frames at their native 448×256 size. Labels and compact action timelines are drawn outside the video panels. It does not invoke VACE, replace frames, crop generated content, or change timing.

Rebuild from the existing videos with:

```bash
python demo/make_showcase.py
```

[`showcase.json`](showcase.json) records source paths and SHA-256 hashes. The GIF is a scaled preview; use the MP4 for presentations.
