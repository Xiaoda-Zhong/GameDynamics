# Evaluation

The final held-out evaluation is implemented by [`training/wan/m6f2_contrastive_eval.py`](../training/wan/m6f2_contrastive_eval.py).

Official artifacts are stored under [`data/experiments/m6f2_action_contrastive_128_6400/eval/`](../data/experiments/m6f2_action_contrastive_128_6400/eval/):

- `protocol.json`: frozen inference and scoring settings
- `metrics.json`: aggregate, per-state, and matched-pair results
- `transitions.jsonl`: future-only optical-flow measurements
- `report.md`: concise result report
- `videos/`: 36 held-out generations
- `contact_sheet.png`: overview of all generations

Re-score the existing videos without running VACE:

```bash
python training/wan/m6f2_contrastive_eval.py score
```

The first-turn failure decomposition and temporal-lag audit are in [`m6g0a_heldout_flip_decomposition`](../data/experiments/m6g0a_heldout_flip_decomposition/) and [`m6g0b_first_turn_temporal_lag`](../data/experiments/m6g0b_first_turn_temporal_lag/).
