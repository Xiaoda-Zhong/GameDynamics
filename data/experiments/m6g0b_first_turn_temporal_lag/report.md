# M6G-0B: First-turn temporal-lag audit

**Classification: A. TRUE FIRST-TURN CONTROL FAILURE.**

The official O1→O2 first-turn flip remains **7/12**. Among the five pairs that fail there, none shows the expected ORIGINAL-positive / REVERSED-negative reversal at O2→O3, O3→O4, or O4→O5. Thus the saved flows show no delayed onset of the expected reversal within this first latent window. This is an audit of these 12 pairs, not a new control metric.

Positive calibrated horizontal flow denotes LEFT; negative denotes RIGHT. A transition passes only when ORIGINAL > 0 and REVERSED < 0. Values below reuse the M6F-2 saved future-only optical-flow rows; no video was decoded or generated.

| State | Seed | O1→O2 O/R (flip) | O2→O3 O/R (flip) | O3→O4 O/R (flip) | O4→O5 O/R (flip) | Earliest | O1→O5 mean O/R (flip) |
|---|---:|---|---|---|---|---|---|
| A | 42 | -2.760/-0.484 (N) | -3.575/+0.788 (N) | +4.646/+1.426 (N) | -0.096/+1.249 (N) | None | -0.446/+0.745 (N) |
| A | 43 | +6.918/-1.012 (Y) | -0.779/+1.128 (N) | +1.668/+0.585 (N) | -3.938/+0.697 (N) | O1->O2 | +0.967/+0.349 (N) |
| A | 44 | +9.024/+3.133 (N) | +6.050/+2.860 (N) | +6.672/+2.746 (N) | +1.690/+2.884 (N) | None | +5.859/+2.906 (N) |
| B | 42 | +2.930/+1.968 (N) | +1.495/+1.440 (N) | +0.929/+1.624 (N) | -0.197/+1.973 (N) | None | +1.289/+1.751 (N) |
| B | 43 | +3.975/-6.389 (Y) | +0.854/-1.456 (Y) | +1.799/+1.133 (N) | -0.342/+2.348 (N) | O1->O2 | +1.572/-1.091 (Y) |
| B | 44 | +1.011/+0.379 (N) | +0.009/+0.122 (N) | +0.323/+0.059 (N) | +0.014/+0.218 (N) | None | +0.339/+0.195 (N) |
| C | 42 | +0.148/+0.113 (N) | +0.203/+0.219 (N) | +0.346/+0.296 (N) | -0.050/+0.188 (N) | None | +0.162/+0.204 (N) |
| C | 43 | +0.022/-0.429 (Y) | -0.094/+0.326 (N) | +0.742/+0.616 (N) | +0.018/+0.567 (N) | O1->O2 | +0.172/+0.270 (N) |
| C | 44 | +0.030/-0.153 (Y) | +0.070/-0.180 (Y) | +0.172/-0.026 (Y) | +0.014/-0.092 (Y) | O1->O2 | +0.072/-0.113 (Y) |
| D | 42 | +7.966/-5.029 (Y) | +3.063/-0.342 (Y) | +1.866/-1.083 (Y) | -3.641/-1.315 (N) | O1->O2 | +2.314/-1.942 (Y) |
| D | 43 | +9.575/-9.205 (Y) | +5.923/-3.569 (Y) | +5.490/-1.187 (Y) | -3.852/+2.025 (N) | O1->O2 | +4.284/-2.984 (Y) |
| D | 44 | +9.613/-5.101 (Y) | +4.597/+0.523 (N) | +5.549/+0.852 (N) | +2.996/+0.302 (N) | O1->O2 | +5.689/-0.856 (Y) |

## Counts

| Transition | Expected reversal |
|---|---:|
| O1->O2 | 7/12 |
| O2->O3 | 4/12 |
| O3->O4 | 3/12 |
| O4->O5 | 1/12 |

| Earliest reversal | Pairs |
|---|---:|
| immediate | 7/12 |
| 1-frame-late | 0/12 |
| 2-frame-late | 0/12 |
| 3-frame-late | 0/12 |
| no reversal | 5/12 |

Diagnostic O1→O5 signed-flow window: **5/12** expected reversals. The window averages four transition flows and is not the official O1→O2 metric; a pair can pass the single-transition test yet fail the window average.

The action schedule turns for A0–A1 and uses NOOP for A2–A9. A later reversal would therefore be evidence of temporal persistence or lag, not proof of correct action timing. None of the five immediate failures recovers in the next three transitions under the same strict sign rule.

Raw values: [raw_transitions.csv](raw_transitions.csv), [raw_pairs.json](raw_pairs.json). Machine-readable counts and source SHA-256 checksums: [summary.json](summary.json). Source: M6F-2 `eval/transitions.jsonl` and `eval/metrics.json`. No model, videos, scorer, thresholds, or dependencies were changed.
