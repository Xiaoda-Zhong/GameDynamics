# M6G-0A: held-out flip decomposition

**Conclusion: A. Most failures come from the first-turn segment.** The existing overall flip result remains 7/12. Four of the five failed pairs miss only the first segment; one misses both. No pair fails only the final segment.

## Exact saved-metric rule

The existing future-only scorer excludes O0→O1. Its first-turn segment is **A1 alone** (generated O1→O2), while the final-turn segment is the mean of **A10…A15** (generated O10→O11 through O15→O16). Positive calibrated horizontal flow is the source-calibrated left-turn direction. First-segment success requires ORIGINAL > 0 and REVERSED < 0; final-segment success requires ORIGINAL < 0 and REVERSED > 0. The different segment lengths matter when interpreting the concentration of failures.

## All 12 matched pairs

| State | Seed | First ORIGINAL u | First REVERSED u | First flip | Final ORIGINAL u | Final REVERSED u | Final flip | Existing overall |
|---|---:|---:|---:|---|---:|---:|---|---|
| A | 42 | -2.760097 | -0.484264 | No | -0.272520 | +7.946653 | Yes | No |
| A | 43 | +6.917599 | -1.011573 | Yes | -8.032847 | +13.903085 | Yes | Yes |
| A | 44 | +9.023968 | +3.133225 | No | -3.042158 | +5.753416 | Yes | No |
| B | 42 | +2.929679 | +1.968428 | No | -1.112330 | +2.317443 | Yes | No |
| B | 43 | +3.975457 | -6.388800 | Yes | -1.787813 | +4.195827 | Yes | Yes |
| B | 44 | +1.010576 | +0.379478 | No | -0.525901 | -1.255624 | No | No |
| C | 42 | +0.147954 | +0.113385 | No | -0.384412 | +0.177361 | Yes | No |
| C | 43 | +0.021889 | -0.429433 | Yes | -1.087173 | +1.361102 | Yes | Yes |
| C | 44 | +0.030119 | -0.152895 | Yes | -0.123282 | +0.048908 | Yes | Yes |
| D | 42 | +7.965899 | -5.029138 | Yes | -4.492592 | +1.520648 | Yes | Yes |
| D | 43 | +9.574679 | -9.205405 | Yes | -6.290639 | +2.257668 | Yes | Yes |
| D | 44 | +9.612783 | -5.101303 | Yes | -4.258913 | +2.496937 | Yes | Yes |

## Aggregates

- First segment: **7/12** expected reversals.
- Final segment: **11/12** expected reversals.
- Both segments correct: **7/12**, matching the official overall flip count.
- Failure types: first only **4**, final only **0**, both **1**.

### By held-out state

| State | First | Final | Both / overall |
|---|---:|---:|---:|
| A | 1/3 | 3/3 | 1/3 |
| B | 1/3 | 2/3 | 1/3 |
| C | 2/3 | 3/3 | 2/3 |
| D | 3/3 | 3/3 | 3/3 |

### By seed

| Seed | First | Final | Both / overall |
|---:|---:|---:|---:|
| 42 | 1/4 | 4/4 | 1/4 |
| 43 | 4/4 | 4/4 | 4/4 |
| 44 | 2/4 | 3/4 | 2/4 |

## Provenance

Every flow value and Boolean above comes from the existing M6F-2 `metrics.json`. The stored flags were independently checked against the unchanged `flip_future` sign tests, and both-segment successes were checked against the official 7/12 aggregate. No VACE forward, video regeneration, video decode, optical-flow recomputation, training, or metric change was performed.

Source metrics: `/root/autodl-tmp/GameDynamics_starter/data/experiments/m6f2_action_contrastive_128_6400/eval/metrics.json` (SHA-256 `55a163d66f0032795040852796470e7e95646c2ea78f501b823787347bdcd0d7`). Raw JSON: `/root/autodl-tmp/GameDynamics_starter/data/experiments/m6g0a_heldout_flip_decomposition/raw_pairs.json`. Raw CSV: `/root/autodl-tmp/GameDynamics_starter/data/experiments/m6g0a_heldout_flip_decomposition/raw_pairs.csv`. Summary: `/root/autodl-tmp/GameDynamics_starter/data/experiments/m6g0a_heldout_flip_decomposition/summary.json`.
