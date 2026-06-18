# indexer_score generalized-multicast — working notes

Worktree branch `skrstic/indexer_score_genmcast` off `skrstic/indexer_score_op`.
Goal: replace the rigid `grid_aligned` mcast with a generalized banded product
scheduler (bands→columns / k-mcast, groups→rows / q-mcast). Full grid + mcast
whenever possible. No regression on deployed GLM5/DSv32; big gains on configs
that get no mcast today.

ALWAYS run with `export TT_METAL_HOME=/localdev/skrstic/tt-metal-genmcast` and the
worktree venv, else kernels JIT from the main repo (split-brain).

## Baseline (current branch, GLX sp7, bf16 q + bfp8 k, HiFi2). util = mmflops/(cores*cycles*peak); lower core*ns = higher util.

| config | G | U | device ms | cores | core*ns | math_util | mcast today |
|---|---|---|---|---|---|---|---|
| glm5_qc2_kc16  | 10 | 110 | 0.344 | 110 | 37.9 | 70.1% | ON (control) |
| glm5_qc1_kc16  | 20 | 110 | 0.666 | 110 | 73.3 | 36.2% | OFF |
| glm5_qc4_kc16  | 5  | 110 | 0.489 | 110 | 53.8 | 49.4% | OFF |
| glm5_qc1_kc32  | 20 | 55  | 0.676 | 110 | 74.4 | 35.7% | OFF |
| glm5_qc2_kc24  | 10 | 74  | 0.576 | 110 | 63.4 | 41.9% | OFF |
| glm5_qc2_kc16_hb4 | 10 | 110 | 13.05 | 110 | 1436 | 1.85% | q OFF (stream) |
| dsv32_qc2_kc8  | 10 | 220 | 0.635 | 110 | 69.8 | 76.1% | ON (control) |
| dsv32_qc1_kc8  | 20 | 220 | 0.754 | 110 | 82.9 | 64.1% | OFF |

## Design (product layout)
- G = Sqt/QC groups → grid rows; U = ceil(Tt/KC) bands → grid columns.
- G>gy: rows phase-stack ceil/floor(G/gy) groups (uniform per column → k lockstep).
- U>gx: columns phase-stack ceil/floor(U/gx) bands.
- Iteration group-OUTER, band-INNER: K-mcast per band down a column (sender row 0);
  Q-mcast once per group along a row (sender on diagonal), rendezvous at group bounds.
- Uneven band counts across columns only cause benign waiting at group boundaries (no deadlock;
  each column is an independent K-mcast rect, Q is a per-group rendezvous).
- mcast rects are FIXED per core (one K column rect, one Q row rect); only the data changes per phase.
- G<gy: stage-5 refinement (split q-rows); until then use G rows (some idle).
