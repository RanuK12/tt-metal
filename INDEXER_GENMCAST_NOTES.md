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

## RESULTS — generalized scheduler (commit a8853b0 + rows_used divisor fix)

math_util before -> after (GLX sp7, bf16 q + bfp8 k, HiFi2):

| config | G | U | baseline | NEW | delta |
|---|---|---|---|---|---|
| glm5_qc2_kc16 (control) | 10 | 110 | 70.1% | 70.1% | 0 (no regression) |
| glm5_qc1_kc16 | 20 | 110 | 36.2% | 71.4% | +35pp (beats control) |
| glm5_qc4_kc16 | 5  | 110 | 49.4% | 71.7% | +22pp |
| glm5_qc1_kc32 | 20 | 55  | 35.7% | 69.5% | +34pp |
| glm5_qc2_kc24 | 10 | 74  | 41.9% | 67.2% | +25pp |
| glm5_qc2_kc16_hb4 (stream) | 10 | 110 | 1.85% | 1.84% | ~0 (q-stream bound; q-mcast off) |
| dsv32_qc2_kc8 (control) | 10 | 220 | 76.1% | 76.1% | 0 (no regression) |
| dsv32_qc1_kc8 | 20 | 220 | 64.1% | 75.6% | +11.6pp |

All 34 original accuracy/determinism/fidelity tests pass. Controls unchanged.

### Correctness fix (rows_used must divide G when G>gy)
Uneven num_groups across a column's rows (gy ∤ G, e.g. G=12) breaks k-mcast lockstep -> the
column's row-0 sender waits forever on receivers that already finished. Fix: rows_used =
largest divisor of G that is <= gy (uniform num_groups per row). Deployed cases (G in {5,10,20})
unchanged. Prime G>gy degrades to rows_used==1 (k-mcast off, still correct). Shared helpers
rows_used_for / cols_used_for in indexer_score_work_split.hpp (factory + perf model agree).
Added test_indexer_score_genmcast_regimes for the G>gy / uneven-U / prime / streaming paths.

### Remaining optimization ideas
- band-outer/group-inner for G>gy: read each band once (reuse across the group phases) instead of
  re-reading per phase -> could lift QC=1 further. Needs >1 group's q resident (cheap). Not done.
- streaming (HB<Hi) is q-re-read bound; q-mcast disabled there. Not deployed (glx uses HB=0).
