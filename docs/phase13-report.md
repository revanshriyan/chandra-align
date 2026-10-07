# Phase 13 Report — Window-tiling batch harness

## What was built

- `chandra_align/eval/tiling.py` — deterministic, I/O-free window generator:
  fixed non-overlapping grid over a declared footprint, dark/flat
  eligibility gates (documented thresholds: std < 8 DN or >50% dark pixels
  skips), row-major order, optional seeded shuffle. `check_no_overlap`
  for verification.
- `scripts/run_phase13_batch.py` — one script, one table. Per eligible
  window: SIFT+Lowe(0.75) → `verify_guarded` → quadrant metrics →
  `validate_registration_gate` (min 8 inliers) → `trust_evaluate` held-out
  split on guarded inliers → 0–100 confidence heuristic (report-only).
  Every exception becomes an `ERROR` row; failures are rows, never crashes.
  Incremental CSV writes + `--resume` survive restarts.
- `results/table_phase13_windows.csv` — 84 rows, 16 columns.
- `tests/test_phase13.py` — 8 tests pass.

## Key correction during development

The first run used same nominal pixel coordinates for both products and
produced 83 uniform DEGENERATE_FAILURE rows (2–4 inliers). Visual + phase-
correlation checks proved the products do NOT share pixel coordinates —
the "failures" were measuring different ground, which would have been a
misleading table. The harness now uses the MEASURED inter-product affines
from `scripts/make_benchmark_crops.py` (OHRC 12:09→14:06: +1060/+1302 px
plus ~0.9° rotation; TMC-2 fore→nadir: −315/+7780 px). B-window origins
are mapped, bounds-checked, and dark/flat-gated on both sides.

## Measured results (84 windows, frozen default pipeline, no tuning)

| Pair | Windows | COARSE_ADVISORY | DEGENERATE_FAILURE | Median inliers | Median RMSE |
| --- | --- | --- | --- | --- | --- |
| TMC-2 fore/nadir | 42 | 21 | 21 | 683 | 2.18 px |
| OHRC 12:09/14:06 | 42 | 41 | 1 | 2166 | 1.49 px |

Held-out RMSE tracks inlier RMSE (TMC-2 median 2.63 px, OHRC median
1.74 px) — no split/independent disagreement of the kind Phase 11's
sun-flip case showed.

## What the table says (honest reading)

- **OHRC stereo is systematically registrable** at COARSE level: 41/42
  windows, thousands of inliers each. The pipeline is consistent, not lucky.
- **TMC-2 fore/nadir splits 50/50**: windows carry 350–1660 inliers but the
  affine residual straddles the 2.5 px gate. Cross-view parallax is not
  fully explained by a per-window affine — the gate honestly separates
  fittable windows from the rest. This is the systematic evidence base for
  future coarse-to-fine / higher-order model work.
- Zero ABSTAIN rows: every window produced correspondences; the pipeline
  never failed to find *something*, it failed (where it failed) on geometry.

## Bug fixed (measurement-only, no pipeline behavior change)

`measure_pair` in `chandra_align/eval/robustness.py` crashed with
`M_gt=None` (ValueError on reshape) although its docstring promises the
independent accuracy is "withheld when unavailable". Added the missing
`M_gt is None` guard plus an additive `return_inliers=True` option
(underscore keys, not part of the stable schema). Phase 11's 8 tests
still pass.
