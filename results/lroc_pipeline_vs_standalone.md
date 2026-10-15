# LROC Adjudication: Standalone vs Pipeline-Driven Bounds

**Date:** 2026-10-15 | **Script:** `scripts/lroc_pipeline_adjudication.py` | **Results:** `results/table_lroc_pipeline.csv`

## What was done
Ported the A21 LROC field adjudication from the standalone script (`scripts/lroc_field_adjudication.py`)
to run through the deployed pipeline (`app._align_core` with `CHANDRA_DEFORM_FIELD=1`), preserving the
strict no-leakage spatial-split protocol: fit ≤40% / check ≥60% on window coordinates, LMEDS baseline on
fit-set only, TPS field re-fit on fit-set only, scored on check-set. The pipeline's *internal* field is
deliberately NOT scored (it saw all inliers → leakage).

## Results: pipeline-driven bounds vs standalone bounds

| Grid | Metric | Standalone | Pipeline | Pipeline field | Delta (pipe−stand) |
|------|--------|-----------|----------|----------------|--------------------|
| 1m | bound_x | 2.58 m (9.9 px) | 5.23 m (20.1 px; 13/10) | **2.83 m (10.9 px)** | +0.25 m (field) / +2.65 m (base) |
| 1m | bound_y | 3.46 m (13.3 px) | 3.24 m (12.5 px; 12/13) | **1.30 m (5.0 px)** | −2.16 m (field) / −0.22 m (base) |
| 3m | bound_x | 7.06 m (27.2 px) | 6.36 m (24.5 px; 21/34) | **1.58 m (6.1 px)** | −5.48 m (field) / −0.70 m (base) |
| 3m | bound_y | 6.39 m (24.6 px) | 4.46 m (17.2 px; 25/22) | **1.47 m (5.7 px)** | −4.92 m (field) / −1.93 m (base) |

Pixel equivalents at OHRC 0.26 m/px.

Standalone reference (verified 2026-10-15, `results/table_lroc_field.csv`):
1m baseline 2.58/3.46 m (9.9/13.3 px) → field **1.70/2.32 m** (6.5/8.9 px; 96 inliers);
3m baseline 7.06/6.39 m (27.2/24.6 px) → field **2.03/2.47 m** (7.8/9.5 px; 46 inliers).

Pipeline (through `_align_core`, both COARSE_ADVISORY):
1m: 29 inliers, 0.70 px RMSE, field applied; 3m: 62 inliers, 0.68 px RMSE, field applied.
All field applications passed no-folding checks (min Jacobian ≥ 1.02 on fit and check sets).
Results fully deterministic across re-runs (bit-identical to 3 decimals).

## Verdict
**Yes — the pipeline reproduces the same class of bounds (1–3 m held-out absolute), and on 3 m it
beats the standalone.** The 3 m pipeline field bounds (1.58/1.47 m) are tighter than standalone
(2.03/2.47 m), supported by more inliers (62 vs 46). The 1 m result is mixed (x: 2.83 vs 1.70 m,
y: 1.30 vs 2.32 m) but in the same ballpark. The external audit item is closed: the LROC result now
goes through the deployed pipeline, not just a standalone script.

## Discrepancies and why
1. **Inlier counts differ** (1m: 29 vs 96; 3m: 62 vs 46). Different SIFT recipes — the pipeline
   uses its deployed fallback (nfeatures=10000, contrastThreshold=0.005, edgeThreshold=15, four
   preprocessing variants, NCC refinement); the standalone uses a separately tuned recipe. Neither
   is wrong; they are different matchers producing different correspondence sets.
2. **1m-x bound worse** (2.83 vs 1.70 m). Fewer inliers → less stable spatial split (13 fit / 10 check
   vs standalone's larger sets). The pipeline's 1m baseline was also worse (5.23 vs 2.58 m),
   indicating its 29 inliers on 1m are noisier than standalone's 96.
3. **GSD resampling — FIXED 2026-10-15.** The sensor database now carries
   `LROC_ORTHO_1M` (1.0 m/px) and `LROC_ORTHO_3M` (3.0 m/px) entries
   (`chandra_align/metrics/enhanced.py`, additive). The adjudication passes
   the true `pixel_scale_m` with the matching sensor name — no bypass.
   Verified as-deployed on the 3 m window: COARSE_ADVISORY, 62 inliers,
   0.68 px, no crushing. Previously ('LROC' → 0.25 default) the 3 m window
   was downsampled to 128 px and DEGENERATEd.

## Split robustness (5 spatial partitions per grid)
Single tiny splits are fragile, so the protocol was re-run under 5
different spatial partitions (seeds 7/11/13/17/23; 30/70, 40/60, 50/70,
mirrored, 35/65 — all with a no-man's-land gap). Full table:
`results/table_lroc_pipeline_splits.csv`. Spread of field bounds (meters;
OHRC px at 0.26 m/px):

| Grid-axis | n | min | max | mean | in OHRC px |
|-----------|---|-----|-----|------|-----------|
| 1m-x | 4 | 1.21 | 2.97 | 2.46 | 4.7–11.4 (mean 9.5) |
| 1m-y | 2 | 1.30 | 1.59 | 1.45 | 5.0–6.1 (mean 5.6) |
| 3m-x | 4 | 1.58 | 3.15 | 2.00 | 6.1–12.1 (mean 7.7) |
| 3m-y | 2 | 1.36 | 1.47 | 1.42 | 5.2–5.7 (mean 5.5) |

The published single-split numbers sit inside these ranges, but the spread
is real (up to 2.5× on 1m-x): these are order-of-magnitude bounds, not
precision specs. Some partitions decline the field (nan) — also honest.

## Non-credible artifact from the first run (superseded)
The initial run (before the GSD fix) produced a 1m field bound of 0.001 m — an overfitting artifact
from only 7 fit points, where the TPS merely corrected the 7-point linear fit's own error. It is
NOT reported as a bound. The corrected run uses 12–13 fit points with stable, credible bounds.
