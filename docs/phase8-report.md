# Phase 8 Report — Sub-pixel refinement + deterministic gating

## What was built

**New modules**
- `chandra_align/refine/subpixel.py` — two refinement contenders with the same
  `(kept_a, refined_b, stats)` contract as the existing NCC refinement:
  - `refine_lk`: pyramidal Lucas-Kanade per RANSAC inlier
    (`cv2.OPTFLOW_USE_INITIAL_FLOW`), kept iff the reprojection residual does
    not worsen vs the RANSAC model.
  - `refine_phasecorr`: per-point FFT phase correlation + 2D paraboloid peak
    fit (Foroosh-style), with Hann windowing and peak-height quality stats.
- `chandra_align/metrics/conditioning.py` — Gate 3 transform conditioning
  (cond < 1e7, |det| > 1e-4, SVD scale-ratio < 20, projectivity < 0.05,
  RMSE ≤ 5 px), unique-inlier dedup (2 px merge, raw + unique reported), and
  pre-fit ABSTAIN classification (`ZERO_CANDIDATES`, `INSUFFICIENT_UNIQUE`,
  `COLLINEAR`, `ILL_CONDITIONED`, `NO_VALID_MODEL`).
- `chandra_align/refine.verify_guarded` — guarded RANSAC path: ABSTAIN check →
  `verify_magsac` → 2 px dedup → span guard (<50% Y-span relaxes the RANSAC
  threshold up to 12 px and refits) → Gate 3 backstop.

**Gate integration**
- `validate_registration_gate(..., model=None)`: when the fitted transform is
  supplied, Gate 3 runs as a final backstop — a degenerate model is REJECTED
  even if residual tiers would pass. Fully backward compatible (no model →
  old behavior).

**Verify-don't-add**: tiled reads already enforce non-overlapping cores
(`chandra_align/matching/tiled.py` raises on core overlap) — no change needed.

## Bake-off result (held-out RMSE, no-peeking)

`scripts/run_phase8_bakeoff.py` → `results/phase8_bakeoff.json`.
SIFT matches → guarded verify → 80/20 fit/held-out split → refit → held-out RMSE.

| Case | none | ncc | lk | phasecorr |
| --- | --- | --- | --- | --- |
| shift (dx=7.3, dy=-3.9) | 0.226 | 0.366 | 0.226 | 0.291 |
| illumination (γ=1.6) | 24.705 | **0.445** | 24.705 | 23.846 |
| scale (10×) | — (too few SIFT matches) | — | — | — |

**Winner: NCC (incumbent), mean held-out 0.41 px.** LK converges to the input
(no-op here — kept points, changed nothing); phase-correlation underperforms.
Neither displaces the default. The illumination case is the honest story:
SIFT+RANSAC inliers were systematically biased (24.7 px held-out), and NCC
template re-localization corrected them to 0.44 px — refinement as rescue,
not polish.

## Acceptance check

- Synthetic suite: 12 new Phase 8 tests pass; neighboring suites show no new
  failures (remaining failures are pre-existing `rasterio`-missing env gaps).
- Degenerate signatures rejected: the 1.24e-20 collapse fails Gate 3
  (`det_gt_1e-4`); extreme anisotropy fails (`scale_ratio_lt_20`).
- ABSTAIN / REJECT / DEGENERATE_FAILURE are distinct in telemetry
  (`verify_guarded` returns `abstain_code` separately from gate verdicts).
