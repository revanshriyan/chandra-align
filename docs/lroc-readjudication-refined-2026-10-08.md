# LROC Re-adjudication with ECC-Refined Correspondences — 2026-10-08

## Question
docs/subpixel-refinement-2026-10-08.md showed LK/ECC refinement tightens
inlier RMSE ~14% on the real OHRC↔OHRC pair but does not break the 1.6px
floor, and named the remaining legitimate path: re-run the LROC absolute-
truth adjudication with ECC-refined correspondences and see whether the
absolute bounds (1m: 2.6–3.5 m; 3m: 6.4–7.1 m) tighten. Claim only what the
absolute number supports — never inlier RMSE.

## Pre-registered success criterion
The held-out ABSOLUTE bound (meters) improves vs baseline on either grid,
with the transform agreeing with baseline (rotation within 0.05°,
scale within 0.001). Inlier-RMSE improvement alone is NOT a claim.

## Method
- **Baseline reproduction first.** Both adjudication pipelines re-implemented
  from the original scripts (/tmp/a1_final.py for 1m, /tmp/a8_3m.py for 3m):
  SIFT nfeatures=8000, CLAHE (percentile 1/99, clipLimit 2.0), Lowe 0.75,
  partial-affine RANSAC 3.0px, seed 7; 1m template 532×532 at (10818,16081),
  900px window; 3m template 178×178, NCC search re-run, 360px window at the
  recovered peak. Held-out: 40%/60% spatial bands, LMEDS fit, check RMSE.
  The run asserts the documented numbers within rounding and STOPS otherwise.
- **Refinement (A9 arm-c logic, LK excluded).** Per inlier pair: 31×31 patches
  from the prepped template (around p1) and prepped ortho window (around p2),
  `cv2.findTransformECC` MOTION_TRANSLATION; `p2 <- p2 + warp[:,2]`
  (sign verified by a synthetic-shift self-test: 0.27px vs 6.87px for the
  wrong sign). Per-image bounds checks, 6px max-shift guard, fallback to the
  original position on `cv2.error`/out-of-bounds/excess shift. Applied to the
  baseline inlier pairs only (96 for 1m, 46 for 3m), then RANSAC + frozen
  gates re-run. Deterministic: `cv2.setRNGSeed(7)` before every estimator.
- **Causal test (added after seeing confounded naive results).** Identical
  fit/check splits as the baseline held-out; LMEDS fit on the baseline fit
  indices with REFINED positions, evaluated on the baseline check indices
  with ORIGINAL positions. This isolates the refinement effect from
  inlier-set selection and fit-set-size effects.
- Full data: results/table_lroc_refined.csv. Repro script:
  scripts/readjudicate_lroc_refined.py.

## Baseline reproduction: EXACT
| Grid | Matches | Inliers | RMSE | Scale | Rot (atan2) | Held-out x | Held-out y |
|------|---------|---------|------|-------|-------------|------------|------------|
| 1m | 164 | 96 | 1.4120px=1.412m | 1.0156 | +2.115° | 2.58m (28/35) | 3.46m (24/48) |
| 3m | 46 | 46 | 0.8611px=2.583m | 1.0129 | +2.679° | 7.06m (20/16) | 6.39m (15/15) |

All values reproduce the adjudication docs to the printed precision,
including quadrant counts (1m: 1/7/50/38, entropy 1.363; 3m: 15/10/13/8,
entropy 1.960) and the 3m NCC peak (0.4510 at (3608,5360)). Rotation is
reported in the matcher's atan2 convention throughout (+2.1° to +2.7°,
consistent with the settled sign resolution).

## Results

| Grid | Arm | Inliers | Inlier RMSE | Scale | Rot | Gate | Held-out x / y (m) |
|------|-----|---------|-------------|-------|-----|------|--------------------|
| 1m | baseline | 96/164 | 1.4120px | 1.0156 | 2.115° | COARSE | 2.58 / 3.46 |
| 1m | ECC-refined | 81/96 | 1.4248px | 1.0143 | 2.091° | COARSE | 1.92 / 2.84 |
| 3m | baseline | 46/46 | 0.8611px | 1.0129 | 2.679° | COARSE | 7.06 / 6.39 |
| 3m | ECC-refined | 46/46 | 1.0235px | 1.0111 | 2.631° | COARSE | 6.81 / 6.32 |

Refinement stats: 1m 90 refined / 6 fallback; 3m 38 refined / 8 fallback.
No crashes, no degenerate arms; RANSAC never returned None.

### Transform agreement (refined vs baseline)
- 1m: Δrot 0.024° (within 0.05°), Δscale 0.00133 (**outside** 0.001),
  Δtrans 0.83px.
- 3m: Δrot 0.048° (within 0.05°), Δscale 0.00179 (**outside** 0.001),
  Δtrans 0.18px.
The pre-registered agreement bar fails on scale for both grids (narrowly).

### Held-out, naive pipeline comparison
- 1m: 2.58→1.92m (x), 3.46→2.84m (y) — apparent improvement, BUT the
  inlier set changed (96→81) and the splits with it (x: 28/35→24/26,
  y: 24/48→30/40). Confounded.
- 3m: 7.06→6.81m, 6.39→6.32m — flat within small-sample noise.

### Causal test (identical splits; only fit positions refined)
- 1m: x 2.58→**2.26**m (−12%), y 3.46→**3.89**m (+12%).
- 3m: x 7.06→**6.54**m (−7%), y 6.39→**6.94**m (+9%).
One split improves, the other worsens, on both grids — the signature of
noise, not signal. The naive-comparison "improvements" were selection and
fit-size artifacts (the refined transforms were fit on 81/46 points vs
28/24 and 20/15 for the baseline held-out fits).

## Verdict on the success criterion: NOT MET — clean negative
ECC refinement of SIFT correspondences does not tighten the absolute
accuracy bounds. The 1m bound stands at **2.6–3.5 m**, the 3m bound at
**6.4–7.1 m**. No gate tier changed (all COARSE_ADVISORY).

## Interpretation
On the OHRC↔OHRC pair (A9), ECC refinement tightened inlier RMSE 14%
because both images share radiometry — patch ECC genuinely denoised
localization. Here it does not even do that: inlier RMSE got slightly
*worse* on both grids (1m: 1.4120→1.4248px; 3m: 0.8611→1.0235px).
Cross-sensor patches (OHRC vs LROC ortho: different radiometry, 1.6%
scale residual) give ECC translation-only fits radiometric texture to
align rather than true geometric offset — the "refinement" adds noise
instead of removing it. The causal test confirms the transform itself is
no better. Additional observations:

- 1m refined dropped 15 marginal inliers (81/96); Q1 went from 1 inlier
  to 0 (counts 0/1/45/35, entropy 1.363→1.072) — refinement concentrated
  the solution further rather than spreading it.
- 3m kept all 46 inliers but with worse self-consistency; the transform
  moved least here (Δtrans 0.18px) yet the causal test still shows no gain.
- The absolute bounds remain limited by spatially-varying distortion and
  grid resolution, not by keypoint localization — consistent with the A9
  finding that the floor is not (mainly) localization noise.

## What this closes
The "achieve the best" follow-up is answered: correspondence refinement
is not a lever for absolute accuracy on this cross-sensor pair. The
remaining legitimate directions are the ones that address the actual
bottleneck — spatially-varying distortion (regularized dense deformation
field, scored by held-out RMSE) and spatial coverage (the empty-Q1
extrapolation) — not finer localization of the same correspondences.
