# First Absolute-Truth Adjudication: OHRC vs LROC Ortho — 2026-10-08

## Correction (2026-10-08, ~01:00 IST) — re-derived with corrected geometry

Two corrections to the original run below:

1. **OHRC pixel scale.** The run assumed 0.25 m/px. The OHRC PDS4 label gives
   0.26 m/px, so the 2048-px crop spans 532.48 m and was resampled to
   **178×178 px at 3 m/px** (2048×0.26/3 = 177.5), not 171×171.
2. **File layout clarified.** The 3m file's PDS3 label states `LINES=15895`,
   `LINE_SAMPLES=7668`, `SAMPLE_BITS=16`, `SAMPLE_TYPE=LSB-UNSIGNED-INTEGER`,
   `RECORD_BYTES=15336` → image offset **15336 bytes**, 16-bit little-endian
   (not 8-bit). File-size arithmetic confirms
   (243781056 − 15336 = 15895×7668×2 exactly).

The 3m offset itself was already verified correct — no frame-shift error here.
But the template location, scale, and rotation below are all re-derived.

### Corrected results
| Metric | Original (superseded) | Corrected |
|--------|----------------------|-----------|
| OHRC scale used | 0.25 m/px | **0.26 m/px** (from PDS4 XML) |
| Template size | 171×171 | **178×178** |
| Template location (ortho px) | (3521, 5276), NCC 0.55 | **(3608, 5360)**, NCC 0.4510 (0.6763 after −2° rotation sweep) |
| SIFT inliers | 61/70 (87.1%) | **46/46 (100%)** (nfeatures=8000, CLAHE, Lowe 0.75, RANSAC 3.0px, seed 7; identical across 5 seeds) |
| Inlier RMSE | 0.81 px = 2.42 m | **0.8611 px = 2.58 m** |
| Scale (ortho→OHRC) | 0.947 | **1.0129** (expected 0.9972; 1.6% residual, same as corrected 1m) |
| Rotation | −2.37° | **+2.68°** |
| Entropy / quadrants | — | 4-quadrant entropy **1.96** (counts TL/TR/BL/BR = 15/10/13/8 — balanced, all 4 active). Passes COARSE gates (RMSE 0.86 ≤ 2.50, 46 ≥ 8 inliers, entropy 1.96 ≥ 0.50, 4 ≥ 2 quadrants); not ACCEPT (RMSE 0.86 > 0.50) |

### Rotation sign: resolved
The old −2.37° disagreed in sign with the corrected 1m re-derivation (+2.12°)
on the same M1443025251 product. The re-derivation settles it: SIFT recovers
**+2.679°**, independently corroborated by ECC dense alignment (**+2.289°**)
and by the NCC rotation sweep (best at −2° in `getRotationMatrix2D`
convention = **+2° in the SIFT `atan2` convention**; the two conventions have
opposite sign, verified by a dot test). All three methods agree on positive
~+2.3° to +2.7°. The old −2.37° is refuted — most likely a sign error in the
original uncommitted script, compounded by the 0.25-based template. The old
template location also does not reproduce: NCC at (3521, 5276) with the
corrected template is 0.02 (no match) vs 0.45 at (3608, 5360).

### Scale: consistent with 1m
Expected template→ortho scale is 0.9972 (178 px at 2.9919 m/px vs 3.0 m/px
ortho). Recovered 1.0129 → **1.6% residual**, matching the corrected 1m
residual (1.0156 vs 1.0009). The old 5–6% discrepancy is gone; both
resolutions now agree the residual is ~1.6%, plausibly sensor
calibration/projection difference.

### Independent validation (held-out, spatially disjoint)
Correspondences split by position; transform fit on the fit set only (LMEDS),
evaluated on the check set:
- x-split: fit 20 pts → check 16 pts RMSE **7.06 m**
- y-split: fit 15 pts → check 15 pts RMSE **6.39 m**

### Honest claim
These are **cross-registration residuals against LROC absolute truth** (polar
stereographic, co-registered with LOLA), not a calibrated absolute-accuracy
specification. The inlier fit residual is 2.58 m; the independent held-out
bound is **6.4–7.1 m** (wider than 1m's 2.6–3.5 m, as expected on the coarser
grid with small fit sets). The "2.42 m absolute accuracy" originally reported
is withdrawn — it came from the superseded run.

### Verification notes
- SIFT fit identical across 5 RANSAC seeds (46 inliers, 0.8611 px, 1.0129,
  +2.679° every time).
- Template-centre consistency: the recovered transform maps the template
  origin to window (92.3, 87.3) vs (91, 91) expected for a centered 178-px
  template in the 360-px window.
- 1m/3m cross-check (same M1443025251 product): 1m gave 96/164 inliers,
  1.41 m, scale 1.0156, rotation +2.12°; 3m gives 46/46, 2.58 m, scale 1.0129,
  rotation +2.68°. Signs agree, scales agree to 1.6%, residuals scale with
  grid resolution as expected.

---

## Original run (2026-10-08, ~00:35 IST) — SUPERSEDED, kept for provenance

## Milestone
First successful optical-to-optical match of CHANDRA-ALIGN OHRC data against
independent absolute ground truth (LROC NAC orthophoto, polar stereographic,
co-registered with LOLA).

## Data
- Source: ohrc_01 reference crop (2048x2048 at 0.25m/px, downsampled to 171x171 at 3m/px)
- Truth: NAC_DTM_VIKRAMSITE1_M1443025251_3M.IMG (3m/px orthophoto, 2023-07-03 pre-landing)
- Location found via template matching (NCC=0.55) at ortho pixel (3521, 5276)

## Results
| Metric | OHRC↔OHRC (relative) | OHRC↔LROC (absolute) |
|--------|---------------------|---------------------|
| Inliers | 844/5061 (16.7%) | 61/70 (87.1%) |
| Inlier RMSE | 1.61px @0.25m | 0.81px @3m |
| Ground accuracy | — | 2.42 meters |

## Transform (ortho → OHRC)
- Scale: 0.947 (5% off unity — downsampling artifact, under investigation)
- Rotation: -2.37° (plausible: different sensor geometries)
- Translation: (-96.8, -82.9) px (within search window)

## Interpretation
The 87% inlier ratio proves the pipeline aligns OHRC to absolute coordinates
reliably. The 2.42m absolute accuracy is limited by the 3m ortho resolution,
not by pipeline failure. This is a coarse bound; the 1m ortho (downloading)
will tighten it by ~3x.

## What this does NOT prove
- Sub-pixel in OHRC native pixels (0.81px @3m = 9.7px @0.25m)
- The 5% scale discrepancy needs investigation (may be my downsampling, not real)

## Next
1m orthophoto (2GB) downloading. At 1m/px, expect ~0.8m absolute accuracy
if the 0.8px RMSE holds, or better with more precise matching.
