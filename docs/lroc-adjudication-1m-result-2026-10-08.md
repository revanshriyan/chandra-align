# 1m Adjudication: OHRC vs LROC Ortho — 2026-10-08

## Correction (2026-10-08, ~01:30 IST) — re-run with verified geometry

Two errors were found in the original run below and corrected:

1. **Wrong file offset.** The extraction used byte offset 47286. The PDS3 label
   embedded in the file states `RECORD_BYTES=46006`, `LABEL_RECORDS=1`,
   `^IMAGE=2`; the detached XML (`/tmp/ortho1m.xml`) states
   `<offset unit="byte">46006</offset>`; file-size arithmetic confirms it
   (2193750104 − 47683×23003×2 = 46006 exactly); the first bytes decode as
   ASCII PDS label text. Correct offset is **46006**. The 1280-byte (= 640
   sample) error shifted every extracted pixel, so the original template
   location and all match statistics came from a mis-registered extraction.
2. **Wrong OHRC pixel scale.** The run assumed 0.25 m/px. The OHRC PDS4 label
   states `<isda:pixel_resolution unit="m/pixel">0.26</isda:pixel_resolution>`
   (verified in the product XML). The 2048-px crop was therefore resampled to
   532×532 px at 1 m/px, not 512×512.

### Corrected results
| Metric | Original (superseded) | Corrected |
|--------|----------------------|-----------|
| File offset | 47286 (wrong) | 46006 (verified 3 ways) |
| OHRC scale used | 0.25 m/px | 0.26 m/px (from PDS4 XML) |
| Template location (ortho px) | (9922, 15825) | **(10818, 16081)**, NCC 0.475 (0.618 after +2.1° rotation) |
| SIFT inliers | 171/274 (62.4%) | **96/164 (58.5%)** (nfeatures=8000, CLAHE, Lowe 0.75, RANSAC 3.0px) |
| Inlier RMSE | 1.47 px = 1.47 m | **1.41 px = 1.41 m** |
| Scale (ortho→OHRC) | 0.9395 | **1.0156** |
| Rotation | −2.43° | **+2.12°** |
| Entropy / quadrants | — | 4-quadrant entropy **1.36** (counts TL/TR/BL/BR = 1/7/50/38 — bottom-heavy but all 4 quadrants active; 16-cell grid entropy 2.41 is diagnostic only). Passes COARSE gates (RMSE 1.41 ≤ 2.50, 96 ≥ 8 inliers, entropy 1.36 ≥ 0.50, 4 ≥ 2 quadrants); not ACCEPT (RMSE 1.41 > 0.50) |

### Scale discrepancy: resolved in large part
With 0.26 m/px OHRC, both the template (532 px) and the ortho are nominally at
1 m/px, so the expected scale is 1.0. Recovered scale is 1.0156 — a **1.6%
residual**, down from the 6% originally reported. The old 6% figure was
contaminated by the offset error (it matched mis-shifted content) and the
0.25 m/px assumption; it is superseded. The remaining 1.6% is plausibly sensor
calibration / projection difference, not pipeline bias.

### Independent validation (held-out, spatially disjoint)
Correspondences were split by position into a fit set and a disjoint check
set; the transform was fit on the fit set only (LMEDS) and evaluated on the
check set:
- x-split: fit 28 pts → check 35 pts RMSE **2.58 m**
- y-split: fit 24 pts → check 48 pts RMSE **3.46 m**

### Honest claim
These are **cross-registration residuals against LROC absolute truth** (polar
stereographic, co-registered with LOLA), not a calibrated absolute-accuracy
specification. The inlier fit residual is 1.41 m; the independent held-out
bound is **2.6–3.5 m**. The "1.47 m absolute accuracy" originally reported is
withdrawn — it came from the mis-registered extraction.

### Verification notes
- SIFT fit is stable across 5 RANSAC seeds (identical 96 inliers, RMSE, scale,
  rotation).
- Inlier correspondences confirmed real: median 21×21 patch NCC 0.84 for
  inlier pairs vs 0.00 for random pairs.
- Rotation +2.12° corroborated by three independent checks: NCC rotation sweep
  (best at +2.1°, NCC 0.618), ECC Euclidean alignment (+2.6°), template-centre
  mapping lands at (447.8, 453.5) vs window centre (450, 450).
- 3m ortho offset verified correct at 15336 bytes (label + file arithmetic);
  see note in docs/lroc-adjudication-first-result-2026-10-08.md.

---

## Original run (2026-10-08, ~00:45 IST) — SUPERSEDED, kept for provenance

## Results
| Metric | 3m ortho | 1m ortho |
|--------|----------|----------|
| Inliers | 61/70 (87.1%) | 171/274 (62.4%) |
| RMSE (px) | 0.81px @3m | 1.47px @1m |
| Absolute accuracy | 2.42 meters | 1.47 meters |
| Scale | 0.947 | 0.940 |
| Rotation | -2.37° | -2.43° |

## Key Findings
1. **Absolute accuracy improves with resolution**: 2.42m → 1.47m (1.6x better)
2. **Inlier ratio drops**: 87% → 62% (more detail = more matcher ambiguity)
3. **Scale discrepancy persists**: ~6% (0.94 vs 1.0) at both resolutions — systematic, not noise
4. **Rotation consistent**: -2.4° at both resolutions (real sensor geometry difference)

## Scale Discrepancy Investigation Needed
The 6% scale offset appears at both 3m and 1m, suggesting a systematic cause:
- OHRC pixel scale may not be exactly 0.25m/px (could be ~0.265m/px)
- Projection differences between OHRC (product geometry) and LROC (polar stereographic)
- My downsampling introduces bias (less likely, consistent across methods)

## Sub-pixel Status
1.47m = 5.9 OHRC pixels (at 0.25m/px). Not sub-pixel, but this is the first
absolute bound. The error is dominated by:
- Ortho resolution limit (1m pixels)
- Scale discrepancy (6% = 0.06 * 512m = 30m across the crop!)

If the scale discrepancy is resolved, the residual RMSE may drop significantly.

## Data
- 1m ortho: NAC_DTM_VIKRAMSITE1_M1443025251_100CM.IMG (47683x23003, 2.04GB)
- Location: (9922, 15825) via template NCC=0.44
- SIFT: nfeatures=4000, Lowe 0.75, RANSAC 3.0px
