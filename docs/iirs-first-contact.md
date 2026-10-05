# IIRS first contact — overlapping-product attempt

## Current attempt

One IIRS-to-TMC-2 fore-strip pair was prepared from the independently overlapping product `ch2_iir_nci_20231225T1904122779_d_img_d18` and fore strip `ch2_tmc_ncf_20231101T0125121344_d_img_d18`. The IIRS observation ran 2023-12-25 19:04:12–19:15:33 UTC; the TMC-2 observation was 2023-11-01, about 7.7 weeks earlier. The product is calibrated IIRS PDS4 data from PRADAN, downloaded 2026-10-05. Refined corner coordinates, extraction details, and all supplied band statistics are preserved in [iirs_provenance_20231225.json](iirs_provenance_20231225.json).

All three supplied arrays passed the specified data check: float32, `(12842, 250)`, all finite. Measured min/max/mean were band 1 `0 / 1188432.25 / 2675.4414`, band 128 `0 / 746.1122 / 191.1576`, and band 256 `0 / 50625.9414 / 739.3106`; these agree with the supplied rounded references. Band 1 at 712.3 nm was selected because it is the nearest candidate to TMC-2's visible panchromatic response (about 400–850 nm), giving it the strongest visible-structure similarity prior. This is a selection rationale, not a claim that the other bands were tested.

## Overlap and crop geometry

The four refined IIRS corners and the fore-strip PDS4 corners were projected into a south-polar stereographic plane using lunar radius 1737.4 km. Their quadrilaterals intersect with computed projected area **3327.42 km²**. The intersection polygon vertices in projected `(x, y)` km are `(189.908, 524.958)`, `(313.047, 914.569)`, `(317.089, 913.306)`, and `(201.438, 521.446)`. Its geographic envelope is approximately latitude **−71.746° to −58.904°**, longitude **18.896° to 21.122°**.

Mapping that polygon through the product-corner homographies gives these inclusive pixel windows (sample, line):

- IIRS: samples **0–218**, lines **0–4634** (219 × 4635 pixels).
- TMC-2 fore: samples **1721–3999**, lines **111844–189999** (2279 × 78156 pixels).

The TMC-2 label-backed raster was read in 4096-line `rasterio` windows; no full strip or full-resolution overlap crop was assembled in memory. Its window was downsampled to the IIRS crop dimensions using the local corner-derived scale ratio (about 10.62 TMC samples per IIRS sample and 16.80 TMC lines per IIRS line). This brought the scales close without warping either image into the other's projected coordinates. The extraction used explicit `rasterio` windows rather than the repository's `tiled_registration` wrapper. It is a crop-scale approximation; no IIRS ground scale or pixel units were inferred.

The supplied provenance says the extracted NPYs are direct BSQ reads with no resampling, and their layout verifies array axis 0 is line and axis 1 is sample. **The available materials do not independently verify whether stored line 0 is the UL/UR or LL/LR end**: the product XML/geometry sidecar and line-to-ground tie points were not supplied. The crop calculation used the provisional UL/UR-at-line-0 interpretation of the named refined corners. The computed geographic overlap itself is independently established by the corner quadrilaterals, but the line-direction association remains unverified. The failed fit below cannot validate that orientation. This attempt is therefore provisional and does not meet the issue's verification bar for closure.

## Preprocessing and attempt

The selected IIRS crop used the repository's `preprocess_iirs_raster` path with a 2nd–98th percentile stretch and CLAHE clip limit 3.0. The TMC-2 crop used its own 2nd–98th percentile stretch and CLAHE clip limit 3.0. The TMC crop was resampled by the local geometric scale ratio only; both arrays were masked to the projected overlap polygon. These are pixel-space results.

The same crop pair was run through `match_pair_hf` during setup: two captures used the initial environment, where the optional `lightglue` module was absent and the route used its SIFT fallback; the first capture completed matching but failed while serializing its measurements. The official `cvg/LightGlue` package (source commit `eb42fee2d71449efb0aa5c10549752b5d75384d8`) and matching CUDA-compatible feature dependencies were then installed in the task-local venv, and the same pair was run again. In that final route call LightGlue/ALIKED executed and returned 19 correspondences, so the route did not invoke SIFT fallback. The prior SIFT fallback run is retained as its own row for the same arrays. RIFT2 remained opt-in and was not attempted.

| Matcher | Correspondences | Inliers | Quadrants | In-sample RMSE / MAE (px) | Split-held-out RMSE / MAE (px) | Verdict |
| --- | ---: | ---: | --- | ---: | ---: | --- |
| LightGlue/ALIKED | 19 | 4 | 2/4 (1,3,0,0) | 1.1748 / 1.0844 | N/A (only 4 inliers) | DEGENERATE_FAILURE |
| SIFT + Brute-Force (RANSAC fallback) | 25 | 10 | 1/4 (0,10,0,0) | 0.000 / 0.000* | 0.000 / 0.000* (2 check points) | DEGENERATE_FAILURE |

LightGlue's gate reason was RMSE above the 0.50 px sub-pixel limit, fewer than 8 inliers, and a degenerate 2/4-quadrant cluster; only 4 inliers meant split-held-out RMSE/MAE could not be computed. \*The SIFT fallback affine collapsed to scale approximately `1.24×10⁻²⁰`, making its zero residuals meaningless. The gate rejected it for a degenerate spatial cluster, one active quadrant, and entropy 0.0000. Do not interpret the zero telemetry as registration accuracy. Failure details and row-level measurements are in [table_issue12_iirs.csv](../results/table_issue12_iirs.csv).

Neither matcher validated cross-sensor registration: LightGlue's candidate failed the existing gate, and the earlier SIFT fallback was degenerate. The overlap crop's exact line-to-corner orientation still needs independent confirmation before the pair can support a matcher conclusion.

## Fallback path and data limits

These are three pre-extracted candidate bands (direct BSQ read, no resampling). The repository's full-product `.qub` reader was **NOT engaged**. The full 3.29 GB cube is retained by the QA engineer for any full-cube rerun. The 712.3 nm band is the nearest available candidate to TMC-2's visible panchromatic response; no band comparison was performed. The IIRS scale remains unverified in this evaluation.

## First attempt history

The initial attempt used `ch2_iir_nci_20260622T1324243428_d_img_d18` and stopped before matching after the product footprint missed both TMC-2 strips by about 193 km. That no-overlap finding was independently checked and is preserved in the original provenance file [iirs_provenance.json](iirs_provenance.json). No matcher metrics were generated for that product.

## Status

The new product's fore-strip geographic overlap was verified and one unique crop pair was evaluated. LightGlue executed and was rejected; a SIFT fallback run was also captured. The crop's line-0 corner orientation remains unverified from the supplied materials. Issue #12 acceptance criteria are therefore **not met**; keep the issue open pending IIRS label/tie-point verification and a valid registration result.
