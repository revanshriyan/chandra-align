# 1m Adjudication: OHRC vs LROC Ortho — 2026-10-08

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
