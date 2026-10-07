# PC+SIFT on Real OHRC Pair: Honest Result — 2026-10-07

## Test
Phase-congruency + SIFT on ohrc_01 real pair (2048x2048, same-day acquisition,
~7° solar azimuth difference).

## Result
| Method | Correspondences | Inliers | Inlier ratio | RMSE |
|--------|-----------------|---------|--------------|------|
| Raw SIFT | 5061 | 844 | 16.7% | 1.6087px |
| PC+SIFT | 861 | 125 | 14.5% | 1.7035px |

PC+SIFT is WORSE on this pair (-5.9% RMSE, 7x fewer inliers).

## Interpretation
The real pair has similar illumination (2hr apart), so PC's illumination
invariance adds no value. PC maps discard intensity texture that SIFT uses
for distinctive matching, resulting in fewer, weaker correspondences.

## Where PC helps (from sun-angle pilot)
- Azimuth difference >10°: raw SIFT DEGENERATE, PC+SIFT works to 30°+
- Cross-day or cross-season pairs with different sun angles
- NOT for same-day pairs where illumination is already similar

## Conclusion
PC is a specialist tool for illumination-varying pairs, not a general
improvement. The 1.6px floor on same-day pairs is a texture/matcher limit
that PC does not address. Both results are honest and define the
operational envelope precisely.

## Recommendation
- Same-day pairs (<10° az diff): use raw SIFT (current pipeline)
- Cross-illumination pairs (>10° az diff): use PC+SIFT front-end
- Gate thresholds unchanged for both paths
