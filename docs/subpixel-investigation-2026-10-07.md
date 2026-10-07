# Sub-Pixel Investigation: Why Real Pairs Stall at ~1.6px — 2026-10-07

## Question
Can the pipeline achieve sub-pixel (<0.5px) RMSE on real OHRC pairs?
Current: 1.54-1.80px (COARSE). Synthetic: 0.37px (ACCEPT).

## Method
Direct dissection of SIFT+RANSAC on ohrc_01 benchmark crops (2048x2048),
with exact residual analysis. No gate changes, no cherry-picking.

## Findings

### 1. Low inlier ratio is the core problem
5061 Lowe matches → 844 RANSAC inliers (16.7%) at 3.0px threshold.
Only 11% of inlier residuals <0.5px; 31% <1.0px. The texture (repetitive
craters, low-sun shadows) produces massive matcher ambiguity.

### 2. Systematic position-dependent error exists but isn't fixable by model
72% of x-residual variance is position-dependent (R²=0.716). But:
- 6-DOF full affine: 1.62px (same as 4-DOF), 4x more inliers
- 8-DOF homography: 9.08px (overfits catastrophically)
- Piecewise 2x2: 1.74px (worse — boundary effects)
Conclusion: model complexity is NOT the bottleneck.

### 3. Matcher tuning doesn't help
- RootSIFT vs SIFT: identical (1.68-1.77px vs 1.61-1.86px)
- Lowe's ratio 0.70/0.75/0.80: no improvement
- Wallis local normalization: WORSE (1.61→1.74px, fewer inliers)

### 4. Tight subset reaches sub-pixel but doesn't generalize
Tightest 10% (84 pts): 0.29px self-consistency — but transform evaluated
on full inlier set stays at 1.61px. The 1.6px reflects real spread in the
data, not a fixable bias. Cherry-picking the tight set would be dishonest.

### 5. Inlier RMSE overestimates true transform error (from scale-ratio test)
On synthetic data with exact truth: inlier RMSE 0.53px corresponded to
truth RMSE 0.0097px (50x overestimation). The reported 1.6px may
substantially overstate the actual transform error — but we cannot prove
it without absolute ground truth.

## Conclusion
SIFT on real lunar imagery has a ~1.6px inlier-RMSE floor on this data.
This is a matcher/texture limit, not a model or tuning problem. The tested
interventions (higher-DOF models, RootSIFT, stricter ratios, Wallis,
tight-subset refit) all fail to break it.

## Path to legitimate sub-pixel claim
1. **Prove transform accuracy via absolute truth** (LROC adjudication):
   if transform error vs absolute coords is sub-pixel, the claim holds
   regardless of inlier RMSE. This is the highest-leverage path.
2. **Dense learned matcher** (LoFTR-style): may handle lunar texture better
   than sparse SIFT, but needs GPU + validation.
3. **Do NOT**: lower gates, cherry-pick tight subsets, or report inlier
   RMSE as transform accuracy. The 1.6px is honest; the claim must be proven,
   not manufactured.

## Data
- Test images: data/benchmark_crops/ohrc_01_{reference,source}.png
- All runs: cv2.setRNGSeed(7), SIFT nfeatures=8000, Lowe 0.75,
  RANSAC 3.0px / 5000 iters / 0.995 confidence.
