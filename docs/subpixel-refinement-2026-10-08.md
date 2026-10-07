# Sub-Pixel Refinement Experiment: Can LK/ECC Break the 1.6px Floor? — 2026-10-08

## Question
docs/subpixel-investigation-2026-10-07.md proved the ~1.6px inlier-RMSE floor
on the real OHRC ohrc_01 pair is not a model, matcher-tuning, or estimator
problem. Remaining hypothesis: the floor is SIFT keypoint localization noise
on lunar texture, and refining each correspondence to sub-pixel
(Lucas-Kanade / ECC on patches) tightens residuals without changing the
transform.

## Pre-registered success criterion
Real-pair inlier RMSE < 0.50px AND refined transform agrees with baseline
(~0.1px translation / 0.05° rotation / 0.001 scale) AND held-out error not
worse than baseline.

## Method
- Baseline reproduced exactly: SIFT nfeatures=8000, Lowe 0.75, partial-affine
  RANSAC 3.0px / 5000 iters / 0.995 confidence, cv2.setRNGSeed(7) →
  5061 matches, 844 inliers, **1.6087px** (matches the documented floor).
- Sanity control FIRST on a synthetic pair (reference warped by exact shift
  (7.3, −3.9)px): refinement must not degrade truth RMSE, else stop.
- Arms (refinement applied to the 844 baseline inlier pairs, then RANSAC +
  the frozen gate re-run):
  - (a) baseline, unrefined
  - (b) LK: cv2.calcOpticalFlowPyrLK(ref, src, prevPts=p1, nextPts=p2 init,
    winSize=(21,21), maxLevel=3, criteria=(EPS|COUNT, 40, 0.03),
    flags=OPTFLOW_USE_INITIAL_FLOW); status==0 → keep original (12 fallbacks)
  - (c) ECC: per-inlier 31×31 patches, cv2.findTransformECC(ref_patch,
    src_patch, eye(2,3), MOTION_TRANSLATION, criteria=(EPS|COUNT, 50, 1e-4));
    p2' = p2 + warp[:,2]; on failure/out-of-bounds → keep original
    (10 fallbacks)
- Held-out per arm: spatial x-split and y-split (fit on 40% band, check on
  disjoint 40% band), same RANSAC fitter, seed 7.
- Gate verdicts from the repo's own frozen gate
  (chandra_align.metrics.quadrant.validate_registration_gate).
- Full data: results/table_subpixel_refinement.csv. Repro script:
  scripts/refine_subpixel_experiment.py.

## Results

| Arm | Inliers | Inlier RMSE | Scale | Rot | Gate (frozen) | Held-out x / y |
|-----|--------|-------------|-------|-----|---------------|----------------|
| (a) baseline | 844/5061 | 1.6087px | 0.99785 | 0.973° | COARSE_ADVISORY | 6.14 / 4.73px |
| (b) LK | 757/844 | 1.4780px (−8%) | 0.99812 | 0.969° | COARSE_ADVISORY | 5.25 / 3.20px |
| (c) ECC | 660/844 | 1.3860px (−14%) | 0.99794 | 0.981° | COARSE_ADVISORY | 4.80 / 2.68px |

Synthetic control: baseline truth RMSE 0.0125px → LK 0.0096px, ECC 0.0112px.
Refinement helps slightly and never hurts on exact truth — the method is
valid; the real-pair result below is about the data, not the method.

### Transform agreement (refined vs baseline)
- Rotation: within 0.004° (LK) / 0.008° (ECC). Scale: within 0.0003 / 0.0001.
  Same geometric solution — refinement did not lock onto different texture.
- Translation: 0.28px (LK) / 0.46px (ECC), outside the pre-registered 0.1px
  bar. Given ~1.5px noise with strong spatial systematics (see below), this
  is within the fit's own uncertainty; the rotation/scale agreement is the
  stronger signal that the solution is unchanged.

### Held-out
Refinement improves held-out error in every split (x: 6.14→4.80,
y: 4.73→2.68). Genuine, not overfitting. But absolute held-out values
(2.7–6.1px) dwarf inlier RMSE (1.4–1.6px): the global partial-affine fit on
one side of the frame predicts the other side poorly — direct evidence of
spatially-varying distortion the model cannot represent.

### Gate verdicts (frozen gate, repo's own function)
All three arms: COARSE_ADVISORY. Quadrant counts (baseline):
Q1/Q2/Q3/Q4 = 0/411/45/388, entropy 1.25 — the top-left quadrant has ZERO
inliers in every arm; the transform is extrapolated over a quarter of the
frame. Refined arms are similarly distributed.

## Verdict on the success criterion: NOT MET
Best inlier RMSE is 1.386px (ECC) — a 14% tightening, nowhere near 0.50px.
No gate tier change in any arm.

## Interpretation: the floor is not (mainly) localization noise
The synthetic control is the decisive contrast: on a pure-shift pair,
refinement denoises localization as expected (0.0125→0.0096px). On the real
pair it barely moves the needle (1.61→1.39px). If SIFT localization noise
were the dominant term, sub-pixel refinement would have crushed the floor.
It didn't — so the dominant terms are the ones refinement cannot touch:
unmodeled spatially-varying distortion (position-dependent residual
variance R²=0.72 per the investigation; held-out 2.7–6.1px vs inlier
1.4–1.6px here) plus texture ambiguity. Refinement is worth keeping as a
polishing step (≈14% tighter residuals, better held-out, same transform),
but it is not a floor-breaker and does not change any verdict.

## Failure modes found (for future work)
- LK can catastrophically mistrack on repetitive crater texture: one
  correspondence shifted 1040px with status==1 (mean residual under the
  baseline transform blew to 3.89px while the median stayed 1.36px). Any
  production use of LK refinement needs a shift-magnitude guard; status
  flags alone do not catch it. ECC (patch-constrained, max shift 5.8px)
  did not exhibit this.
- Refinement pushed marginal inliers out, not in: the 87 (LK) / 184 (ECC)
  dropped inliers had 2.1–2.3px baseline residuals (near the 3px RANSAC
  threshold) and landed at 2.7–2.9px after refinement. Ambiguous matches
  refine to wrong local texture.

## What to try next
1. The empty Q1 quadrant is a bigger lever than refinement: no inliers at
   all in the top-left means a quarter-frame extrapolation. Improving
   spatial coverage (more aggressive detection in texture-poor regions,
   or multi-scale SIFT) attacks the held-out problem directly.
2. The actual bottleneck is spatially-varying distortion — but piecewise
   models were already tried and failed (boundary effects). A regularized
   dense deformation field is the remaining model direction, with
   held-out (not inlier) RMSE as the only honest score.
3. The legitimate sub-pixel path remains absolute truth: re-run the LROC
   adjudication with ECC-refined correspondences and see whether the
   1.41m / 2.6–3.5m bounds tighten. Claim only what the absolute number
   supports — never inlier RMSE.
