# Spatial-Coverage Experiment: Can Forced Detection Fill the Empty Q1 Quadrant? — 2026-10-08

## Question

Every arm on the real OHRC `ohrc_01` pair has **zero inliers in Q1 (top-left)** —
baseline counts 0/411/45/388 — so a quarter of the frame is pure extrapolation,
and spatial held-out error (6.14/4.73px) dwarfs inlier RMSE (1.61px). The
refinement study's own conclusion: "the empty Q1 quadrant is a bigger lever
than refinement." Default SIFT finds nothing matchable there. This experiment
forces it: multi-scale detection, denser detection, and both combined.

## Pre-registered success criterion

**COVERAGE WIN** requires all three: (1) Q1 gains inliers, (2) the transform
agrees with baseline (|Δrot| ≤ 0.05°, |Δscale| ≤ 0.001), (3) held-out improves
on ≥1 split vs baseline without worsening the other by >0.5px. Coverage
without geometric agreement is decoration: arms that fill Q1 but disagree
with baseline or worsen held-out are FAILED arms, never wins.

## Method

Baseline pipeline reproduced exactly from `scripts/refine_subpixel_experiment.py`:
SIFT(nfeatures=8000) / Lowe 0.75 / RANSAC partial-affine (3.0px, 5000 iters,
0.995 confidence), seed 7; spatial x/y-split held-out (RANSAC-fit on a 40%
band of inlier source-x (resp. y), check on a disjoint 40% band). Quadrants and
entropy use the repo's own `chandra_align.metrics.quadrant` convention
(Q1=Top-Left, entropy = −Σ p·log₂p over non-empty quadrants). Frozen gates:
ACCEPT needs RMSE ≤ 0.50px, ≥8 inliers, entropy ≥ 0.75, ≥3 quadrants;
COARSE needs RMSE ≤ 2.50px, ≥8 inliers, entropy ≥ 0.50, ≥2 quadrants.

- **a_baseline** — as above. Must reproduce 5061/844/1.6087px/Q1=0 or stop.
- **b_multiscale** — Gaussian pyramid at 0.5x/1x/2x, SIFT(8000) per level,
  keypoint coordinates mapped back to full-res, merged, deduped (<1px),
  single RANSAC on the merged set.
- **c_dense** — SIFT(nfeatures=20000, contrastThreshold=0.01, edgeThreshold=20).
- **d_combined** — pyramid 0.5x/1x/2x with dense params per level
  (nfeatures=12000, contrastThreshold=0.02, edgeThreshold=15). Ran because
  b and c both completed without error.

## Results

| Arm | Corr | Inliers | Inlier RMSE | Scale | Rot | Q1/Q2/Q3/Q4 | Entropy | Gate | Held-out x / y |
|-----|-----:|--------:|------------:|------:|----:|------------:|--------:|:-----|:---------------|
| a_baseline | 5061 | 844 | 1.6087px | 0.99785 | 0.973° | 0/411/45/388 | 1.246 | COARSE | 6.135 / 4.732 |
| b_multiscale | 10997 | 1870 | 1.6612px | 0.99786 | 0.968° | 0/900/109/861 | 1.262 | COARSE | 4.413 / 3.067 |
| c_dense | 12480 | 1974 | 1.6136px | 0.99624 | 1.041° | 0/803/177/994 | 1.338 | COARSE | 3.036 / 3.445 |
| d_combined | 16305 | 2872 | 1.7156px | 0.99683 | 1.008° | 0/1213/244/1415 | 1.331 | COARSE | 2.419 / 3.543 |

Full per-arm numbers (transform agreement, fit/check sizes, keypoint counts):
`results/table_coverage.csv`.

**Q1 status: still zero inliers in every arm** — including d_combined with
2,872 inliers from 16,305 correspondences across three scales. The
pre-registered criterion fails at its first clause. **No coverage win.**

## Transform agreement (the trap guard)

- b_multiscale agrees with baseline (Δrot 0.0045°, Δscale 0.00000) — same
  solution, more of it. Held-out improves on both splits (6.14→4.41,
  4.73→3.07) but Q1 is still empty, so it is not a win.
- c_dense **fails the agreement guard**: Δrot 0.0678° (>0.05°),
  Δscale 0.00161 (>0.001), translation shifted 3.34px. Forcing keypoints in
  texture-poor regions added ambiguous matches that pulled the fit to a
  different solution — exactly the trap the brief warned about. Reported as
  a FAILED arm, not a win, despite better held-out.
- d_combined narrowly fails the scale guard (Δscale 0.00103 > 0.001;
  Δrot 0.0346° within tolerance). Also not a win.

## Why Q1 is empty: mechanism diagnostic

Q1 emptiness is **not a detection problem**. Keypoint counts per quadrant are
roughly uniform (dense settings: ref Q1 has 4,744 of 20,000 keypoints, src Q1
4,895). The dense arm produced **2,165 Lowe-passing matches in Q1** — but zero
RANSAC inliers, with median residual **19.1px** under the global transform.

The matches are not random garbage either: RANSAC on Q1 matches alone finds
**884/2165 inliers (41%) at 1.75px RMSE** — a coherent local solution that
disagrees with the global one. Per-quadrant partial-affine fits (dense
correspondences, seed 7):

| Region | Inliers | RMSE | Scale | Rot | tx | ty |
|--------|--------:|-----:|------:|----:|---:|---:|
| global | 1974 | 1.61px | 0.99624 | 1.041° | 137.3 | 354.5 |
| Q1 (TL) | 884 | 1.75px | 0.97827 | 0.170° | 147.7 | 384.6 |
| Q2 (TR) | 1266 | 1.79px | 0.99295 | 0.302° | 136.8 | 369.0 |
| Q3 (BL) | 790 | 1.76px | 1.00487 | 0.384° | 126.5 | 367.2 |
| Q4 (BR) | 974 | 1.55px | 0.99799 | 0.973° | 133.9 | 353.9 |

Q1's local transform differs enormously from the global fit (scale 0.978 vs
0.996, ty 384.6 vs 354.5 — 30px). Each quadrant is internally consistent at
~1.6–1.8px but lives under a **different local transform**. No single
partial-affine can represent the frame; global RANSAC necessarily rejects all
Q1 matches as outliers. This is direct, quantified evidence for the
spatially-varying distortion hypothesis: the "empty quadrant" is a model
limitation, not a coverage limitation.

## Verdict

**Pre-registered success criterion: NOT MET.** Forcing detection
(multi-scale, dense, combined) cannot fill Q1 because Q1 was never empty of
features — it is empty of features consistent with a single global
partial-affine. More keypoints only add ambiguous matches that can drag the
fit (c_dense demonstrably did). The coverage thread is closed: the remaining
lever is a distortion model richer than partial-affine (e.g. a regularized
dense deformation field), scored on spatial held-out — never on inlier RMSE.

## Reproduction

- Script: `scripts/coverage_multiscale_experiment.py` (deterministic, seed 7;
  baseline asserted in-script: 5061/844/1.6087px/Q1=0).
- Table: `results/table_coverage.csv` (4 rows: baseline + 3 arms).
- Per-quadrant diagnostic was a follow-up analysis on the dense arm's
  correspondences (same seed, deterministic); numbers above are exact.
