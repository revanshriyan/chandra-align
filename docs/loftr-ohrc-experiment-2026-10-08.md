# LoFTR (Detector-Free Dense) on the Real OHRC Pair — 2026-10-08

## Question
LoFTR is the one matcher family never tried on this data (pipeline is 100%
SIFT+BF in practice; RIFT2 yields 0 correspondences; LightGlue/ALIKED need
GPU). A peer-reviewed ISPRS 2026 paper (Hou et al., Tianwen-1 HiRIC vs CTX —
same no-GCP regime) found LoFTR gives 3–4x the correspondences of LightGlue
hybrids, roughly halves outlier rate, and ~10% better BA RMSE, with SIFT
failing completely on one block. Can dense detector-free matching break the
1.61px floor on the real OHRC pair?

## Pre-registered success criterion
Beat the ECC-refined SIFT 1.386px inlier RMSE with transform agreement to
the SIFT baseline (drot ≤ 0.05°, dscale ≤ 0.001) AND held-out not worse —
or fill Q1 (top-left on the source frame, 0/411/45/388 under SIFT) with an
agreeing transform. Inlier-RMSE-only wins without held-out support are
cosmetic. **Verdict: NOT MET** (see Results).

## Method
- `kornia.feature.LoFTR(pretrained='outdoor')` (MegaDepth weights), CPU-only,
  eval mode. Installed torch 2.14.1+cpu / kornia 0.8.3 in a venv.
- CPU feasibility: full 2048² coarse attention is infeasible, so LoFTR runs
  at 1024 (long side, 63–79 s inference) and keypoints are scaled back ×2 to
  full resolution. This is a documented compromise, not hidden: the dense
  coverage comes from LoFTR, localization is bounded by the 1024 grid.
- Filtering: LoFTR has no Lowe ratio (dense). kornia applies its internal
  0.2 coarse-match threshold; all 11,048 returned matches already satisfy
  conf ≥ 0.2 (observed conf range 0.202–1.000), so there is no further
  threshold sensitivity to explore — the returned set IS the thresholded set.
- Arms: (a) SIFT baseline reproduction; (b) LoFTR raw scaled to full res;
  (c) LoFTR + A9's full-resolution patch-ECC refinement (translation-only,
  31px patch, 5.8px shift guard) for sub-pixel localization.
- RANSAC partial-affine (3.0px, 5000 iters, 0.995), `cv2.setRNGSeed(7)`
  before every fit; frozen gates; A9's spatial x/y held-out (fit 40% band,
  check disjoint 60%+ band). Quadrant counts reported on both frames: p2
  (source, matches A9's 0/411/45/388 convention) primary, p1 (reference)
  for completeness.
- LoFTR inference is deterministic in eval mode (no dropout); not
  seed-controlled beyond `torch.manual_seed(7)`. RANSAC is seeded.

## Results (seed 7, frozen gates)

| Arm | Corr | Inliers | Inlier RMSE | Scale | Rot (°) | Q1/Q2/Q3/Q4 (src) | Ent | Held-out x / y |
|-----|-----:|--------:|------------:|------:|--------:|-------------------|-----|----------------|
| a_sift_baseline | 5061 | 844 | 1.6087px | 0.99785 | 0.973 | 0/411/45/388 | 1.246 | 6.14 / 4.73px |
| b_loftr_raw | 11048 | 1612 | 1.6967px | 0.99846 | 0.963 | 0/765/76/771 | 1.227 | 5.04 / 4.74px |
| c_loftr_ecc | 11048 | 1666 | 1.6805px | 0.99834 | 0.932 | 0/773/110/783 | 1.285 | 7.71 / 4.74px |

All arms COARSE-tier. Baseline reproduced exactly (5061/844/1.6087px,
quadrants 0/411/45/388).

### Transform agreement (LoFTR vs SIFT baseline)
- Raw: drot 0.0099°, dscale 0.00060, dtrans 0.52px — **agrees** (within bar).
- ECC: drot 0.0410°, dscale 0.00048, dtrans 1.01px — **agrees** (within bar).
- Same geometric solution, not a different texture lock and not a dragged
  fit. The translation differs ~0.5–1.0px, within the fit's own ~1.6px noise.

### Q1 status
Q1 (source frame) has **0 inliers in both LoFTR arms** — 11,048 dense
correspondences and the quadrant is still empty under global RANSAC. (On the
reference frame Q1 shows 40/9 inliers vs SIFT's 19; the source-frame
emptiness is partly the (134,354)px shift, but the global fit still rejects
everything there.)

### Held-out
- Raw: x 6.14→5.04px (better), y 4.73→4.74px (flat).
- ECC: x 6.14→7.71px (**worse**), y flat.
- Mixed; the pre-registered "not worse on both" bar fails for the ECC arm.

## Interpretation: convergent evidence the floor is the model, not the matcher
LoFTR is now the fourth matcher family tested on this pair
(SIFT, SIFT+LK, SIFT+ECC, LoFTR, LoFTR+ECC). Findings:

1. **Dense coverage does not move the floor.** 2.2x the correspondences
   (11,048 vs 5,061), 1.9x the inliers (1,612 vs 844) — inlier RMSE 1.70px
   vs 1.61px (worse). Outlier-rate halving (the ISPRS headline) does not
   translate into a better geometric fit here because the residuals are not
   outlier-dominated; they are distortion-dominated.
2. **Q1 stays empty under global RANSAC despite dense matches.** This
   independently reproduces the spatial-coverage experiment's mechanism
   finding: Q1 follows a coherent *different* local solution (per-quadrant
   fits mutually inconsistent; no single partial-affine represents the
   frame). A global partial-affine cannot use Q1 no matter how dense the
   correspondences — the quadrant is rejected, not starved.
3. **ECC refinement barely helps LoFTR** (1.6967→1.6805px, −1%) vs −14% on
   SIFT. LoFTR's fine-level matches are already at their localization limit
   for this texture at 1024; the remaining residual is distortion, which
   translation-only patch refinement cannot touch.
4. **No trap-drag.** Unlike dense multi-scale SIFT (which dragged the global
   fit to a disagreeing transform), LoFTR's global solution agrees with
   SIFT's to 0.01–0.04° / 0.0005 scale. The agreement across four matcher
   families strengthens confidence that the recovered global transform is
   the right *global* answer — it is just not a sufficient *model*.

## Why the LROC stretch was skipped
The brief allowed the 1m LROC stretch only if the pair result were clean.
It is a clean *negative*: LoFTR underperforms SIFT on the same-sensor pair,
and the cross-sensor re-adjudication already showed refinement-style
polishing does not transfer across sensors. Running a MegaDepth-trained
dense matcher on the cross-sensor grid would be compute without a
motivating hypothesis.

## Failure modes / implementation notes
- kornia 0.8.3's `LoFTR` returns **unbatched** `(N,2)`/`(N,)` tensors —
  indexing `[0]` silently yields one keypoint, not the batch. Verified
  against an identity pair (784 matches at 256²) before trusting outputs.
- 1024 is the practical CPU ceiling here (63–79 s/pair); 2048² coarse
  attention (65k tokens) is infeasible on CPU.
- LoFTR's internal coarse threshold (0.2) is the effective filter; no
  additional confidence tuning applies to the returned set.
- Determinism note: LoFTR weights are fixed and eval-mode deterministic;
  the RANSAC stage is seeded (cv2.setRNGSeed(7)). Exact bit-reproducibility
  of the neural stage across machines is not claimed.

## Bottom line
LoFTR — the most credible untried matcher — does not break the floor
(1.68–1.70px vs 1.61px SIFT, 1.39px SIFT+ECC), does not fill Q1, and agrees
with SIFT's transform. Four matcher families, one floor: the bottleneck is
the partial-affine model against spatially-varying distortion, not
correspondence quality or coverage. The remaining lever is a distortion
model richer than partial-affine, scored on spatial held-out.
