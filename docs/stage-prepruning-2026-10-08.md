# Stage-Before-Pruning Experiment: can the field deliver in-pipeline? — 2026-10-08

## Question

A15 found the opt-in stage honestly declines on ohrc_01's true pipeline path
(19 pruned inliers → `no_improvement`), while A14's headline (1.61→0.41 px,
SUCCESS_SUBPIXEL) was measured on the pre-pruning set. Option (i) from the
multi-pair doc: run the stage *before* the pruning step, on the larger
RANSAC inlier set. Does the win transfer into the ordered pipeline?

## Premise correction (measured, not assumed)

The brief supposed the true pipeline's pre-pruning RANSAC set was ~842
inliers (A14's standalone number). Capturing the ACTUAL set flowing into
`refine_subpixel_ncc` inside `_align_core` (wrappers + assertion that it
equals the last RANSAC inlier mask) shows:

| Pair | Pre-NCC ("rich") set | Post-NCC-refit (final) |
|------|---------------------:|-----------------------:|
| ohrc_01 | 19 | 19 |
| ohrc_02 | 34 | 16 |
| ohrc_03 | 24 | 14 |
| ohrc_05 | 10 | 10 |

The quadrant-balance cap (50/quadrant) + 8×8 grid bucketing (8/bucket) +
RANSAC already starved the set: RANSAC on a spatially *spread* set under
spatially-varying distortion keeps only a locally-consistent cluster
(compare: unbucketed RANSAC gives 842/2153/1436/415 inliers on the same
pairs). The "larger RANSAC inlier set" the brief meant does not exist at
that point in the pipeline. The experiment therefore has two parts.

## Method

**Part A (as specified):** stage on the pre-NCC set (19–34 pts) via the
real `chandra_align.deform_field.apply_deform_field_stage` (default λ grid,
seed 7); if applied, grid-bucket (8×8, 8/bucket) + NCC-refine (window 11,
search 1, on the pipeline's captured CLAHE-processed images) + RANSAC refit
(3 px, `cv2.setRNGSeed(7)`, documented) the *field-corrected* points
(`cref = M @ sec + F(sec)`, field refit at the chosen λ with the stage
module's own functions and cross-checked to 1e-6 against the stage's
returned residuals); then the frozen gate. The refit affine is a pruning
vehicle only; the gate scores field-corrected residuals under (M_rich, F).

**Part B (the reformulation the brief actually meant):** stage on the
UNBUCKETED RANSAC set (A14's proven `sift_match`/`ransac_fit` port:
5061/842/1.6087 px reproduced on ohrc_01 with assertions), i.e. *before*
the quadrant-balance + bucketing + NCC chain; then run that full chain on
the corrected points; then the frozen gate.

**Gate basis** (both parts, mirrors the current hook): gate RMSE = the
stage's internal stride-80/20 field held-out (generalization); inlier
count / quadrants / entropy from the final pruned set. Frozen thresholds:
ACCEPT ≤0.50 px / ≥8 inl / entropy ≥0.75 / ≥3 quads;
COARSE ≤2.50 px / ≥8 inl / entropy ≥0.50 / ≥2 quads.

**Reproduction gate (in-script, STOP on mismatch):** A15's true-path
numbers reproduced exactly first — ohrc_01: 19 inliers / 1.3920 px /
gate 1.6445, stage declines (`no_improvement`); ohrc_02/03/05: 16/14/10
inliers. Determinism re-check: both new orders byte-identical on re-run.

## Results

Full table: `results/table_stage_prepruning.csv` (20 rows: old_off, old_on,
new_A, new_B per pair).

### Part A: stage before NCC pruning (on the 19–34 pt sets)

| Pair | Stage | Verdict | Inliers | Inlier RMSE | Gate RMSE |
|------|-------|---------|--------:|------------:|----------:|
| ohrc_01 | declined (`no_improvement`) | COARSE | 19 | 1.3920 | 1.6445 |
| ohrc_02 | applied (λ=0.1) | COARSE | 11 | 0.4838 | 1.2560 |
| ohrc_03 | applied (λ=0.1) | COARSE | 10 | 0.3080 | 1.0241 |
| ohrc_05 | applied (λ=0.1) | COARSE | 10 | 0.0000 | 1.3832 |

No verdict changes vs old order. The field engages on three pairs and
crushes *inlier* RMSE (0.31–0.48 px, even 0.00 px on ohrc_05's 10 points —
pure interpolation on a tiny set, honestly scored), but the held-out
basis the gate uses stays above 0.50 px everywhere. Zero downgrades, no
folding (min Jacobian det ≥ 1.004 — the field is nearly rigid on these
small sets), deterministic.

### Part B: stage before ALL pruning (on the unbucketed sets)

| Pair | Unbucketed inliers | Stage | Verdict | Inliers | Inlier RMSE | Gate RMSE | Quadrants |
|------|-------------------:|-------|---------|--------:|------------:|----------:|:----------|
| ohrc_01 | 842 | applied (λ=0.1) | **SUCCESS_SUBPIXEL** | 30 | 0.4232 | 0.4536 | 5/10/6/9 |
| ohrc_02 | 2153 | applied (λ=0.1) | **SUCCESS_SUBPIXEL** | 33 | 0.4384 | 0.4516 | 11/0/14/8 |
| ohrc_03 | 1436 | applied (λ=0.1) | **SUCCESS_SUBPIXEL** | 28 | 0.4755 | 0.3443 | 0/10/8/10 |
| ohrc_05 | 415 | applied (λ=0.1) | COARSE_ADVISORY | 24 | 0.4055 | 0.5252 | 10/8/0/6 |

3 of 4 pairs reach **SUCCESS_SUBPIXEL** through the ordered pipeline;
ohrc_05 lands at 0.5252 px — 0.03 px above the line (borderline COARSE).
Zero downgrades vs old order on all pairs. No folding anywhere (min
Jacobian det 0.978–1.002). Deterministic on re-run.

Note what the pruning did on corrected points: final sets are *larger*
than the old path's (30/33/28/24 vs 19/16/14/10) and Q1 gained inliers on
three pairs (ohrc_01: 5, ohrc_02: 11, ohrc_05: 10) — the field correction
lets the rebel quadrant survive the RANSAC refit. That is the mechanism
working end-to-end, not a threshold trick.

## Pre-registered success clauses

- (a) **New order reaches SUCCESS_SUBPIXEL: MET** (Part B: ohrc_01,
  ohrc_02, ohrc_03; Part A: none).
- (b) **Zero verdict downgrades vs old order: HOLD** (both parts, all
  pairs).
- (c) **Fuse still declines where there is no signal: HOLD** —
  ohrc_01's 19-pt true-path set declines (`no_improvement`) in both
  Part A and the old order; the fuse is not a rubber stamp.
- (d) **No folding (min Jacobian det ≥ 0.5): HOLD** (0.978–1.013).
- (e) **Deterministic seed 7: HOLD** (re-run identical).

## Honest caveats (read before citing)

1. **The gate basis is interpolation-like.** The stage's internal
   stride-80/20 held-out interleaves check points through the inlier
   cloud — the same quantity the pipeline has always gated on, and the
   same basis as the current hook. It does not measure extrapolation;
   A13's stricter spatial-band score (1.99–3.88 px) remains the harder
   benchmark.
2. **Borderline cases.** ohrc_05 Part B reads 0.5252 px (COARSE by
   0.03 px); A14's bootstraps on ohrc_01 read 0.40–0.52 px. Report
   ranges, not just point estimates.
3. **Tiny-set interpolation.** Part A ohrc_05 shows inlier RMSE 0.0000
   px on 10 points — the field nearly interpolates; the *held-out*
   (1.38 px) is the honest score and is what the gate uses. On tiny
   sets the inlier number is meaningless; the design already scores
   held-out, but the fuse's 2-point check split is noisy there.
4. **The exported warp is still the affine matrix.** As with the
   current stage, the field refines scored residuals and the verdict,
   not the exported transform (dense-remap export is future work).

## Recommendation: RE-HOOK — in the Part-B formulation

The re-hook idea is **not dead**, but the brief's formulation ("before
NCC pruning") is moot: there is no rich set at that point. The working
formulation, tested here: **run the stage on the unbucketed RANSAC
inlier set — i.e. move the hook earlier, to right after the primary
RANSAC fit and before quadrant balancing** — then run the existing
balance → bucket → NCC-prune chain on the corrected points, then the
frozen gate. That is a pipeline-design change for the maintainer; this
experiment does not make it (no `app.py` changes were made). If
adopted: keep the flag opt-in, keep the meaningful-improvement fuse
(it correctly declined the 19-pt set), keep the folding fuse, and keep
reporting the interpolation-vs-extrapolation caveat alongside any
ACCEPT.

## Reproducibility

- Script: `scripts/stage_prepruning_experiment.py` (seed 7; in-script
  assertions reproduce A15's numbers and A14's 5061/844/1.6087 before
  building; standalone RANSAC refit `cv2.setRNGSeed(7)`, documented).
- `results/table_stage_prepruning.csv`: 20 rows (old_off, old_on, new_A,
  new_B × 4 pairs).
- `app` imported with A15's stub modules (`/tmp/stubs`, outside the
  repo); the stage/pruning/gate all use the repo's real functions.
- ~3.5 min total, sequential, memory-light.
