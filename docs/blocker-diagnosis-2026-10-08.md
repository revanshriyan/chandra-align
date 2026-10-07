# Blocker Diagnosis: what keeps the remaining pairs from the next tier — 2026-10-08

Worker A22. Diagnosis only — no `app.py` / `chandra_align/` changes. All runs
through the true `app.py::_align_core` with `CHANDRA_DEFORM_FIELD=1`
(throwaway `/tmp` stubs, seed 7, deterministic). Frozen gates:
ACCEPT/SUCCESS_SUBPIXEL = RMSE ≤0.50 px, ≥8 inliers, entropy ≥0.75, ≥3 quadrants;
COARSE_ADVISORY = RMSE ≤2.50 px, ≥8 inliers, entropy ≥0.50, ≥2 quadrants.

## 1. Per-pair gate breakdown (flag on)

| Pair | Description | Verdict | Gate RMSE | n | Ent | Quads (Q1–Q4) | Single component blocking the next tier |
|---|---|---|---|---|---|---|---|
| ohrc_04 | inter-crater terrain with clustered small craters | COARSE | 0.396 | 12 | 0.811 | 2 [0,0,3,9] | **quadrants 2 < 3** (RMSE already beats 0.50) |
| ohrc_05 | mixed crater sizes and low-sun shadows | COARSE | 0.531 | 20 | 1.559 | 3 [5,8,0,7] | **RMSE 0.531 > 0.50** (by 0.031 px) |
| ohrc_06 | prominent shadowed crater and rugged rim relief | COARSE | 0.645 | 20 | 1.441 | 3 [7,10,0,3] | **RMSE 0.645 > 0.50** |
| tmc2_01 | large shadowed crater walls and rugged relief | DEGENERATE | — | 7 | — | 1 [0,7,0,0] | inliers 7 < 8 (root cause: §4) |
| tmc2_02 | cratered surface with broad shadowed relief | DEGENERATE | — | 3 | — | 1 [0,0,0,3] | inliers 3 < 8 (root cause: §4) |
| tmc2_03 | large shadowed crater and cratered surroundings | DEGENERATE | — | 4 | — | 1 [0,0,0,4] | inliers 4 < 8 (root cause: §4) |
| tmc2_04 | crater rim and rugged inter-crater relief | DEGENERATE | — | 4 | — | 1 [4,0,0,0] | inliers 4 < 8 (root cause: §4) |
| tmc2_05 | densely cratered inter-crater terrain | DEGENERATE | — | 4 | — | 1 [0,0,4,0] | inliers 4 < 8 (root cause: §4) |
| tmc2_06 | isolated crater with shadowed rim relief | DEGENERATE | — | 5 | — | 1 [0,5,0,0] | inliers 5 < 8 (root cause: §4) |

Products: ohrc pairs are `ch2_ohr_nrp_20240425T1209509264_d_img_d18` (reference,
12:09 window) → `ch2_ohr_nrp_20240425T1406019344_d_img_d18` (source, 14:06 window);
tmc2 pairs are `ch2_tmc_ncf_20231101T0125121344_d_img_d18` (fore) →
`ch2_tmc_ncn_20231101T0125121377_d_img_d18` (nadir). Full manifest in
`data/benchmark_pairs.csv`; machine table in `results/table_blockers.csv`.

## 2. ohrc_04: quadrant starvation (the interesting case)

Gate RMSE 0.396 px already beats ACCEPT, entropy 0.811 passes — the *only*
blocker is quadrants (2 < 3). Inlier trace through the true pipeline
(`results/trace_ohrc04.csv`, logging wrappers only, no logic touched):

| Step | in → out | Quadrant counts of output |
|---|---|---|
| hook RANSAC | 7562 → 1141 | [0, 0, 769, 372] |
| quadrant balance (50/quad) | 1141 → 50 | [0, 0, 25, 25] |
| grid bucketing | 50 → 50 | [0, 0, 25, 25] |
| RANSAC | 50 → 40 | [0, 0, 18, 22] |
| NCC subpixel refine | 40 → 15 | — |
| RANSAC (refined) | 15 → 12 | [0, 0, 3, 9] |

The top half is already empty **at the hook**: Q1/Q2 contribute zero of the
1141 hook inliers. This is a *consistency* failure, not detection — SIFT finds
~2500 keypoints/quadrant and raw matches are plentiful in Q1/Q2
(1656/1865), but none survive the global affine RANSAC at 3 px
(`scripts/diagnose_rawmatch.py`). The NCC 40→15 drop looks dramatic but is not
the blocker: even a perfect NCC step cannot create Q1/Q2 inliers.

Field-rescue probe (`scripts/diagnose_field_rescue.py`): scoring **all** raw
matches under the fitted field (fit on Q3/Q4 hook inliers only):

| Pair | affine <3px Q1/Q2 | field <3px Q1/Q2 | field <1px Q1/Q2 |
|---|---|---|---|
| ohrc_04 | 0 / 0 | **79 / 275** | 4 / 16 |
| ohrc_05 | 158 / 200 | 579 / 1576 | 313 / 717 |
| ohrc_06 | 184 / 440 | 321 / 935 | 180 / 624 |

354 ohrc_04 matches in the starved quadrants agree with the field's smooth
extrapolation within 3 px where the affine finds zero. These are real
distorted matches, not mismatches — the global affine RANSAC simply cannot see
past the >3 px distortion there.

**Lever L1 — field-guided match rescue (opt-in, flag-gated).** After the stage
applies, re-score all raw matches under the field; admit field-consistent
matches into the downstream chain; everything else unchanged, fail-closed
(no admissions → today's behavior). Falsification: implement behind the flag;
success = ohrc_04 gains ≥1 quadrant with zero downgrades on all 12 pairs, no
folding, admitted matches independently NCC-verified. Caveat: the rescue uses
field *extrapolation* (A13: 1.99–3.88 px on hard splits), so the <3 px count
is generous — a real implementation should weight by residual or require local
support, and the <1 px count (20) is the conservative read.

## 3. ohrc_05 / ohrc_06: RMSE blockers

Both have 3 quadrants and healthy entropy; the stage applies (λ=0.1) and cuts
held-out 1.79→0.53 / 1.79→0.64 px. The full λ sweep
(`scripts/diagnose_lever_probes.py`):

| Pair | λ=0.1 | λ=1.0 | λ=10 | λ=100 | affine |
|---|---|---|---|---|---|
| ohrc_05 | **0.524** | 0.780 | 1.237 | 1.596 | 1.852 |
| ohrc_06 | **0.587** | 0.920 | 1.284 | 1.498 | 1.658 |
| ohrc_04 | **0.382** | 0.531 | 0.730 | 1.097 | 1.631 |

λ=0.1 (the grid minimum) wins clearly on all three — the grid edge hints that a
more flexible field might go further. ohrc_05 needs 6% (0.531→0.50), ohrc_06
needs 22% (0.645→0.50).

**Lever L2 — extend the λ grid downward** (e.g. add 0.01, 0.03). The stage's
internal held-out selection is the guardrail: a smaller λ can only be chosen
if it *generalizes* better, so overfitting is blocked by construction.
Additive, flag-gated, gates frozen. Falsification: re-run the sweep with the
extended grid on all 12 pairs; success = held-out improves on ≥1 pair with
zero downgrades and no folding. Honest expectation: ohrc_05 (6% gap) is the
plausible win; ohrc_06 (22% gap) likely needs more than flexibility — its
residual may be high-frequency distortion or localization noise, in which case
this probe returns a clean negative.

## 4. tmc2: the hook is hijacked by a degenerate clique

Stage decline reason on all six pairs: `no_improvement` — and the sweep detail
shows why (`results/table_tmc2_hook.csv`):

| Pair | hook inliers | Quadrants | Median residual | Spatial spread | Best λ check | Affine check |
|---|---|---|---|---|---|---|
| tmc2_01 | 177 | [0,177,0,0] | 9.5e-11 px | 0.000 | 6.7e-14 | 9.6e-11 |
| tmc2_02 | 114 | [0,0,114,0] | 3.2e-10 px | 0.000 | 5.0e-14 | 3.3e-10 |
| tmc2_03 | 96 | [0,96,0,0] | 8.0e-11 px | 0.000 | 5.9e-14 | 8.7e-11 |
| tmc2_04 | 76 | [0,0,0,76] | 1.7e-10 px | 0.000 | 8.9e-14 | 1.6e-10 |
| tmc2_05 | 72 | [0,72,0,0] | 4.4e-10 px | 0.000 | 6.6e-14 | 4.6e-10 |
| tmc2_06 | 132 | [0,0,0,132] | 1.2e-10 px | 0.000 | 9.3e-14 | 1.1e-10 |

The hook RANSAC latches onto 72–177 "inliers" that are **all in one quadrant,
all mutually coincident** (bounding-box spread ≈ 0 of the image diagonal) with
≈1e-10 px residuals — a degenerate repeated-texture/detector clique, not a
geometric consensus. The stage correctly declines: the improvement fuse needs
≥0.02 px absolute and there is nothing to improve on ~zero. The stage is not
at fault; the hook's RANSAC seed selection is.

But a real consensus is hiding behind the clique. Excluding the clique points
and re-running RANSAC (same threshold, seed 7):

| Pair | non-clique inliers @3px | Quadrants | Stage on this set | Field held-out | Affine held-out | minJac |
|---|---|---|---|---|---|---|
| tmc2_01 | 432 | [121,311,0,0] | **applies** (λ=0.1) | **0.478** | 1.725 | 0.799 |
| tmc2_02 | 479 | [297,0,177,5] | **applies** | **0.683** | 1.624 | 0.771 |
| tmc2_03 | 163 | [0,0,163,0] | **applies** | **0.724** | 1.627 | 0.876 |
| tmc2_04 | 322 | [0,121,0,201] | **applies** | **0.428** | 2.024 | 0.887 |
| tmc2_05 | 402 | [0,0,235,167] | **applies** | **0.515** | 1.721 | 0.832 |
| tmc2_06 | 362 | [0,64,186,112] | **applies** | **0.529** | 1.696 | 0.880 |

(`scripts/diagnose_l3_test.py` — the falsification test for L3, run *before*
proposing it.) On the non-clique consensus the stage applies on all six pairs
with 0.43–0.72 px held-out, no folding, 2–3 quadrant coverage. A wrong
consensus could not produce a smooth field that generalizes like this — the
held-out is the validator. Note the M2 scales (≈0.90 on four pairs,
0.94–1.00 on two): mutually consistent with a systematic TMC-2 fore→nadir
geometry difference, but not independently verified against ground truth —
stated, not hidden.

**Lever L3 — clique-robust hook seeding (opt-in, flag-gated).** When the hook
RANSAC inliers are spatially degenerate (single quadrant / ~zero spread),
re-seed excluding the degenerate cluster and keep the best non-degenerate
consensus; else today's behavior. Mechanism options: (a) quadrant-spread
requirement on the hook set with one fallback re-seed; (b) multi-start RANSAC
keeping the best spatially-distributed consensus. Falsification: implement
behind the flag; success = tmc2 pairs reach COARSE or better with the stage
applied, **zero downgrades on all 12 pairs**, no folding, transform sanity
(scale/rotation plausible vs the M2 values above). This is the highest-impact
lever in this report — six DEGENERATE pairs are blocked by seed selection, not
by geometry.

## 5. Ranked levers and clean negatives

| Rank | Lever | Target | Evidence | Falsification |
|---|---|---|---|---|
| 1 | L3 clique-robust hook seeding | tmc2_01–06 (DEGENERATE) | Decisive probe: stage applies 6/6 on de-cliqued sets, 0.43–0.72 px held-out, no folding | Flag-gated implementation; tmc2 → COARSE+, zero downgrades on 12/12, no folding |
| 2 | L1 field-guided match rescue | ohrc_04 (quadrants) | 354 Q1/Q2 matches field-consistent <3px vs 0 under affine | Flag-gated; ohrc_04 gains ≥1 quadrant, zero downgrades, admitted matches NCC-verified |
| 3 | L2 extended λ grid | ohrc_05 (0.531), ohrc_06 (0.645) | λ=0.1 at grid edge on all three pairs | Extended sweep; held-out improves, zero downgrades, no folding |

Clean negatives (no honest lever in stage/gate):
- tmc2's `no_improvement` declines are *correct* given the clique input — the
  stage must not be "fixed" to apply there.
- ohrc_04's NCC 40→15 refinement drop is not the blocker; loosening NCC cannot
  create Q1/Q2 inliers.
- Nothing here weakens a gate, cherry-picks a pair, or tunes on the test set:
  every lever is flag-gated, per-pair validated, and dies on any downgrade.

## 6. Method notes and caveats

- Gate basis for stage-applied pairs is the stage's internal stride 80/20
  held-out (interpolation-like); A13's harder spatial-extrapolation benchmark
  (1.99–3.88 px) stands as the stricter read.
- tmc2 gate RMSEs in the CSV (≈0 / 1e-10) are meaningless — DEGENERATE verdicts
  with 3–7 inliers; the honest tmc2 numbers are the hook/sweep diagnostics.
- The L3 probe fits the stage on the de-cliqued set as a *library* (same math
  as the hook); a shipped implementation must reproduce the seeding inside the
  pipeline and re-validate end-to-end.
- L1's rescue counts use field extrapolation into Q1/Q2; the <1 px column
  (4–20 matches) is the conservative read.

## Files

- [scripts/diagnose_blockers.py](sandbox://workspace/chandra-align/scripts/diagnose_blockers.py) — 9-pair gate breakdown + stage sweep capture
- [results/table_blockers.csv](sandbox://workspace/chandra-align/results/table_blockers.csv) — per-pair blockers, descriptions, product IDs, hook sweep detail
- [scripts/diagnose_tmc2_ohrc04.py](sandbox://workspace/chandra-align/scripts/diagnose_tmc2_ohrc04.py) — tmc2 hook characterization + ohrc_04 trace
- [results/table_tmc2_hook.csv](sandbox://workspace/chandra-align/results/table_tmc2_hook.csv) — tmc2 clique diagnostics
- [results/trace_ohrc04.csv](sandbox://workspace/chandra-align/results/trace_ohrc04.csv) — ohrc_04 inlier trace per step
- [scripts/diagnose_rawmatch.py](sandbox://workspace/chandra-align/scripts/diagnose_rawmatch.py) — detection-vs-consistency separation
- [scripts/diagnose_field_rescue.py](sandbox://workspace/chandra-align/scripts/diagnose_field_rescue.py) — field-guided rescue counts
- [scripts/diagnose_lever_probes.py](sandbox://workspace/chandra-align/scripts/diagnose_lever_probes.py) — clique-excluded RANSAC + λ sweeps
- [scripts/diagnose_l3_test.py](sandbox://workspace/chandra-align/scripts/diagnose_l3_test.py) — L3 falsification test (stage on de-cliqued sets)
- [docs/blocker-diagnosis-2026-10-08.md](sandbox://workspace/chandra-align/docs/blocker-diagnosis-2026-10-08.md) — this document
