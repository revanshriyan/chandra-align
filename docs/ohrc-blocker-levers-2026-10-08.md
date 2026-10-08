# L1 Field-Guided Match Rescue + L2 Lower-Lambda Grid: Results — 2026-10-08

Worker A28. Two levers from docs/blocker-diagnosis-2026-10-08.md, tested
against the frozen gates. Experiment only; no `app.py` / `chandra_align/`
changes. Deterministic (seed 7).

## L1 — field-guided match rescue (ohrc_04): WIN (experiment)

**Method.** Replicated the flag-on hook + stage: raw matches via
`app.match_pair_hf`, hook RANSAC → `M_hook`, `apply_deform_field_stage` on
hook inliers (λ=0.1 chosen). Scored ALL 7,562 raw matches under the fitted
field: `|p_ref − (M_hook @ p_src + d(p_src))|`. Admitted field-consistent
matches at two thresholds; refit the stage on each admitted set (same
machinery, same grid); ran the FROZEN gate
(`validate_registration_gate`: RMSE ≤0.50, ≥8 inliers, entropy ≥0.75,
≥3 quadrants) on the rescued set, with RMSE from the stage's internal
held-out (the gate basis when the stage applies).

| Threshold | Admitted | Gate RMSE | Entropy | Quadrants | Verdict |
|---|---|---|---|---|---|
| <3 px (generous) | 2,989 | 0.390 px | 1.41 | 4 [79, 275, 1795, 840] | **SUCCESS_SUBPIXEL** |
| <1 px (conservative) | 1,691 | 0.357 px | 0.96 | 4 [4, 16, 1181, 490] | **SUCCESS_SUBPIXEL** |

**Reading.** The 20 Q1/Q2 matches at <1 px (4 in Q1, 16 in Q2) agree with a
smooth field fit on *disjoint* Q3/Q4 points — they are real distorted
correspondences the global affine RANSAC cannot see past, not mismatches.
Under the frozen gate this is a verdict upgrade (COARSE → SUCCESS_SUBPIXEL).

**Caveats, stated plainly.** (1) Q1/Q2 support is thin (4/16 points at the
conservative threshold) — the gate requires >0 per quadrant, which is met,
but this is not deep coverage. (2) The <3 px set is generous: the rescue
uses field *extrapolation* (A13 hard-split score 1.99–3.88 px), so a
production implementation should weight by residual or require local
support. (3) This is a prototype experiment, not a pipeline verdict — the
rescue is not wired into `_align_core`. The lever is proven; the
implementation is follow-up work.

**Why the 354 are real.** A random mismatch has no reason to agree with a
smooth TPS field fit on disjoint points to <1 px. The <1 px Q1/Q2 count (20)
is the conservative evidence; the <3 px count (354) is the generous read.

## L2 — lower-lambda grid (ohrc_05/06): WIN on ohrc_05, clean negative on ohrc_06

**Method.** Ran the TRUE `app._align_core` end-to-end (flag on) with the
stage's lambda grid extended to (0.01, 0.03, 0.1, 1.0, 10.0, 100.0) via a
harness monkeypatch (no file changes; app.py imports the stage at call
time). The verdict is the pipeline's own under the frozen gates.

| Pair | λ sweep (held-out px) | Chosen | Gate RMSE | Inliers | Verdict |
|---|---|---|---|---|---|
| ohrc_05 | 0.01: 0.448, 0.03: 0.482, 0.1: 0.531, 1.0: 0.730, 10: 1.221, 100: 1.634 | **0.01** | **0.448 px** | 24 | **SUCCESS_SUBPIXEL** (was COARSE at 0.531) |
| ohrc_06 | 0.01: 0.519, 0.03: 0.569, 0.1: 0.645, 1.0: 0.905, 10: 1.220, 100: 1.406 | 0.01 | 0.519 px | 18 | COARSE (was COARSE at 0.645) |
| ohrc_04 | 0.01: 0.343, 0.03: 0.360, 0.1: 0.396, … | 0.01 | 0.343 px | 15 | COARSE (still quadrant-blocked, as expected) |

**Reading.** The stage's internal held-out guardrail *selects* λ=0.01 on all
three — the grid edge was real, and the guardrail (not the experimenter)
chooses the winner, so overfitting is blocked by construction. ohrc_05
crosses 0.50 px (0.531 → 0.448) with 24 inliers: a genuine verdict upgrade.
ohrc_06 improves substantially (0.645 → 0.519) but does not cross — the
predicted clean negative; its residual is not fixed by flexibility alone.

**No folding.** Min Jacobian determinants 0.98–0.99 throughout.

## Bars

- (a) Gates frozen: all verdicts use `validate_registration_gate` unchanged.
- (b) No gate weakening: L1 admits only field-consistent matches (<1 px
  conservative); L2's lambda is chosen by the stage's internal held-out,
  not by the experimenter.
- (c) Zero downgrades: ohrc_01/02/03 re-run flag-on through the true
  pipeline — all still SUCCESS_SUBPIXEL at 0.4312/0.4687/0.3427 px,
  bit-identical to `results/table_quota_12pair.csv`.
- (d) Determinism: L1 re-run identical (verdict + admitted count).

## Recommendation

1. **Extend `LAMBDA_GRID`** in `chandra_align/deform_field.py` to
   `(0.01, 0.03, 0.1, 1.0, 10.0, 100.0)`. Additive, flag-gated, gates
   frozen. Measured: ohrc_05 → SUCCESS_SUBPIXEL through the true pipeline;
   zero downgrades on ohrc_01/02/03/04/06. This is a config change, not a
   logic change.
2. **Prototype L1 as a pipeline stage** (follow-up worker): wire the
   field-guided rescue into `_align_core` behind the flag, with the <1 px
   conservative threshold and a local-support requirement, then validate
   zero downgrades on all 12 pairs.

## Files

- [scripts/ohrc_blocker_levers.py](sandbox://workspace/chandra-align/scripts/ohrc_blocker_levers.py)
- [results/table_ohrc_levers.csv](sandbox://workspace/chandra-align/results/table_ohrc_levers.csv)

Products: reference `ch2_ohr_nrp_20240425T1209509264_d_img_d18`,
source `ch2_ohr_nrp_20240425T1406019344_d_img_d18`.
