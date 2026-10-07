# Quota-12pair validation: flag-gated keypoint quota uncapped under CHANDRA_DEFORM_FIELD=1 — 2026-10-08

## Question
The parent implemented A18's recommendation in
`chandra_align/features/distribution.py::select_detector_keypoints`: when
`CHANDRA_DEFORM_FIELD=1`, the per-cell quota is lifted to 10**9 (uncapped);
flag off honors quota 64 exactly. A18 validated 5 pairs at uncapped (3 ohrc
SUCCESS_SUBPIXEL, zero downgrades); the other 7 pairs were untested.
**Decision-critical bar: zero verdict downgrades flag-on vs flag-off across
all 12 pairs.** Any downgrade → recommend reverting the quota change.

## Method
All 12 pairs (`data/benchmark_crops/`: ohrc_01–06, tmc2_01–06) run twice
through the **true** `app.py::_align_core` — flag OFF then flag ON; the only
difference is the flag. `app.py` imported via throwaway stubs
(`/tmp/stubs/{gradio,spaces,matplotlib}` on PYTHONPATH, outside the repo,
never committed); `_align_core` never touches the UI or plot calls.
Sequential, memory-light, seed 7 (pipeline-deterministic).
Script: `scripts/validate_quota_12pair.py`. Raw data:
`results/table_quota_12pair.csv` (25 rows incl. determinism re-check).

## Results (frozen gates)

| Pair | Flag OFF (verdict / inl / gate px) | Flag ON (verdict / inl / gate px) | Stage | Δgate |
|---|---|---|---|---|
| ohrc_01 | COARSE / 19 / 1.6445 | **SUCCESS_SUBPIXEL** / 17 / 0.4312 | applied λ=0.1 | −1.21 |
| ohrc_02 | COARSE / 16 / 1.3041 | **SUCCESS_SUBPIXEL** / 23 / 0.4687 | applied λ=0.1 | −0.84 |
| ohrc_03 | COARSE / 14 / 1.1021 | **SUCCESS_SUBPIXEL** / 20 / 0.3427 | applied λ=0.1 | −0.76 |
| ohrc_04 | COARSE / 16 / 1.7057 | COARSE / 12 / 0.3957 | applied λ=0.1 | −1.31 |
| ohrc_05 | COARSE / 10 / 2.0549 | COARSE / 20 / 0.5314 | applied λ=0.1 | −1.52 |
| ohrc_06 | COARSE / 8 / 1.1510 | COARSE / 20 / 0.6449 | applied λ=0.1 | −0.51 |
| tmc2_01 | DEGENERATE / 8 / 0.0 | DEGENERATE / 7 / 0.0 | declined | 0.00 |
| tmc2_02 | DEGENERATE / 8 / 0.0 | DEGENERATE / 3 / 0.0 | declined | 0.00 |
| tmc2_03 | DEGENERATE / 8 / 0.0 | DEGENERATE / 4 / 0.0 | declined | 0.00 |
| tmc2_04 | DEGENERATE / 8 / 0.0 | DEGENERATE / 4 / 0.0 | declined | 0.00 |
| tmc2_05 | DEGENERATE / 6 / 0.0 | DEGENERATE / 4 / 0.0 | declined | 0.00 |
| tmc2_06 | DEGENERATE / 8 / 0.0 | DEGENERATE / 5 / 0.0 | declined | 0.00 |

- **Downgrade check: PASS.** Zero verdict downgrades across all 12 pairs
  (3 upgrades, 9 unchanged). No pair gets worse under the quota change.
- **SUCCESS_SUBPIXEL flag-on: 3** (ohrc_01/02/03, gate 0.43/0.47/0.34 px,
  17–23 inliers — comfortably above the 8-inlier gate minimum, unlike the
  borderline tmc2_04 case).
- Gate RMSE improves on 6/6 ohrc pairs (deltas −0.51 to −1.52 px); tmc2
  unchanged (stage correctly declines on degenerate fits).
- ohrc_04 stays COARSE despite gate 0.3957 px < 0.50 — the ACCEPT tier also
  requires ≥3 quadrants / entropy ≥0.75; the verdict is honest, not a bug.
- ohrc_05 lands 0.03 px above the ACCEPT line (0.5314), same as A16.
- Determinism re-check (ohrc_01 flag-off re-run after all 24 runs):
  **IDENTICAL**.
- Flag-off bit-identical to A15's `table_stage_multipair.csv`: **YES, 12/12
  MATCH** (status, inlier count, gate RMSE to 1e-6). The quota change is
  invisible with the flag off.

## A17 discrepancy note (for the record)
A17's report claimed flag-off COARSE on tmc2_01/02/06 and a flag-on
tmc2_04 SUCCESS_SUBPIXEL (re-hook, capped quota). Both are contradicted by
two independent measurements: A15's CSV and this run both show flag-off
**DEGENERATE_FAILURE** on all six tmc2 pairs (bit-identical match), and with
the uncapped quota the stage declines on all tmc2 pairs (no tmc2_04
upgrade). Most likely explanation: A17's tmc2 measurement had a harness
issue (e.g. flag leakage between runs); its ohrc numbers were consistent
with A15. The uncapped quota changes the hook-RANSAC input set, so A17's
capped-quota tmc2_04 upgrade is not expected to reproduce here — and it
doesn't need to: the decision bar is zero downgrades, which holds.

## Recommendation: KEEP the quota change
- Zero downgrades (hard bar met); 3 robust sub-pixel upgrades on ohrc
  (17–23 inliers, not borderline); gate RMSE improves on every ohrc pair.
- Flag-off provably bit-identical (12/12 vs A15) — the default pipeline is
  untouched.
- The stage+uncapped-quota combination is strictly better than either alone:
  uncapped quota without the stage would just add keypoints; the stage is
  what converts them into a better model, and its fuses decline honestly
  where there is no signal (all tmc2 pairs).
- Caveats carried forward: gate basis remains the interpolation-like stride
  held-out (A13's 1.99–3.88 px spatial-extrapolation score is the harder
  benchmark); exported warp is still affine-only (dense-remap future work).
