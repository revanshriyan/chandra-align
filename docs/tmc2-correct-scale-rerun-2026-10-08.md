# TMC-2 Correct-Scale Re-run: All Six Pairs with pixel_scale_m=5.0 — 2026-10-08

Worker A27. A26 proved every tmc2 DEGENERATE in the 12-pair benchmark was a
harness misconfiguration: the scripts called
`app._align_core(ref, sec, reference_sensor_name="TMC-2")` without
`pixel_scale_m`, so it defaulted to 0.25 (the OHRC value) while
`get_sensor_pixel_scale("TMC-2")` = 5.0. The reference was downsampled 16x to
128x128 against a full-resolution secondary — a many-to-one matching collapse
(174 matches, 38 unique targets, 101 hits on one target). The deployed UI
passes `pixel_scale_m=get_sensor_pixel_scale(reference_sensor)` = 5.0, so
both images match at native 2048x2048. All six tmc2 verdicts from A15/A17/A19
were withdrawn pending this re-measurement.

## Method

All six tmc2 pairs (`data/benchmark_crops/tmc2_*`, TMC-2 fore
`ch2_tmc_ncf_20231101T0125121344_d_img_d18` -> nadir
`ch2_tmc_ncn_20231101T0125121377_d_img_d18`) through the true
`app.py::_align_core`, twice each: flag OFF and flag ON
(`CHANDRA_DEFORM_FIELD=1`), with `pixel_scale_m=5.0` passed explicitly.
`/tmp` stubs for gradio/spaces/matplotlib (never committed), seed 7,
sequential, deterministic. Collapse quantified per arm by re-running
`match_pair_hf` on the GSD-rescaled images exactly as the hook sees them
(A26's method). Control: ohrc_01 with `pixel_scale_m=0.25` passed explicitly
(not defaulted), flag off/on.

Script: `scripts/rerun_tmc2_correct_scale.py`.
Results: `results/table_tmc2_correct_scale.csv`,
`results/table_tmc2_collapse_check.csv`.

## Results

| Pair (terrain) | Flag OFF | Flag ON |
|---|---|---|
| tmc2_01 — large shadowed crater walls and rugged relief | COARSE / 12 inl / 2.31px / ent 1.48 / 3q | **SUCCESS_SUBPIXEL** / 23 inl / 0.46px / ent 1.76 / 4q / stage λ=0.1 |
| tmc2_02 — cratered surface with broad shadowed relief | COARSE / 14 inl / 1.86px / ent 1.30 / 3q | COARSE / 9 inl / 0.54px / ent 0.76 / 2q / stage λ=0.1 |
| tmc2_03 — large shadowed crater and cratered surroundings | DEGENERATE / 6 inl / 8.41px | DEGENERATE / 22 inl / 1.00px / ent 0.00 / 1q [0,0,22,0] / stage λ=0.1 |
| tmc2_04 — crater rim and rugged inter-crater relief | DEGENERATE / 7 inl / 1.72px | COARSE / 10 inl / 0.25px / ent 0.97 / 2q / stage λ=0.1 |
| tmc2_05 — densely cratered inter-crater terrain | DEGENERATE / 8 inl / 3.03px | COARSE / 11 inl / 0.59px / ent 0.99 / 2q / stage λ=0.1 |
| tmc2_06 — isolated crater with shadowed rim relief | COARSE / 9 inl / 1.80px / ent 1.44 / 3q | **SUCCESS_SUBPIXEL** / 23 inl / 0.42px / ent 1.63 / 4q / stage λ=0.1 |

Frozen gates throughout (ACCEPT: RMSE<=0.50, >=8 inliers, entropy>=0.75,
>=3 quadrants; COARSE: RMSE<=2.50, >=8 inliers, entropy>=0.50, >=2 quadrants).

## Pre-registered bar verdicts

- **(a) Collapse gone: PASS, all 12 arms.** Match images are 2048x2048 on
  both sides. Max hits per reference target: 3–4 (< 10). Unique-target
  ratio: 0.82–0.92 (> 0.5). Flag-off match counts 384–735 (vs 174 collapsed);
  flag-on (uncapped quota) 5,318–7,712. The collapse was purely the scale
  misconfiguration — no pair collapses at the correct scale.
- **(b) Zero flag-on downgrades: PASS.** Four upgrades: tmc2_01
  COARSE->SUCCESS_SUBPIXEL, tmc2_04 DEGENERATE->COARSE_ADVISORY, tmc2_05
  DEGENERATE->COARSE_ADVISORY, tmc2_06 COARSE->SUCCESS_SUBPIXEL. Two
  improved within tier: tmc2_02 (1.86->0.54px), tmc2_03 (8.41->1.00px).
- **(c) Determinism: PASS.** tmc2_01 flag-on re-run IDENTICAL.
- **(d) Stage's first real look at tmc2: applied on all 6 flag-on pairs**
  (λ=0.1 throughout — the grid minimum wins clearly, as on the ohrc pairs).
  No folding: min Jacobian determinant 0.79–0.89 (>= 0.5 bar) on every pair.
  Exports carry `field_remap` where a warp was produced. No forced wins:
  tmc2_03's verdict honestly stays DEGENERATE.

## Honest notes

- **tmc2_03** is the instructive case: the stage applies (22 inliers, gate
  1.00px) but all 22 inliers sit in ONE quadrant [0,0,22,0], entropy 0.00 —
  the gate refuses on spatial coverage, correctly. A better RMSE does not
  buy a better verdict when the geometry is one-sided.
- **tmc2_04** flag-on: gate 0.25px — deep in sub-pixel territory — yet the
  verdict is honestly COARSE (10 inliers, 2 quadrants). Same lesson as
  ohrc_04: the gate is not an RMSE contest.
- **Control: PASS.** ohrc_01 with `pixel_scale_m=0.25` passed explicitly is
  bit-identical to `results/table_quota_12pair.csv` (flag off: COARSE /
  1.6445px; flag on: SUCCESS_SUBPIXEL / 0.4312px). Passing the scale
  explicitly changes nothing; only the *value* mattered.
- The withdrawn A15/A17/A19 tmc2 verdicts are superseded by this table. The
  A17 tmc2_04 0.47px claim was already withdrawn as a harness artifact
  (separate issue); this re-run replaces the entire tmc2 column, including
  the flag-off DEGENERATEs, which were the misconfiguration's casualties.
- With the correct scale, the tmc2 side of the benchmark is no longer a
  write-off: 2 SUCCESS_SUBPIXEL + 3 COARSE flag-on, and the stage engages
  everywhere. The remaining DEGENERATE (tmc2_03) is a coverage failure, not
  a matching failure.

## Files

- `scripts/rerun_tmc2_correct_scale.py` — full experiment (12 runs + collapse
  probes + control + determinism check)
- `results/table_tmc2_correct_scale.csv` — 12 rows: verdict/inliers/RMSE/
  entropy/quadrants/stage per pair x flag
- `results/table_tmc2_collapse_check.csv` — 12 rows: many-to-one stats per
  pair x flag
