# Coarse-to-fine re-run — pre-registration (2026-10-23)

Written before the run. No results existed when this file was created.

## Question

Issue #4 (2026-10-05) measured the opt-in `coarse_to_fine` matching mode on
12 pairs x 2 matchers: 5 pairs improved, 3 regressed (verdict movement).
The recorded bar for default-on was: at least 10/12 improved with zero
regressions. It was not met, so the mode stayed opt-in.

Since then the pipeline around the matcher changed (deform stage, grid
bucketing default-on, strip telemetry), while
`chandra_align/matching/coarse_to_fine.py` itself is byte-identical to the
trial (zero commits since ab70003). Does coarse-to-fine, exercised through
the current deployed pipeline, meet the bar now?

## Protocol

- Harness: `scripts/run_coarse2fine_rerun.py`, run on this CPU-only VM
  (torch 2.14.1+cpu, numpy 2.5.3). LightGlue/ALIKED cannot run here; the
  pipeline's primary falls back to SIFT + Brute-Force, so both arms are
  SIFT-backed. This is recorded as a hardware limitation, not hidden: the
  original LightGlue rows (RTX 5070) are not reproduced by this run.
- Pairs (12): `data/benchmark_crops/{ohrc_01..06,tmc2_01..06}_{reference,source}.png`,
  2048x2048 each. OHRC pairs: sensor OHRC, pixel scale 0.25 m/px.
  TMC-2 pairs: sensor TMC-2, pixel scale 5.0 m/px (affects metre telemetry
  only; the gate and all px metrics are scale-free).
- Arms: `app._align_core(..., matching_mode="single_scale")` vs
  `app._align_core(..., matching_mode="coarse_to_fine")`, all other
  arguments default, no opt-in environment flags set.
- Per pair and arm, recorded: gate verdict (`status_code`), held-out RMSE
  (`rmse_heldout_px`, the pipeline's own seeded holdout split), in-sample
  inlier RMSE (`rmse_in_sample_px`), inlier count, check-point count,
  engine name, runtime.
- Determinism: ohrc_01 single_scale was run twice before the trial and
  produced identical verdict, held-out RMSE, and inlier count. One run per
  pair per arm is therefore sufficient.
- Movement classification (same as issue #4): verdict tiers ordered
  DEGENERATE_FAILURE < COARSE_ADVISORY < SUCCESS_SUBPIXEL. A pair is
  "improved" if the coarse-to-fine verdict tier is higher, "regressed" if
  lower, else "unchanged". Held-out RMSE delta is reported as a secondary
  measure and does not change the classification.

## Win bar (fixed in advance)

Coarse-to-fine earns a default-on proposal only if at least 10 of 12 pairs
are "improved" AND zero pairs are "regressed". Any other outcome: the mode
stays opt-in and the result is reported as measured, win or clean negative.
A default-on proposal, if earned, still requires the human's approval and
does not by itself change any default.

## Output

`results/table_coarse2fine_rerun_2026-10-23.csv` (one row per pair) plus a
short results doc appended after the run, marked as written after the fact.
