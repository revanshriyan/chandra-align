# Coarse-to-fine re-run — results (2026-10-23)

Written after the run. Protocol and win bar were fixed beforehand in
`docs/coarse2fine-rerun-prereg-2026-10-23.md`. Raw table:
`results/table_coarse2fine_rerun_2026-10-23.csv`. Harness:
`scripts/run_coarse2fine_rerun.py`. Hardware: this CPU-only VM
(torch 2.14.1+cpu); both arms SIFT-backed via the pipeline's fallback.
Backend commit: main 5ae3a89.

## Verdict

**Clean negative.** Movement across the 12 pairs: 1 improved, 2 regressed,
9 unchanged. The pre-registered bar (at least 10/12 improved AND zero
regressed) is not met. Coarse-to-fine stays opt-in; no default changes.

## Detail

- Improved: tmc2_05 (DEGENERATE_FAILURE -> COARSE_ADVISORY, held-out
  RMSE 3.03 -> 0.93 px).
- Regressed: ohrc_01 (COARSE_ADVISORY -> DEGENERATE_FAILURE, held-out
  1.64 -> 2.71 px) and ohrc_05 (COARSE_ADVISORY -> DEGENERATE_FAILURE,
  2.05 -> 2.96 px). Both regressions cross the 2.50 px coarse held-out
  limit upward.
- Secondary measure (does not change the classification): held-out RMSE
  was lower under coarse-to-fine in 7 of 12 pairs; median delta -0.11 px,
  mean delta -0.86 px. The mode often improves the residual while
  occasionally breaking a verdict — the opposite of what a default needs.
- tmc2_03 is a caution against reading the secondary measure alone:
  held-out RMSE fell 8.41 -> 0.10 px on only 6 inliers (a handful of check
  points), verdict DEGENERATE_FAILURE in both arms.
- Context vs issue #4 (2026-10-05): the single-scale baseline itself is
  stronger now (9 of 12 pairs at COARSE_ADVISORY or better in the current
  pipeline, vs the trial-era mix), which shrinks coarse-to-fine's headroom.
  The coarse_to_fine code is unchanged since the original trial; what
  changed is everything around it, and the regressions persist through the
  current pipeline on the same OHRC pairs.
- Runtime cost: coarse-to-fine added roughly 4–9 s per pair on CPU
  (about +30–45%).

## Decision recorded

`matching_mode` default remains `single_scale`. Coarse-to-fine remains
available as an explicit opt-in for pairs where an operator wants it,
with this table as the standing evidence of its risk profile.
