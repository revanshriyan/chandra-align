# Phase 12 Report — Confidence calibration

## What was built

- `chandra_align/trust/calibration.py`: logistic map (numpy IRLS, deterministic,
  no sklearn) from the four confidence features to P(SUCCESS_SUBPIXEL |
  features). Includes `reliability_diagram`, `brier_score`, `is_calibrated`,
  and `format_upper_bound`.
- `scripts/run_phase12_calibration.py`: builds the calibration set, fits,
  evaluates, writes `results/phase12_calibration.json`.
- `tests/test_phase12.py`: 9 tests, all pass.

Hard constraint kept: the calibrator scores outcomes post-hoc only. No gate,
matcher, or RANSAC path imports it (asserted by test).

## Calibration set (n = 84)

- 78 synthetic pairs run through the real pipeline (SIFT → Lowe 0.75 →
  `verify_guarded` → quadrant metrics → gate): 24 easy shifts, 16 rotated,
  20 illumination-stressed, 12 scale, 6 degenerate. Outcomes: 47
  SUCCESS_SUBPIXEL, 14 COARSE_ADVISORY, 11 DEGENERATE_FAILURE, 6 ABSTAIN.
- 6 complete-feature rows from `results/table_canonical_v1.csv`
  (2 ACCEPTED, 4 COARSE).

## Measured weights vs the guessed 30/25/20/25

Fitted logistic weights (features in [0, 1]):

| feature | guessed share | fitted weight |
| --- | --- | --- |
| support (inliers ≥ 8) | 30% | +0.12 |
| inlier ratio | 25% | +0.47 |
| spread (entropy/2) | 20% | +2.25 |
| residual quality (1 − rmse/2.5) | 25% | +2.46 |
| intercept | — | −3.93 |

The guess was wrong in emphasis: spatial spread and residual quality carry
almost all the signal; raw inlier support barely matters once the others are
known. The heuristic is not updated in place — the fitted map is reported
alongside it, and any future weight change needs this calibration evidence.

## Reliability: honest verdict

- Brier score: fitted **0.120** vs raw heuristic/100 baseline **0.220** —
  the fitted map is substantially better as a probability.
- Reliability diagram (10 bins, 0.15 tolerance, bins with n ≥ 5 checked):
  **NOT CALIBRATED**. Bin [0.6, 0.7] (n=19) predicts 0.665 but observes
  0.263 (overconfident in the middle); bin [0.7, 0.8] (n=44) predicts 0.755
  but observes 1.000 (underconfident at the top). Mass is bimodal — the
  features separate well, starving the middle bins.

Conclusion: the fitted score is a better *ranking* than the heuristic but
is **not yet a calibrated probability** and must not be quoted as one.
More mid-difficulty cases (the Phase 13 window batch is the natural source)
are needed before the numbers earn the word "probability".

## Upper-bound framing

Adopted: `format_upper_bound(rmse_px)` renders
"true error ≤ X px (includes 1 px marking noise)". Use this framing in docs
and reports wherever an RMSE is quoted.

## Artifacts

- `results/phase12_calibration.json` (fit, bins, Brier, per-sample table)
- `scripts/run_phase12_calibration.py` (re-runnable, seeded)
- 9 `tests/test_phase12.py` tests pass
