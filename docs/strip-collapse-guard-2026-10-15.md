# Strip-Collapse Guard — TMC-2 Remediation (2026-10-15)

## What happened
The 2026-10-15 human-landmark rerun (audit item #1) confirmed the TMC-2
fore/nadir pair reports COARSE_ADVISORY at 1.77 px inlier RMSE while the
20 human landmarks show **28.03 px** frame-wide error (~125 m). Root cause:
the 13 SIFT inliers collapsed to a thin horizontal strip (ref y in
[944,1096] of a 2048 frame) while the true correspondence field spanned
the frame. Quadrant entropy was 1.95 with 4/4 quadrants — the gate did not
catch the collapse because the strip straddled the y=1024 quadrant
boundary. Evidence: `results/table_issue01_landmark_rerun_2026-10-15.csv`.

## The metric
`chandra_align/metrics/spatial_support.py` measures frame-normalized PCA
support ratios of the inlier cloud: `minor_ratio` = minor-axis extent /
frame minor dim, `major_ratio` = major-axis extent / frame major dim.
Collapsed iff either < 0.25 (a principled geometric constant: <1/4 frame
span along any principal direction leaves ≥3/4 of the frame as pure
extrapolation). Frame normalization is required — raw PCA axis ratios are
confounded by non-square frames (healthy OHRC 525×4096 sets score 0.10
raw). Separation on the two ground-truth pairs:

| metric | TMC-2 (collapsed) | OHRC (healthy) |
|---|---|---|
| PCA minor / frame minor | **0.075** | **0.717** |
| bbox linear coverage (min) | 0.075 | 0.701 |
| hull area / frame area | 0.036 | 0.397 |

The metric is always computed and reported in telemetry as
`strip_support_metrics` (diagnostic fields, never gate-affecting).

## Why the verdict downgrade is opt-in, not default
A 12-pair regression (`results/table_strip_guard_regression_2026-10-15.csv`,
`scripts/regress_strip_guard_12pair.py`) showed a default-ON downgrade
would flip **ohrc_01** (verified SUCCESS_SUBPIXEL, 0.328 px, 28 inliers;
minor_ratio 0.243) to DEGENERATE_FAILURE, and ohrc_04's main fit scores
0.19 (verified 0.29 px via L1 rescue). The 0.25 threshold splits a tight
cluster of healthy pairs (0.2428 vs 0.260/0.262). Without independent
ground truth for those pairs, a verdict-changing default cannot be
justified — it would invalidate verified claims on an uncalibrated
constant. Per the project's additive/opt-in rules, the fail-closed
downgrade (code STRIP_COLLAPSE) is therefore **opt-in via
CHANDRA_STRIP_GUARD=1**, clearly marked experimental. The telemetry red
flag is always on: analysts see `strip_support_metrics.collapsed=true`
for TMC-2-class cases.

## Deform-field rescue on TMC-2: 28.03 → 14.68 px (clean non-win)
`scripts/experiment_tmc2_deform_rescue.py` ran the deform field (λ=0.01)
against the 20 human landmarks (landmarks never touch the fit):
**14.6757 px** landmark RMSE vs the 28.0317 px affine baseline — a genuine
~48% reduction, but still DEGENERATE under the frozen gates. The L1 rescue
refit did not trigger (`coverage_sufficient`). Same Phase-11 signature as
the affine fit: field internal held-out 0.57 px vs 14.68 px on landmarks
(~26× understatement). No gates or thresholds were changed.
Evidence: `results/table_tmc2_deform_rescue_2026-10-15.csv`.

## Status
- Metric + telemetry: shipped, always-on, additive.
- Verdict downgrade: DEFAULT-ON since 2026-10-24 (opt-out via `CHANDRA_STRIP_GUARD=0`); see the 2026-10-24 addendum below. The "opt-in, not default" reasoning in the section above is the superseded 2026-10-15 decision, kept as history.
- TMC-2 fore/nadir: root cause diagnosed; no remediation reaches the
  frozen gates. Wider landmark ground truth is the prerequisite for
  calibrating a default-ON guard.

## Addendum 2026-10-24 — calibrated, default-ON

The prerequisite above is now met. Revan human-verified 39 landmarks
(ohrc_01: 21, ohrc_04: 18; `~/workspace/your_files/gt_ohrc_01_human_clicked.csv`,
`gt_ohrc_04_human_clicked.csv`) and the calibration rig ran on them
(`results/table_strip_guard_calibration_2026-10-24.csv`):
ohrc_01 landmark RMSE 1.8869 px at minor_ratio 0.2428; ohrc_04 landmark
RMSE 2.2591 px at minor_ratio 0.1926; both pipeline SUCCESS_SUBPIXEL on
inlier RMSE (0.328 / 0.291 px), distinct from these landmark RMSEs.
The old 0.25 threshold would have downgraded both pairs.
The calibrated threshold is T* = (0.075 + 0.2428) / 2 = 0.1589 → 0.159,
the midpoint of the rig's LABELED separation: the known-collapsed
anchor (0.075) vs labeled-good ohrc_01 (0.2428, under the rig's
< 2.0 px landmark-RMSE labeling bar). ohrc_04 is UNLABELED under that
bar (2.2591 px) and thin-covered (18 < the ~20 protocol target): it sits
above the threshold as a consistency observation, not a calibration
point. (An earlier draft of this addendum misstated the derivation as
the midpoint of 0.075 and 0.1926, which is 0.1338, not 0.159; the
2026-10-25 Auditor review caught it, and this text is the correction.)
Clean separation holds on every measured pair: all six OHRC regression
pairs sit above 0.159 on both ratios, and all TMC-2 regression rows are
already DEGENERATE_FAILURE, so the guard changes no TMC-2 verdict
(tmc2_01's support measurement is valid with minor ratio 0.0, but its
verdict is already DEGENERATE and a downgrade cannot change it).
Change: `STRIP_SUPPORT_RATIO_MIN` 0.25 → 0.159 and the downgrade is
DEFAULT-ON, opt-out via `CHANDRA_STRIP_GUARD=0`. Fragility on the
record: the labeled calibration rests on the single pair ohrc_01,
clearing the 2.0 px bar by 0.11 px; a third full-coverage pair would
strengthen it further. Gate thresholds themselves are untouched; the
guard only ever downgrades to DEGENERATE_FAILURE.

## Approval record (2026-10-26)

Decision (a) STRIP_SUPPORT_RATIO_MIN 0.25 → 0.159 and (b) the default
flip to default-ON were delegated by Revan to the assistant on
2026-10-26 ~02:24 IST, after the corrected derivation
(T* = (0.075 + 0.2428) / 2 = 0.1589 → 0.159; ohrc_04 unlabeled) and the
explicit (a)/(b) question were presented to him. The assistant's
decision is to APPROVE both. This is recorded faithfully as delegation,
not as a direct tap approval, for the Integrity Auditor to rule on.
Merge awaits Auditor sign-off on this record plus the GitHub issue
number (filing in progress at time of writing).
