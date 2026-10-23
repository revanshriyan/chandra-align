# Arm-vs-fallback arbitration — trial results (2026-10-23)

Written after the run. Protocol and win bar were fixed beforehand in
`docs/arm-arbitration-prereg-2026-10-23.md`. Raw table:
`results/table_arbitration_trial_2026-10-23.csv` (30 runs). Harness:
`scripts/run_arbitration_trial.py`. Hardware: this CPU-only VM; backend
commit: main 5ae3a89 plus the arbitration branch.

## Verdict

**Clean negative — the win bar is not met on any of its three clauses.**
`CHANDRA_ARBITRATE` stays off. The implementation remains in the tree
behind its flag as documented negative evidence; this table is the
standing reason not to enable it.

- Safety: FAILED. Arbitrated was worse than the fallback parent in 2 of
  12 cells (ohrc_03 PC and ohrc_04 PC: arm DEGENERATE_FAILURE, fallback
  COARSE_ADVISORY, arbitrated DEGENERATE_FAILURE — the selector picked
  the arm both times).
- Benefit: FAILED. 1 of 10 ohrc cells improved over arm-only
  (bar: >= 3). The one success: ohrc_04 XoFTR, where the selector picked
  the fallback and recovered COARSE_ADVISORY from the arm's
  DEGENERATE_FAILURE.
- Rescue preserved: FAILED as posed — and the failure is informative.
  In this deployed path (`_align_core`), XoFTR arm-only on tmc2_04 is
  itself DEGENERATE_FAILURE (held-out 2.84 px), not the COARSE_ADVISORY
  rescue the ablation harness recorded. There was no rescue in this
  path for arbitration to preserve. The ablation result stands for its
  own harness; it does not reproduce through `_align_core` unchanged.

## Why the selector failed

The selector picked the arm in 11 of 12 cells. Raw arm point sets are
dense (thousands of correspondences) and score systematically better on
a raw held-out RANSAC-affine metric than the sparse raw SIFT set
(fallback scores 24–223 px vs arm scores 12–114 px). The metric mostly
measures point-set density and self-consistency, not downstream quality:
the pipeline's own filtering, refinement, and deform stage decide the
verdict, and they absorb or reject raw sets in ways the raw score does
not predict (e.g. ohrc_01: PC arm-only and fallback-only produce the
identical verdict, held-out RMSE, and inlier count despite raw scores
of 16.6 vs 165.7 px).

A working selector would have to score candidates through (a replica
of) the downstream itself — at which point it is no longer a lightweight
pre-filter but a second full pipeline run per arm, with its own
calibration burden. That is a different, larger proposal; nothing in
this trial justifies building it.

## What did not change

- Default behavior: untouched. With the flag unset the arm chain and
  the default path are bit-identical (app.py change is an additive
  block that only runs when the flag is set and an arm fired).
- The documented limitation from the ablation stands, now with a
  measured reason: naive arm-vs-fallback arbitration by raw-set
  held-out score is worse than no arbitration on the cells that matter.
- The selector's scope limit from the pre-registration also stands: it
  cannot detect coherently-wrong-but-consistent transforms (Phase-11
  class); that remains the independent verification machinery's job.
