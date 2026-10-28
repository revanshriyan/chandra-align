# Verdict rename mapping — PROPOSED, not yet approved (2026-10-28)

**Status: draft for Integrity Auditor sign-off + human approval.
Nothing here has landed on main. The rename lands, if approved, as
one atomic commit carrying every site below.**

## Why

`SUCCESS_SUBPIXEL` reads as a claim about true accuracy. The gate
measures inlier self-consistency only (fit residuals against the
fit's own inliers). The Oct-15 landmark cases proved the gap: a
`SUCCESS_SUBPIXEL` verdict at 0.33 px gate RMSE alongside 1.89 px
landmark RMSE, and a `COARSE_ADVISORY` at 1.77 px alongside 28.03 px
landmark RMSE. The top verdict's name must not claim what the gate
does not measure. Thresholds, gate logic, and the other two verdicts
do NOT change — this is a rename, not a threshold change.

## Mapping (proposed)

| Old | New | Meaning (unchanged) |
|-----|-----|---------------------|
| `SUCCESS_SUBPIXEL` | `SELF_CONSISTENT_SUBPIXEL` | Inlier RMSE ≤0.50 px, entropy ≥0.75, ≥3/4 quadrants, min 8 inliers |
| `COARSE_ADVISORY` | `COARSE_ADVISORY` (unchanged) | Inlier RMSE ≤2.50 px, entropy ≥0.50, ≥2 quadrants |
| `DEGENERATE_FAILURE` | `DEGENERATE_FAILURE` (unchanged) | Everything else, including non-finite and strip-collapse downgrades |

Display label: `ACCEPTED (Sub-Pixel Precision)` →
`ACCEPTED (Self-Consistent, Sub-Pixel)`. Only the top verdict is
renamed; the asymmetry is deliberate and recorded here.

## `evidence_tier` (new, additive field)

Every verdict payload gains `evidence_tier`:

- `none` — no independent evidence attached (default for
  self-consistency verdicts with no landmark cross-check).
- `self_consistency` — the verdict rests on inlier/holdout metrics
  only. **Model-predicted verdicts stay here.** A prediction is not
  independent evidence and must never claim the tier below.
- `independent_landmarks` — the reported accuracy was cross-checked
  against human-verified landmarks for that pair, with the landmark
  RMSE stated alongside (never replacing the self-consistency
  figure; both are shown).

`evidence_tier` never changes a verdict by itself; it labels what
the verdict rests on.

## Sites that must change atomically (miss one and behavior, not just labels, changes)

- Gate core: `chandra_align/metrics/quadrant.py` (return value AND
  the `is_accepted` tuple — miss the tuple and accepted fits stop
  counting as accepted); `chandra_align/metrics/reporting.py`
  (label map). Precision note: `chandra_align/metrics/__init__.py`
  (gate wiring, strip downgrade) and
  `chandra_align/metrics/spatial_support.py` contain no renamed-
  verdict strings — passthrough only (their `DEGENERATE_FAILURE`
  references are unchanged, one in a docstring) — and needed no
  change; the earlier draft over-listed them here.
- App: `app.py` — rescue promotion, tier check, display emoji,
  payload booleans (`safety_gate_passed` / `coarse_advisory`), and
  every `status_code` passthrough.
- Eval/optimizer: `chandra_align/eval/behavioral.py` (expectation
  checks), `chandra_align/optimize/posegraph.py` (`GATED_VERDICTS`),
  `chandra_align/trust/calibration.py` (docstring),
  `chandra_align/visualization.py` (display list).
- Schema: `src/exporters/dossier.py` has no `status_code` handling
  — it does not pass the verdict through (and `src/schemas/models.py`
  carries no `status_code` field); the downloadable metrics/
  transform JSON field values change via the `app.py` transform
  payload (`safety_gate_passed` / `coarse_advisory`) and metrics
  payload (`status_code`), so consumers need this mapping note.
  `docs/dossier-schema.md` does not name verdict values and needs
  no change.
- Diagrams: `docs/images/pipeline-gates.svg`,
  `sunflip-two-rulers.svg`, `verdict-coverage.svg` — regenerate.
- Tests: every test file referencing verdicts — updated in the same
  commit, full suite green.
- Living docs only: README, model card, results one-pager.
  Historical docs keep old names under this mapping — do not
  rewrite them.

## Explicitly NOT rewritten

- All `results/*.csv` / `*.json` files (historical record; new runs
  emit new names). The table generator and the README number
  checker must handle both names via this mapping.
- Historical docs (audit trail, trial reports, failure gallery).
- The frozen gate numbers themselves (RMSE ≤0.50 / ≤2.50 px,
  entropy ≥0.75 / ≥0.50, min inliers 8, quadrant floors).

## Approval required before landing

Integrity Auditor sign-off (wording + merge) AND explicit human
approval. A threshold-issue is not triggered (no numbers change),
but an issue is filed for traceability given the blast radius.
