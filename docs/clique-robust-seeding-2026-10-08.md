# L3 Clique-Robust Hook Seeding: Clean Negative — 2026-10-08

Worker A25. **The L3 lever does not work through the real pipeline hook.
The implementation was reverted; no `app.py` / `chandra_align/` changes
remain in the tree.** This document records why, so the negative result
is evidence, not a gap.

## What was tried

A22 (docs/blocker-diagnosis-2026-10-08.md) found the six TMC-2 pairs'
hook RANSAC latching onto a degenerate clique (72–177 "inliers", one
quadrant, ~zero spread, ~zero residuals), and a standalone probe showed
that excluding the clique let RANSAC find a real consensus (163–479
inliers) on which the stage applied at 0.43–0.72 px held-out.

A25 implemented clique-robust re-seeding behind `CHANDRA_DEFORM_FIELD=1`:
detect a degenerate clique (single-quadrant AND ~zero spread AND ~zero
residuals, all required), exclude it, re-seed RANSAC, iterate up to 5
rounds. Flag-off path untouched; any failure falls back to today's
behavior.

## Why it fails: the probe's point set is not the pipeline's point set

The probe ran on **standalone** `match_pair_hf` output (5,921 matches from
raw crops). The real hook runs on **GSD-rescaled** matching output
(456–795 matches). These are different point sets with different
structure.

On the pipeline's point set, the degeneracy is not one clique but
**nested cliques, all the way down**:

| Pair | Hook points | Hook clique | Nested cliques found (10 rounds) | Points remaining |
|---|---|---|---|---|
| tmc2_01 | 795 | 177 | 10 | 329 |
| tmc2_02 | 467 | 114 | 10 | 239 |
| tmc2_03 | 621 | 96 | 10 | 292 |
| tmc2_04 | 694 | 76 | 10 | 406 |
| tmc2_05 | 456 | 72 | 10 | 246 |
| tmc2_06 | 592 | 132 | 10 | 262 |

Every consensus RANSAC finds — at every seed, at every exclusion round —
is degenerate (scale ≈ 0, median residual ~1e-10 px). After excluding 10
cliques, RANSAC still finds an 11th. There is **no real consensus hiding
behind the clique** in the pipeline's point set; the GSD-rescaled TMC-2
matching output is dominated by many-to-one false matches.

(The clique structure itself was characterized precisely: 177 source
points spread across the frame all matched to literally identical
reference coordinates (1474.6, 902.1), with the fitted affine at
determinant 1e-26 — a many-to-one match collapse, likely repeated
texture.)

## Bar verdicts

- (a) Flag-off bit-identical: **PASS** (12/12 gate RMSEs byte-identical;
  the change never touched the flag-off path).
- (b) Flag-on zero downgrades: **PASS** (no verdict changes anywhere).
- (c) Stage applies through the real hook on tmc2: **FAIL** — the stage
  declines (`no_improvement`) on all six pairs because no non-degenerate
  consensus exists in the hook's point set.
- (d) Determinism: **PASS**.

Per the pre-registered rule (any bar fails → revert), the implementation
was reverted with `git checkout`. The unit tests
(`tests/test_deform_field_clique.py`, 10 passing) were removed with it;
the detector logic is preserved in this document's description.

## The honest conclusion

A22's L3 probe was correct on its own point set but does not transfer to
the pipeline: the probe's "real consensus" came from standalone matching,
while the deployed hook sees GSD-rescaled matching output that contains
no recoverable consensus. **The tmc2 DEGENERATE verdicts are not a seed-
selection problem; they are a matching problem.** The GSD-rescaled
TMC-2 fore→nadir matching produces point sets dominated by degenerate
many-to-one cliques.

The correct next lever is not in the hook — it is in understanding why
GSD-rescaled TMC-2 matching collapses (matching scope, out of scope for
this worker), or in running the tmc2 pairs through standalone matching
(a pipeline-configuration question, not a stage question).

## Files

- `docs/clique-robust-seeding-2026-10-08.md` — this document
- `scripts/dbg_all_tmc2.py` equivalent evidence: nested-clique counts
  table above (from `/tmp/dbg_all_tmc2.py`, ephemeral)

No `app.py` / `chandra_align/` / `tests/` changes remain. Working tree
clean.
