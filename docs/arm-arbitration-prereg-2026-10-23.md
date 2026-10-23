# Arm-vs-fallback arbitration — design and pre-registration (2026-10-23)

Written before implementation and before any trial run.

## The documented limitation

From `docs/ablation-2026-10-16.md`: when an opt-in matcher arm fires
(>= 3 points), `_align_core` uses its matches unconditionally — the first
firing arm wins by priority (MatchAnything > WildMatch > AnyMatch > XoFTR
> PC). On SIFT's home turf (ohrc_01–05, where SIFT+deform reaches
SUCCESS_SUBPIXEL) a firing PC or XoFTR arm drops the verdict to
COARSE_ADVISORY. The arms are specialists: they rescue pairs where SIFT
fails (XoFTR on tmc2_04; PC at extreme sun angles; WildMatch on IIRS N13).
Nothing compares arm output against the fallback before choosing.
"No arbitration yet — documented."

## Design (additive, opt-in, fail-closed)

- New env flag `CHANDRA_ARBITRATE=1`. With the flag unset, behavior is
  bit-identical to today (the arm chain and the default path are
  untouched).
- The flag matters only when an opt-in arm fires. In that case the
  default matching path (`match_pair_hf` / the `matching_mode` path) is
  also run, producing the fallback candidate point set. (Cost: one extra
  fallback match on arm-firing runs only.)
- A selector scores BOTH candidate point sets with the same evidence
  protocol and picks one; the winner flows through the unchanged
  downstream (refinement, deform stage, frozen gate). The gate is never
  bypassed: arbitration chooses input points, not verdicts.
- Selector (new module `chandra_align/matching/arbitration.py`): for
  each candidate set with >= 8 points, repeat over 3 fixed seeds
  (7, 77, 777): seeded 80/20 split of correspondences, fit
  `cv2.estimateAffinePartial2D` (RANSAC, threshold 3.0 px) on the fit
  split, compute RMSE on the check split. Score = median check RMSE
  across seeds; a set that cannot be fit on any seed scores +inf.
  The lower score wins; ties (within 0.05 px) go to the set with more
  correspondences. A candidate with < 8 points cannot win.
- Telemetry: the result records `arbitration: {candidates, scores,
  winner}` so every arbitrated run is auditable after the fact.
- Fail-closed: any exception in arbitration returns the arm's points
  (today's behavior) and records the failure in telemetry.

## Trial (pre-registered)

- Arms under test: PC (`CHANDRA_PC=1`) and XoFTR (`CHANDRA_XOFTR=1`) —
  the two arms that run on this CPU-only VM at full crop (the LoFTR
  family OOMs at 2048x2048 here; their arbitration is the same code path
  but is not measured by this trial — recorded as a scope limit).
- Pairs: ohrc_01–05 (the documented harm zone) and tmc2_04 (the
  documented XoFTR rescue).
- Conditions per pair per arm: (a) arm-only = today's behavior (arm flag
  on, arbitration off); (b) fallback-only = no arm flag; (c) arbitrated =
  arm flag on + `CHANDRA_ARBITRATE=1`.
- Recorded per run: verdict tier, held-out RMSE, inlier count, engine,
  and (for c) the arbitration telemetry.

## Win bar (fixed in advance)

Arbitration earns a "works as designed" verdict only if ALL hold:

1. Safety: on every pair x arm cell, the arbitrated verdict tier is
   >= the arm-only tier AND >= the fallback-only tier (never worse
   than either parent).
2. Benefit: the arbitrated tier is strictly better than the arm-only
   tier in at least 3 of the 10 ohrc cells (5 pairs x 2 arms) — it
   recovers what the arm displaced.
3. Rescue preserved: tmc2_04 under XoFTR stays COARSE_ADVISORY or
   better under arbitration.

Any failure: clean negative — the flag stays off, the design and the
failing cells are documented, and no default behavior changes. Even a
pass changes no default by itself; it earns a proposal to the human.

## Scope limit found during implementation (added before the trial)

The selector measures correspondence quality (noise, mismatches). It
cannot detect a coherently transformed but wrong point set: a wrong
similarity is still a similarity, so held-out affine scoring fits it
perfectly. That is the Phase-11 confident-but-wrong class, and it remains
the job of the independent verification machinery, not of this selector.
The unit tests were corrected to use the realistic failure mode
(mismatched correspondences) after the first version of the wrong-set
generator — a pure extra rotation — scored identically to the good set
(0.394 px both), demonstrating the limit empirically.

## Unit tests (written with the implementation)

- Selector picks the geometrically consistent set over a rotated
  (confidently wrong) set with the same point count.
- Selector is deterministic across calls (fixed seeds).
- Selector refuses sets with < 8 points (cannot win).
- With `CHANDRA_ARBITRATE` unset, the arm chain is unchanged
  (existing tests cover the default path).
