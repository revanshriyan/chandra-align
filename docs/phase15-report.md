# Phase 15 — Joint 2D pose-graph optimization: report

## What was built

- `chandra_align/optimize/posegraph.py` — joint optimizer. Absolute 2x3
  affines T_i (image 0 pinned to identity) solved simultaneously from gated
  pairwise edges via `scipy.optimize.least_squares` (trust-region
  reflective + Huber loss). Residual per edge: the 6 entries of
  (M_ij @ T_i - T_j), weighted. Returns optimized affines, per-edge px
  RMSE, and loop misclosure before/after on the triplet cycle.
- `scripts/run_phase15_posegraph.py` — driver (real-triplet search, then
  synthetic demo through the real frozen gate).
- `tests/test_phase15.py` — 8 tests, all passing.
- `results/phase15_posegraph.json` — machine-readable results.

## Triplet used: synthetic only

**No real-data triplet exists.** The repo holds gated *pairwise*
registrations (Phase 9 IIRS<->TMC-2; Phase 13 OHRC-stereo and TMC-2
fore/nadir pairs) but no set of three images with gated affines on all
three legs (A->B, B->C, C->A). The driver records
`"real_triplet": {"status": "ABSTAIN", ...}` with this reason instead of
fabricating one. This phase is therefore **accepted synthetic-only**.

## Measured results (synthetic demo)

Ground-truth absolute affines known; each leg independently fitted from
noisy synthetic correspondences (81 points, sigma 0.35/1.10/0.45 px) and
passed through the real frozen `validate_registration_gate` first:

| Leg | Gate verdict | Fit RMSE |
|-----|--------------|----------|
| 0->1 | SUCCESS_SUBPIXEL | 0.39 px |
| 1->2 | COARSE_ADVISORY | 1.36 px |
| 2->0 | COARSE_ADVISORY | 0.55 px |

- Loop misclosure before optimization: **0.3616 px**
- Loop misclosure after optimization: **0.0000 px**
- Per-edge RMSE after: small on all legs; optimizer converged (nfev
  reported in the JSON).

**Acceptance (loop misclosure collapses): met — synthetic-only.**

## Honest notes

1. **TRF+Huber, not LM+Huber.** The plan says Levenberg-Marquardt +
   Huber, but scipy's LM implementation only supports a linear loss.
   TRF+Huber is the only scipy path that is both iterative least-squares
   and robust; the docstring says so plainly.
2. **A single 3-cycle cannot isolate a fault.** Testing showed that with
   one bad edge in a lone triangle, the 30 px error must live on *some*
   edge and Huber merely caps its influence — it cannot decide which leg
   lied. The Huber test therefore uses a redundant graph (two independent
   measurements of one leg): the bad edge loses the vote and is flagged
   by a large per-edge residual. Redundancy, not robustness magic, is what
   makes pose graphs work.
3. **Huber scale must sit above inlier noise.** With px-unit residuals, a
   scale below the inlier RMSE linearizes even good edges and lets a bad
   edge win tug-of-war. Documented in the docstring; pinned by tests.
4. **Gates run before the optimizer, always.** `check_gated` refuses
   DEGENERATE_FAILURE, ABSTAIN codes, and missing transforms with
   ValueError. The optimizer never sees ungated pairs.

## What would promote this beyond synthetic

A real triplet needs three images with gated pairwise affines on all
three legs — e.g. {IIRS, TMC-2 fore, TMC-2 nadir} once an IIRS<->nadir leg
exists, or {OHRC 12:09, OHRC 14:06, TMC-2} once OHRC<->TMC-2 legs exist.
Until then the optimizer is validated machinery waiting on data.
