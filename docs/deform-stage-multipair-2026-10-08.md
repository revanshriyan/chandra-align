# Deformation-Field Stage: Multi-Pair Evaluation — 2026-10-08

## What was done

Every benchmark pair (6 OHRC + 6 TMC-2, `data/benchmark_crops/`) was run
twice through the **true** `app.py::_align_core` — once with
`CHANDRA_DEFORM_FIELD` unset (baseline), once set to `1` (stage). The
only difference between arms is the flag; methodology never changes.

**Import path (documented):** `app.py` imports `gradio`, `spaces`, and
`matplotlib` at module level. The system matplotlib (3.6.3) is
ABI-broken against numpy 2.5.3, and gradio/spaces are not installed, so
this evaluation imports `app` with minimal stub modules on PYTHONPATH
(`/tmp/stubs/{gradio,spaces,matplotlib}` — outside the repo, never
committed). `_align_core` never touches the UI or any plot call, so the
stubs are inert. Stub sources are throwaway; the method is what matters:
the real `_align_core`, flag off vs on.

**Determinism:** the pipeline seeds OpenCV RNG per fit; a flag-off
re-run of ohrc_01 after all 24 runs reproduced the first run
**identically** (status, inlier count, RMSE to 1e-9).

## Key finding: the true pipeline path ≠ the standalone replication path

`_align_core` does **not** gate on the 842 SIFT/RANSAC inliers that the
A13/A14 experiments used. After RANSAC it applies 8×8 grid bucketing
(max 8 per bucket), NCC sub-pixel refinement of inlier positions, and a
model **re-fit** on the refined positions — which prunes ohrc_01 to **19
inliers at 1.39 px**. (Consistent with the 2026-09-30 live audit: 9
inliers on the same pair.) A14's headline (1.61→0.41 px,
SUCCESS_SUBPIXEL) was measured on a standalone replication that skips
the bucketing and the refine-and-refit steps.

Consequence, measured honestly: on ohrc_01's **true** path the stage
declines (`no_improvement`) — 19 inliers → 15 fit / 4 check in the
internal split leaves no distortion signal the fuse will certify. The
stage's headline win does not transfer to the integrated pipeline
as-is. What *does* transfer is verified below: the stage never harms,
declines honestly, and applies where genuine signal exists.

## Per-pair results (frozen gates)

| Pair | Flag | Verdict | Inliers | Inlier RMSE | Gate RMSE (basis) | Stage | λ | Held-out after |
|------|------|---------|--------:|------------:|-------------------|-------|---|---------------|
| ohrc_01 | off | COARSE | 19 | 1.3920 | 1.6445 (held-out) | — | — | — |
| ohrc_01 | on | COARSE | 19 | 1.3920 | 1.6445 (held-out) | declined (`no_improvement`) | — | — |
| ohrc_02 | off | COARSE | 16 | 1.2537 | 1.3041 (held-out) | — | — | — |
| ohrc_02 | on | COARSE | 16 | 1.2537 | 1.3041 (held-out) | declined (`no_improvement`) | — | — |
| ohrc_03 | off | COARSE | 14 | 1.2920 | 1.1021 (held-out) | — | — | — |
| ohrc_03 | on | COARSE | 14 | 1.0084 | 0.8540 (field held-out) | **applied** | 100 | 0.8540 |
| ohrc_04 | off | COARSE | 16 | 2.0214 | 1.7057 (held-out) | — | — | — |
| ohrc_04 | on | COARSE | 16 | 2.0214 | 1.7057 (held-out) | declined (`no_improvement`) | — | — |
| ohrc_05 | off | COARSE | 10 | 1.5645 | 2.0549 (held-out) | — | — | — |
| ohrc_05 | on | COARSE | 10 | 0.3989 | 1.3832 (field held-out) | **applied** | 0.1 | 1.3832 |
| ohrc_06 | off | COARSE | 8 | 0.9533 | 1.1510 (held-out) | — | — | — |
| ohrc_06 | on | COARSE | 8 | 0.9533 | 1.1510 (held-out) | declined (`no_improvement`) | — | — |
| tmc2_01–04, 06 | off/on | DEGENERATE | 8 | ~0.0000 | 0.0000 | declined (`no_improvement`) | — | — |
| tmc2_05 | off/on | DEGENERATE | 6 | ~0.0000 | 0.0000 | declined (`insufficient_points_for_split`) | — | — |

Full numbers: `results/table_stage_multipair.csv` (25 rows incl. the
determinism re-check).

## Pre-registered bars

**(a) The stage NEVER downgrades a verdict: HOLD.** All 12 pairs have
identical verdicts flag-off vs flag-on. Zero downgrades, zero upgrades.
No stage bug found.

**(b) Honest declines are correct behavior: HOLD.** 10 of 12 pairs
decline — 4 OHRC via the meaningful-improvement fuse (`no_improvement`),
5 TMC-2 via the same fuse, 1 TMC-2 via `insufficient_points_for_split`
(6 inliers → split too small; fail-closed as designed). The TMC-2 pairs
are ~0-residual minimal fits the gate already rejects as DEGENERATE;
the stage correctly does not attempt to rescue degenerate fits.

**(c) Held-out scoreboard: 2 improve / 0 decline / 10 same.**
- ohrc_03: held-out 1.1021 → **0.8540** (−22%), λ=100.
- ohrc_05: held-out 2.0549 → **1.3832** (−33%), λ=0.1; inlier RMSE
  1.5645 → **0.3989** (sub-pixel in-sample, but the verdict honestly
  stays COARSE — the gate scores the 1.38 px held-out basis, and
  entropy/quadrant requirements with 10 inliers are not met).
- No pair's held-out worsened.

## Interpretation

1. **The stage is safe.** It never downgrades, never crashes, declines
   with named reasons, and its fuse correctly refuses degenerate and
   near-affine inputs. As an opt-in, default-off stage, it carries no
   regression risk to the current pipeline.
2. **The stage helps where genuine distortion signal survives the
   pipeline's own pruning** — ohrc_03 and ohrc_05 improved on held-out
   with only 14 and 10 inliers. Small inlier sets are not a
   disqualifier; absent signal is.
3. **The headline needs re-scoping.** A14's 1.61→0.41 px
   SUCCESS_SUBPIXEL was measured on the pre-pruning inlier set (842).
   Through the true `_align_core` path, ohrc_01 presents 19
   NCC-refined inliers and the stage declines. If the project wants the
   headline win inside the real pipeline, the honest options are: (i)
   run the stage *before* the NCC-refine-and-refit pruning step, on the
   larger RANSAC inlier set; or (ii) keep the stage as a
   sometimes-applied polish that fires on ~1/6 of real pairs. Both are
   legitimate; (i) is a pipeline-design decision for the maintainer.
4. **TMC-2 pairs are out of scope for this stage** — all six are
   DEGENERATE minimal fits (~0 px residual, 6–8 inliers); there is no
   geometric signal to model, and the stage correctly stands down.

## Reproducibility

- Script: `scripts/evaluate_stage_multipair.py` (seed: pipeline-internal;
  determinism re-check IDENTICAL).
- `results/table_stage_multipair.csv`: per-arm rows, both flags, stage
  telemetry columns, plus the determinism re-check row.
- ~17 s per `_align_core` run on 2048² pairs; 24 runs + 1 re-check ≈
  6 min total, sequential, memory-light.
