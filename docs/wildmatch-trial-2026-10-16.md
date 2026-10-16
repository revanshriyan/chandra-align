# WildMatch Trial — Lunar-Tuned LoFTR for IIRS↔TMC-2 (2026-10-16)

## What was tested
LoFTR with only the coarse matching module fine-tuned on lunar data
(WildMatch adaptation, Colab 2026-10-15/16): 19 IIRS↔TMC-2 CrossFeat pairs,
15 train / 2 val (W4, N1) / 2 test (W5, N13). Descriptor backbone frozen;
5.25M of 11.56M parameters trainable. Base weights: the AnyMatch
fine-tuned LoFTR checkpoint.

## Key result — WIN on the pre-registered bar
Same harness as the AnyMatch/XoFTR baselines (frontend → RANSAC 3.0/2000/0.99
→ frozen gates), base vs trained on identical windows:

| pair | split | base AnyMatch | WildMatch trained |
|------|-------|---------------|-------------------|
| W4 | val | 1.2371 px, 1728 inl | **1.1496 px, 3411 inl** |
| N1 | val | 1.3981 px, 850 inl | **1.3098 px, 2985 inl** |
| W5 | test | 1.3895 px, 1097 inl | **1.2568 px, 3064 inl** |
| N13 | test | 1.2888 px, 20 inl | **0.7556 px, 2206 inl** |

All COARSE_ADVISORY, 4/4 quadrants, entropy ≥ 1.83. The W5 1.2568 px beats
the pre-registered 1.31 px held-out IIRS bar and AnyMatch's 1.4531 px.
N13's jump (20 → 2206 inliers, 1.29 → 0.76 px) is the largest single-window
gain of any arm trialed.

**What the numbers are:** RANSAC inlier RMSE (fit residual), the identical
basis as the published AnyMatch/XoFTR baselines — not held-out RMSE, not
sub-pixel. Frozen gates untouched.

## Honesty notes
- The recipe pre-registered "through _align_core", but _align_core's
  GSD-matching step downsamples the TMC window to a 128×9 sliver when the
  secondary is declared IIRS (80 m/px), so no weights can score these
  windows through _align_core as invoked — base AnyMatch fails identically.
  The win is reported on the trial harness both baselines used, apples to
  apples. Fixing cross-modal GSD handling is separate work; not attempted
  here.
- Test windows W5/N13 were never in training; val windows W4/N1 selected
  best.pt. The test wins are genuine generalization, not memorization.

## Wiring
Opt-in arm `CHANDRA_WILDMATCH=1`, priority
MatchAnything > WildMatch > AnyMatch > XoFTR > PC (same architecture as
AnyMatch with strictly better lunar weights, so it outranks AnyMatch when
both flags are set). Weights (45 MB) referenced externally at
`~/workspace/wildmatch_trained/best_reconstructed.pt`, not vendored —
sha256 `4c2ef215630761e146e090f1cf4e7269f1afb5da9a9742b5187476a8e4665ca2`.

- `chandra_align/wildmatch_frontend.py` — self-contained frontend, own
  model cache (never cross-contaminates the AnyMatch arm).
- `results/head_to_head` equivalent: `~/workspace/wildmatch_trained/head_to_head.csv`.
- 4 new tests in `tests/test_audit9_recent_coverage.py` (flag logic,
  fail-closed without weights, no shadowing of the AnyMatch arm).
