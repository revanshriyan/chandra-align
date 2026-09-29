# Live audit RMSE follow-up

**Date:** 2026-09-29

## Root cause

The CPU and local/GPU entry points call the same `_run_alignment_core` and `_align_core`; there is no CPU-only omission of sub-pixel refinement. The pipeline uses a 3.0 px RANSAC threshold for optical pairs with `refineIters=10`, then attempts NCC/parabolic refinement. On the audit pair, only one correspondence survived NCC refinement, so the code retained the RANSAC model.

The reproducible gap came from the Gradio adapter. With the default `OHRC` selection, `_process_alignment_from_ui` silently assigned **TMC-2** to every non-IIRS secondary input. The synthetic pair has no sensor label, so the local harness correctly treated both images as OHRC, while the live adapter declared them OHRC/TMC-2 and resized one image under the false 0.25 m vs 5 m GSD assumption. The exact old UI-default route locally reproduced the live Space result.

## Fix

The adapter now infers each input sensor from a PDS4 `.img` XML label or known native example filename. For unlabelled uploads it uses the selected sensor as the fallback for the reference image and then for the secondary image; it no longer invents TMC-2. This preserves the existing seven UI inputs and direct native `gr.Examples` wiring. No RANSAC threshold, matcher, NCC refinement, acceptance-gate, export, or rejection behavior was changed.

Native examples resolve to these sensor pairs:

| Example | Reference sensor | Secondary sensor |
|---|---|---|
| NAC / OHRC | LROC_NAC | OHRC |
| NAC / TMC-2 | LROC_NAC | TMC-2 |
| NAC / IIRS | LROC_NAC | IIRS |
| Synthetic pair | selected fallback (default OHRC) | selected fallback (default OHRC) |

## Full UI-callback reproduction

All runs used file paths and `_process_alignment_from_ui`, through `process_wrapper` and the full `process_alignment` path, with UI defaults (ANMS on, CLAHE on, shadow suppression off, Wallis off). They ran locally through CPU fallback. Metrics for the gentle pair are bit-identical over two consecutive runs.

| Pair/configuration | Result | Inliers / matches | RMSE | Entropy | Quadrants | Runtime |
|---|---|---:|---:|---:|---:|---:|
| Gentle, old adapter assumption OHRC/TMC-2 | Coarse advisory | 46 / 66 | 0.5727469 px | 1.8837298 | 4: [10, 5, 16, 15] | 16.70 s |
| Gentle, corrected inference, run 1 | Accepted (Sub-Pixel Precision) | 62 / 150 | 0.3840431 px | 1.9532075 | 4: [16, 10, 21, 15] | 24.02 s |
| Gentle, corrected inference, run 2 | Accepted (Sub-Pixel Precision) | 62 / 150 | 0.3840431 px | 1.9532075 | 4: [16, 10, 21, 15] | 8.19 s |
| Strong, corrected inference | Accepted (Sub-Pixel Precision) | 61 / 141 | 0.4140796 px | 1.9342746 | 4: [13, 19, 20, 9] | 15.33 s |

The gentle RMSE improved by **0.1887038 px** versus the live-audit reproduction and meets the requested ≤0.45 px target. It also retains more than the required 25 inliers, entropy ≥1.5, and all four quadrants. The strong result improves from the reported 1.0753 px advisory to 0.4141 px.

## Rejection checks

The callback was run against unrelated synthetic-vs-LROC NAC imagery with each inferred source sensor. Every case rejected; no degenerate transform was accepted:

| Mismatch | Outcome |
|---|---|
| OHRC vs LROC NAC | Rejected: degenerate one-quadrant support; RMSE shown as N/A in telemetry |
| TMC-2 vs LROC NAC | Rejected: zero verified inliers and zero spatial support |
| IIRS vs LROC NAC | Rejected: degenerate one-quadrant support; RMSE shown as N/A in telemetry |

All three surfaced the failed validation checklist and masked the zero-RMSE sentinel in telemetry. The app's existing cross-sensor example assets are legitimate overlap examples, so they were not misrepresented as mismatch negatives.

## Verification

- Sensor routing adapter tests: **2 passed**.
- Gentle app-path run: two consecutive runs returned identical inliers, matches, RMSE, entropy, and quadrant counts.
- Strong app-path run: improved result shown above.
- Mismatch rejection checks: all rejected with specific gate reasons.
- Syntax and full test suite: pending final post-change run.
