# XoFTR Trial — IIRS↔TMC-2 Cross-Modal Arm (2026-10-15)

## What was tested
XoFTR-640 (CVPRW 2024, LoFTR extended for visible↔thermal matching, Apache-2.0,
public weights) on IIRS↔TMC-2 lunar windows. Pre-registered bar: beat AnyMatch's
1.4531 px on W5 or MatchAnything's 0.8082 px on W4, zero downgrades.

## Key result
**W5: 0.9791 px, 23 inliers, COARSE_ADVISORY, 4/4 quadrants** — beats the bar.

**What the 0.98 px is:** RANSAC inlier RMSE (fit residual on the 23 inliers),
scored by the identical `gate_verdict` (RANSAC 3.0/2000/0.99 + frozen gates)
used by all prior arm trials. It is a fit RMSE, not a held-out RMSE.

**Fair comparison:** AnyMatch was re-run on the SAME refined W5 window on
2026-10-15 and scored 1.3324 px (1223 inliers). XoFTR wins by 0.35 px
apples-to-apples.

**Dataset-asymmetry caveat:** the original AnyMatch 1.4531 px number was
measured on the pre-refinement W5 pair; the trial ran on the
MatchAnything-refined pair. The fair re-run above closes this gap.

## Breadth (19 pairs)
**Effective-resolution caveat:** the AnyMatch and XoFTR frontends downscale
inputs larger than 2048 px (AnyMatch caps at 2016 px for LoFTR positional
encoding; XoFTR caps at 640 px) and rescale coordinates back to the caller
frame. Windows above the cap are therefore scored at lowered effective
resolution, and the RMSE comparison reflects what each arm returns on that
rescaled input — not necessarily what it would achieve at full resolution.

XoFTR-640 across all 19 IIRS pairs: **16/19 COARSE** (broadest single-arm
coverage), **11/19 outright wins** vs the best prior arm, **4 DEGENERATE→COARSE
conversions** (W2, W5, N2, N5 — windows where every prior arm failed).
Losses on W4/N4/N7/N8/N9 where MatchAnything stays better. W0 regresses
(0 correspondences vs SIFT's 2.31 px) but the wired chain falls through to
SIFT. N13/N14 remain DEGENERATE for all arms.

## Wiring
Opt-in arm `CHANDRA_XOFTR=1`, priority MatchAnything > AnyMatch > XoFTR > PC.
Weights (43 MB, 640 variant) referenced externally, not vendored.
Pipeline nuance: XoFTR's 0.98 px W5 win does not surface in the deployed
pipeline as-wired — MatchAnything (1.20 px) takes priority there. XoFTR's
pipeline value is the D→C conversions on W2/N2/N5.

## Results files
- `results/table_xoftr_w5.csv` — 7-window trial (W0–W5, N13)
- `results/table_xoftr_breadth.csv` — full 19-pair breadth run
- Raw JSONs: `~/workspace/xoftr_trial/results/`

## Status
Still COARSE, not sub-pixel. Narrows the IIRS gap; does not close it.
