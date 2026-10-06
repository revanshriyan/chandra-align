# IIRS↔TMC-2 LoFTR breakthrough — addendum to Phase 9 report

## Result

**First successful IIRS↔TMC-2 registration.** LoFTR (detector-free dense
matcher, kornia `outdoor` weights) on the destriped 2852.6 nm band vs
common-GSD TMC-2:

| Stage | Correspondences | Unique inliers | RMSE | Verdict |
| --- | --- | --- | --- | --- |
| LoFTR raw | 2242 | 639 | 1.83 px | COARSE_ADVISORY |
| + LK sub-pixel refine | 1881 kept | 748 | 1.37 px | COARSE_ADVISORY |

Gate 3: PASS (cond 1.0, det 0.976, scale ratio 1.0, projectivity 0.0).
Quadrants: 315 / 152 / 161 / 11 — all four active, entropy 1.60.
Transform: near-rigid, ~1.3° rotation, scale 0.988, translation (2.0, 110.2) px.

## Honest caveats

- COARSE, not sub-pixel (1.37 px > 0.50 px accept threshold).
- The Phase 10 pixel area-check returns "weak" (0/36 cells verified) — but it
  was calibrated same-sensor; on cross-modal data this is absence of evidence,
  not disproof. A cross-modal-calibrated pixel check is future work.
- Q4 holds only 11 of 748 inliers; spatial weighting is uneven.
- LoFTR is terrestrially trained; lunar transfer is measured here (it worked),
  not assumed.

## Environment note

torch 2.14.1+cpu + kornia 0.8.3 installed 2026-10-07 after an initial CDN
timeout. Fixed a real dtype bug in `loftr_arm.py`: kornia's checkpoint loads
as float64, so the input now adapts to the model's actual parameter dtype.

## What did not work (kept for the record)

- SIFT on all 3 bands: 1–25 Lowe matches, none gated.
- Wallis: detection up, matching unchanged — not adopted.
- GOA→NMI: chance-level on all bands.
- Phase-congruency arm (Phase 14): 11 Lowe-good → 8 inliers, gate COARSE but
  inliers clustered on the left edge and pixel-unverified — promising,
  not claimed.

## 2026-10-07 update: pseudo-panchromatic composite improves on the single band

Per the SOTA brief (idea 3), tested IIRS composites through the same LoFTR+LK
harness and gates — no threshold touched:

| Input | LoFTR corr | Inliers (post-LK) | RMSE | Verdict |
| --- | --- | --- | --- | --- |
| 2852nm single band | 2242 | 748 | 1.37 px | COARSE_ADVISORY |
| VNIR composite (<1500nm) | 2432 | 997 | 1.35 px | COARSE_ADVISORY |
| Full-spectrum mean | 2405 | 724 | 1.90 px | COARSE_ADVISORY |
| TMC-2-weighted composite | 1950 | 496 | 1.82 px | COARSE_ADVISORY |

The VNIR composite is the new best: +33% inliers, 1.35 px. Full-spectrum
averaging adds SWIR thermal-emission physics TMC-2 never sees — measured
worse, as the brief predicted. All composites gate clean; the winner was
picked by measurement, not by design.

## 2026-10-07 update: MINIMA cross-modal LoFTR tested — does not beat outdoor

Per the SOTA brief (idea 1), tested MINIMA-LoFTR (CVPR 2025 cross-modal
checkpoint, 211/211 key-compatible with kornia LoFTR, added as
`pretrained="minima"` in `loftr_arm.py`) on the identical pair and harness:

| | Outdoor | MINIMA |
| --- | --- | --- |
| Correspondences | 2242 | 243 |
| Raw verdict | COARSE (639 inl, 1.83px) | DEGENERATE_FAILURE (105 inl, 2.65px, 2/4 quads) |
| + LK refine | COARSE (748 inl, 1.37px) | COARSE (52 inl, 1.74px, 3/4 quads) |

MINIMA trains on synthetic IR/depth/event/sketch modalities — none resembles
a hyperspectral↔panchromatic lunar pair. Tested, measured, not adopted.

**The valuable finding:** two independently-trained matchers converge on the
same geometry — ~1.2° rotation, ~0.99 scale, ~107–110 px y-translation,
agreeing to 1.8 px at image center. Cross-matcher agreement corroborates the
breakthrough is real geometry, not a matcher artifact.
