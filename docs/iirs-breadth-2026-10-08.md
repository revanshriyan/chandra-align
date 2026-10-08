# IIRS strip breadth: LoFTR beyond the single pair — 2026-10-08

Worker A23. The Phase 9 LoFTR breakthrough (docs/phase9-loftr-addendum.md:
VNIR composite ↔ TMC-2, 2432 corr → 997 inliers → 1.35 px COARSE) was ONE
window. This experiment runs the IDENTICAL recipe on 5 NEW windows from the
real IIRS cube to test whether the breakthrough transfers.

## Recipe (unchanged from the addendum)

- IIRS: VNIR composite = mean of bands 0–46 (<1500 nm, from the PDS4 label
  wavelengths) from the full 3.29 GB `.qub`, per-band per-column-median
  destripe, then mean. No parameter touched.
- TMC-2: fore strip (`ch2_tmc_ncf_20231101T0125121344_d_img_d18`), cropped by
  geographic corner mapping through the PDS4 tie grids, resized to the IIRS
  window shape (the "common GSD" step).
- `u8` normalization (1–99 percentile), `LoFTRMatcher` (kornia `outdoor`,
  conf ≥ 0.2), `refine_lk` (no model: keep all tracked), then the Phase 9
  `gate_verdict` (verify_guarded RANSAC 3.0/2000/0.99 + frozen gates).
- Seed 7, deterministic, CPU, sequential.

## Results

| Window | IIRS lines | IIRS samples | TMC-2 overlap | Corr | LK kept | Inliers | RMSE (px) | Q1/Q2/Q3/Q4 | Entropy | Verdict |
|---|---|---|---|---|---|---|---|---|---|---|
| W0 (repro) | 0:4634 | 0:218 | 65% | 2472 | 2103 | 826 | 1.40 | 436/242/145/3 | 1.48 | COARSE_ADVISORY |
| W1 | 0:2000 | 0:180 | 93% | 1274 | 1138 | 219 | 1.69 | 76/13/107/23 | 1.62 | COARSE_ADVISORY |
| W2 | 2000:4000 | 0:150 | 88% | 1211 | 1023 | 262 | 1.57 | 126/107/21/8 | 1.48 | COARSE_ADVISORY |
| W3 | 1000:3000 | 0:160 | 94% | 1558 | 1356 | 220 | 1.67 | 100/16/92/12 | 1.55 | COARSE_ADVISORY |
| W4 | 0:2000 | 60:200 | 79% | 1024 | 958 | 456 | 1.31 | 136/115/144/61 | 1.94 | COARSE_ADVISORY |
| W5 | 2000:4000 | 40:180 | 69% | 1110 | 899 | 278 | 1.63 | 193/36/47/2 | 1.23 | COARSE_ADVISORY |

Products: IIRS `ch2_iir_nci_20231225T1904122779_d_img_d18` ↔ TMC-2 fore
`ch2_tmc_ncf_20231101T0125121344_d_img_d18` throughout.

Transforms are mutually consistent: scale 0.95–0.98, rotation −1.0° to
−1.5°, matching the addendum's ~1.2° / 0.99 geometry. The scale <1 reflects
the approximate common-GSD resize (see caveat below), absorbed by the
affine fit; gates do not constrain scale.

## Pre-registered bars

- (a) ≥3 of 5 new windows COARSE or better → **5/5 COARSE. PASS.** Breadth
  claimed: the LoFTR IIRS↔TMC-2 recipe works across the strip, not just on
  one window.
- (b) Recipe transfers unchanged → **PASS.** No parameter changed; the only
  differences are window geometry and the TMC-2 pairing (documented below).
- (c) Deterministic re-run identical → **PASS.** W4 re-run: 1024/958/456/
  1.3100 px, bit-identical to the first run.

## Is the 1.35 px single-pair result typical or lucky?

**Typical.** The five new windows span 1.31–1.69 px; the original 1.35 px
sits inside that range, and W4 beats it outright (1.31 px, 456 inliers,
entropy 1.94 — the best-balanced quadrant spread of the set). No window
failed; no cherry-picking (every window run is in the table).

## Caveats (honest)

1. **TMC-2 pairing differs from the original.** The addendum's
   `phase9_tmc_common_gsd.npy` was made by an undocumented process. Mine
   maps IIRS window corners → TMC-2 (scan, pixel) through the PDS4 tie grids
   and resizes. W0 reproduces the regime (2472 vs 2432 corr, 826 vs 997
   inliers, 1.40 vs 1.35 px) but not exact numbers — the pairing, not the
   recipe, accounts for the difference.
2. **IIRS ground sampling is ~300 m, not 80 m.** The tie grid shows 311 m
   per line along-track (≈217 m/sample cross-track at lat −71.7°); the
   `SENSOR_PIXEL_SCALES` 80 m figure is nominal. The "common GSD" resize is
   therefore approximate (measured affine scales 0.95–0.98, not 1.0). This
   does not affect the verdict — the affine absorbs it — but the pixel RMSE
   is in IIRS working pixels, not calibrated ground units.
3. **Windows are smaller than W0** (2000×140–180 vs 4634×218) because TMC-2
   fore overlap only covers IIRS lines 0–4500 (verified by tie-point
   proximity), and the strips cross diagonally. Inlier *density* is actually
   higher on the new windows (W4: 1.63/kpx² vs W0: 0.82/kpx²).
4. **Q4 is thin on W0/W2/W5** (3/8/2 inliers). All pass the frozen COARSE
   quadrant requirements (entropy ≥ 0.50, ≥2 quadrants), but spatial
   weighting is uneven — same caveat the addendum noted for its Q4=11.
5. Still COARSE, not sub-pixel (1.31–1.69 px > 0.50 px). The breadth win is
   generality, not accuracy.

## Files

- `scripts/iirs_breadth.py` — full experiment (geometry, VNIR, pairing,
  LoFTR+LK, gates)
- `results/table_iirs_breadth.csv` — per-window table
