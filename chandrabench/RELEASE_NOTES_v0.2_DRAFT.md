# ChandraBench v0.2 — DRAFT release notes (DO NOT PUBLISH)

**Status: draft, prepared 2026-10-26, amended 2026-10-28, on branch
`chandrabench-v0.2`. Publication: release from the branch (maintainer
decision 2026-10-28). It needs the maintainer's approval and a human
Zenodo login (see `ZENODO_UPLOAD_CHECKLIST.md` — no API upload from
agent sessions).**

## What's new

- 4 pairs, 79 human-verified landmarks (was 2 pairs, 40).
- New: `ohrc_01` (21 points) and `ohrc_04` (18 points), human-verified
  2026-10-24 via the landmark clicker (SIFT+RANSAC candidates, NCC>=0.96
  gate, spread-enforced, independent of the pipeline fit).
- Coordinate unification (2026-10-28 amendment): the three OHRC pairs
  now score in one unified v0.1 space (loaded-4096). The 39 new
  landmarks were re-derived from their 2048x2048 benchmark-crop
  coordinates via `X_v01 = (origin_x + x_crop) x 4096/93693`
  (origins in each CSV's caveat and in README/CHANGELOG). The
  2026-10-26 draft's crop-coordinate statement is superseded;
  `tmc2_fore_nadir` keeps its audited 2048x2048 windows.
- `eval.py --self-test` passes on all 4 pairs.

## Read before citing

- `ohrc_01` (LT=2) and `ohrc_04` (LB=2) fall one point short of the
  >=3/quadrant protocol floor; every eval report flags `floor_pass=false`
  for them, and the shortfall is in their CSV caveats. Re-derivation
  does not change quadrant coverage.
- `ohrc_04` is thin-covered (18 points) and UNLABELED for strip-guard
  calibration (landmark RMSE 2.2591 px fails the <2.0 px bar).
- Precision caveat: the native-to-loaded point map is a linear
  convention good to ~half a loaded pixel (~11 native px) in the worst
  case; cross-space landmark comparisons inherit that quantization.
- The branch README carries the 2026-10-26 strip-guard note (under the
  deployed default pipeline, the TMC-2 SIFT/RANSAC path reports
  DEGENERATE_FAILURE / STRIP_COLLAPSE instead of COARSE); the issue #1
  COARSE row is preserved as measured.
- **v0.1 pin correction:** the sha256 pins shipped in v0.1 never matched
  the frozen CSVs, so the v0.1 evaluator aborted on its own guard. The
  CSVs are byte-identical to the v0.1 tag and untouched; only the pins
  were corrected. If you hit a v0.1 sha256 mismatch, this is why.

## Verification (on branch, before publication)

- `python eval.py --self-test` → PASSED (4 pairs; identity RMSEs match
  direct computation; GT self-fit residuals in v0.1 space: ohrc_pair
  4.27 px, tmc2_fore_nadir 3.49 px — reproducing the v0.1 README
  figures — ohrc_01 0.0613 px, ohrc_04 0.0662 px, equal to the
  crop-space 1.40/1.51 px times the scale 4096/93693 exactly).
- All 4 sha256 pins verified programmatically against their files
  (new pins: ohrc_01
  `c51326bbffccbf7e2ad3dabaf134ebee340f94820e42f518fd6398f77a9a59be`,
  ohrc_04
  `b98ad22bc3b4506b56cd53e72ce1389f591df72e68b1d4560207d79931c1165e`).
- v0.1 CSVs byte-identical to the `chandrabench-v0.1` tag.
- License: CC-BY-4.0 (unchanged).
