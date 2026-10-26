# ChandraBench v0.2 — DRAFT release notes (DO NOT PUBLISH)

**Status: draft, prepared 2026-10-26 on branch `chandrabench-v0.2`.
Publication needs the maintainer's approval and a human Zenodo login
(see `ZENODO_UPLOAD_CHECKLIST.md` — no API upload from agent sessions).**

## What's new

- 4 pairs, 79 human-verified landmarks (was 2 pairs, 40).
- New: `ohrc_01` (21 points) and `ohrc_04` (18 points), human-verified
  2026-10-24 via the landmark clicker (SIFT+RANSAC candidates, NCC>=0.96
  gate, spread-enforced, independent of the pipeline fit).
- New pairs use **2048x2048 benchmark-crop coordinates** — not
  full-resolution coordinates. Do not mix with v0.1 pairs.
- `eval.py --self-test` passes on all 4 pairs.

## Read before citing

- `ohrc_01` (LT=2) and `ohrc_04` (LB=2) fall one point short of the
  >=3/quadrant protocol floor; every eval report flags `floor_pass=false`
  for them, and the shortfall is in their CSV caveats.
- `ohrc_04` is thin-covered (18 points) and UNLABELED for strip-guard
  calibration (landmark RMSE 2.2591 px fails the <2.0 px bar).
- **v0.1 pin correction:** the sha256 pins shipped in v0.1 never matched
  the frozen CSVs, so the v0.1 evaluator aborted on its own guard. The
  CSVs are byte-identical to the v0.1 tag and untouched; only the pins
  were corrected. If you hit a v0.1 sha256 mismatch, this is why.

## Verification (on branch, before publication)

- `python eval.py --self-test` → PASSED (4 pairs; identity RMSEs match
  direct computation; GT self-fit residuals: ohrc_pair 4.27 px,
  tmc2_fore_nadir 3.49 px — reproducing the v0.1 README figures —
  ohrc_01 1.40 px, ohrc_04 1.51 px).
- All 4 sha256 pins verified programmatically against their files.
- v0.1 CSVs byte-identical to the `chandrabench-v0.1` tag.
- License: CC-BY-4.0 (unchanged).
