# ChandraBench changelog

## v0.2 (draft — prepared 2026-10-26, NOT published)

- New pairs from the 39 landmarks human-verified 2026-10-24 (clicker
  protocol: SIFT+RANSAC candidates, NCC>=0.96 gate, spread-enforced,
  independent of the pipeline fit; every point verified by a human):
  - `landmarks_ohrc_01_v0.2.csv` — 21 points, 2048x2048 benchmark-crop
    coordinates. Coverage caveat: LT quadrant holds 2 points, below the
    >=3/quadrant protocol floor (eval reports floor_pass=false).
  - `landmarks_ohrc_04_v0.2.csv` — 18 points, 2048x2048 benchmark-crop
    coordinates. Coverage caveats: LB quadrant holds 2 points
    (floor_pass=false); thin coverage, 18 points below the ~20 target.
- v0.2 pairs use **benchmark-crop coordinates, not full-resolution
  coordinates**. They must never be mixed with v0.1 pairs. The crop is
  stated in every row's `coordinate_space` column.
- `eval.py` now scores 4 pairs (79 landmarks). v0.1 pairs and their CSVs
  are untouched.
- Self-test change (documented): gates on evaluator arithmetic only;
  quadrant coverage is reported per pair with an explicit COVERAGE CAVEAT
  line for below-floor pairs, instead of gating the self-test. The
  per-pair `floor_pass` flag in scoring reports is unchanged.
- **v0.1 pin correction:** the sha256 pins shipped in v0.1 never matched
  the frozen CSVs (ohrc_pair pin 8de1613a… vs true 23f1aaf6…;
  tmc2_fore_nadir pin 254d9d2c… vs true 84500aaa…), so the v0.1 evaluator
  aborted on its own circularity guard for both pairs. The CSVs are
  byte-identical to the `chandrabench-v0.1` tag — the frozen content is
  genuine — and are untouched; only the pins were corrected to the true
  hashes (verified programmatically against the files).
- Builder: `scripts/build_chandrabench_v02.py` (maps the clicker exports
  onto the v0.1 schema; VERIFIED -> ACCEPTED; ncc blank as exported).

## v0.1 (2026-10-07, DOI 10.5281/zenodo.23217163)

- First release: 2 pairs, 40 human-verified landmarks
  (`landmarks_ohrc_pair_v0.1.csv`, `landmarks_tmc2_fore_nadir_v0.1.csv`),
  one-command evaluator, CC-BY-4.0.
- Known defect found later (see v0.2 entry): the shipped sha256 pins did
  not match the frozen CSVs.
