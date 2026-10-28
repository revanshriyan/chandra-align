# ChandraBench changelog

## v0.2 (draft — prepared 2026-10-26, NOT published)

- **Coordinate unification amendment (2026-10-28, draft only — this
  package is still unpublished): the ohrc_01/ohrc_04 landmarks were
  re-derived from 2048x2048 benchmark-crop coordinates into v0.1 space
  (loaded-4096), so the three OHRC pairs now score in one unified space.**
  Transform: `X_v01 = (origin_x + x_crop) x 4096/93693`,
  `Y_v01 = (origin_y + y_crop) x 4096/93693`, with committed crop
  origins ohrc_01 ref (593,1010), src (1630,2349); ohrc_04 ref
  (5311,49009), src (5617,50688); native strips 12000x93693 px loaded
  at `max_dimension=4096` (loaded space 525x4096 px). Builder script
  (workspace scratch, not in the repo):
  `~/workspace/rederive_v02_to_v01_2026-10-28.py`. The v0.1 pairs and
  their CSVs/pins are untouched; the v0.2 CSVs' `coordinate_space`
  column now states the unified space, and every row's `caveat` carries
  the transform (origins + scale) plus the precision caveat below.
  New sha256 pins in `eval.py`: ohrc_01
  `c51326bbffccbf7e2ad3dabaf134ebee340f94820e42f518fd6398f77a9a59be`,
  ohrc_04
  `b98ad22bc3b4506b56cd53e72ce1389f591df72e68b1d4560207d79931c1165e`.
  **Precision caveat:** average-resampling downsampling means the
  native-to-loaded point map is a linear convention good to ~half a
  loaded pixel (~11 native px) in the worst case; cross-space landmark
  comparisons inherit that quantization. **Self-test (3-part, all 39
  points):** (a) round-trip crop -> v0.1 -> crop max error 1.023e-11
  crop px; (b) LSQ self-fit residuals on re-derived pairs equal
  crop-space residuals x scale exactly (ohrc_01 1.402507 px -> 0.061314
  px, ohrc_04 1.514809 px -> 0.066223 px; max per-point relative
  deviation 3.4e-11); (c) native-raster-window vs crop-PNG-window NCC
  0.947–0.99995 over 3 ref-side points per pair (plus src-side bonus
  points; one src window scores Pearson 0.86 only because 40.9% of its
  crop pixels are clipped at 255 by the per-crop intensity stretch —
  rank-NCC 0.965 and best shift (0,0) confirm the geometry).
  Quadrant coverage unchanged by re-derivation (ohrc_01 LT=2, ohrc_04
  LB=2, both still `floor_pass=false`); `eval.py --self-test` prints
  SELF-TEST PASSED. The pre-2026-10-28 crop-coordinate disclosure in
  `README.md` is retained as a dated historical note.
- Original v0.2 draft entry (2026-10-26) follows; the notes below
  describe the package as first prepared, before the 2026-10-28
  re-derivation above.

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
