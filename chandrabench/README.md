# ChandraBench v0.2 (draft — not published)

Four Chandrayaan-2 pairs, 79 human-verified landmarks, one fixed metric.
v0.2 adds the 39 landmarks human-verified on 2026-10-24 to the frozen
v0.1 sets. This package is prepared for review; it is NOT a release until
the maintainer approves and publishes it (Zenodo upload needs a human
login — see `ZENODO_UPLOAD_CHECKLIST.md`).

## What is in this directory

| File | Contents |
|---|---|
| `landmarks_ohrc_pair_v0.1.csv` | 20 human-verified landmarks, OHRC 12:09 vs 14:06 (Shiv Shakti Point) |
| `landmarks_tmc2_fore_nadir_v0.1.csv` | 20 human-verified landmarks, TMC-2 fore vs nadir |
| `landmarks_ohrc_01_v0.2.csv` | 21 human-verified landmarks, OHRC 12:09 vs 14:06, 2048×2048 crop |
| `landmarks_ohrc_04_v0.2.csv` | 18 human-verified landmarks, OHRC 12:09 vs 14:06, 2048×2048 crop |
| `eval.py` | One-command evaluator (only dependency: numpy) |
| `LICENSE` | CC-BY-4.0 — the terms under which the landmark sets are released |
| `CHANGELOG.md` | Version history, including the v0.1 pin correction below |

## The task (fixed)

Register the pair: estimate the 2×3 affine transform mapping **source**
pixel coordinates to **reference** pixel coordinates (the `cv2.warpAffine`
convention: `x_ref = a*x_src + b*y_src + c`, `y_ref = d*x_src + e*y_src + f`).

To run the benchmark you need the input products (see "Inputs" below);
this package contains only the frozen landmark sets and the scorer.

## The metric (fixed)

- **Headline score: forward RMSE** = `sqrt(mean(||M·p_src − p_ref||²))` over
  all 20 landmarks, in pixels.
- **Inverse RMSE** is always reported alongside (via M⁻¹); both directions,
  neither hidden.
- Per-point residuals are included in the JSON report for inspection.

## One command

```bash
python eval.py --pair ohrc_pair --matrix "a,b,c,d,e,f" \
    --i-declare-gt-not-used-for-fitting
python eval.py --pair tmc2_fore_nadir --matrix-file my_transform.csv \
    --i-declare-gt-not-used-for-fitting --json report.json
python eval.py --pair ohrc_01 --matrix "a,b,c,d,e,f" \
    --i-declare-gt-not-used-for-fitting
python eval.py --self-test   # evaluator self-consistency checks
```

The declaration flag is mandatory: it records that the landmark set was not
used to fit the submitted transform. The evaluator additionally fail-closes
if the landmark CSV has been modified (sha256 pinned inside `eval.py`) and
reports whether the GT quadrant-coverage floor (≥3 per quadrant) holds.

## File format

Landmark CSV columns:

```
id, pair, x_ref, y_ref, x_src, y_src, ncc,
review_status, generated_by, reviewed_by, review_date,
reference_product, source_product, coordinate_space, caveat
```

- `(x_ref, y_ref)`: landmark in the reference image, pixels.
- `(x_src, y_src)`: the same terrain feature in the source image, pixels.
- `ncc`: normalized cross-correlation of the proposal chips (triage signal,
  not an accuracy claim).
- `caveat`: per-point reviewer notes (striping, shadows) where applicable.

## Inputs (not included)

- **ohrc_pair**: `ch2_ohr_nrp_20240425T1209509264_d_img_d18` (reference) and
  `ch2_ohr_nrp_20240425T1406019344_d_img_d18` (source); coordinates are in the
  canonical full-resolution pixel space with products loaded at
  `max_dimension=4096`.
- **tmc2_fore_nadir**: `ch2_tmc_ncf_20231101T0125121344_d_img_d18` (fore,
  reference) and `ch2_tmc_ncn_20231101T0125121377_d_img_d18` (nadir, source);
  coordinates are in the audited 2048×2048 windows
  (fore `Window(col_off=361, row_off=93328)`,
  nadir `Window(col_off=416, row_off=101611)`).
- **ohrc_01 / ohrc_04 (v0.2)**: same OHRC products as `ohrc_pair`
  (`ch2_ohr_nrp_20240425T1209509264_d_img_d18` reference,
  `ch2_ohr_nrp_20240425T1406019344_d_img_d18` source), but coordinates are
  in the **2048×2048 benchmark crops**
  (`data/benchmark_crops/ohrc_01_reference.png` /
  `ohrc_01_source.png`, `ohrc_04_reference.png` / `ohrc_04_source.png`).
  **Do not mix v0.2 crop coordinates with v0.1 full-resolution
  coordinates.** The v0.2 CSVs state this in every row's
  `coordinate_space` column.

Products are available from the PRADAN/ISSDC archive.

## Honest limitations (read before citing a number)

- Landmarks were proposed by Harris+NCC and confirmed by independent visual
  review (2026-10-05), 40/40 accepted. Review is at **feature-correspondence
  level, not sub-pixel**; points carry ~0.5–1 px NCC localization noise.
- The two views of each pair have real parallax (different orbits / fore vs
  nadir). The 20 landmarks are **not consistent with a single global affine
  to better than ~3–4 px** — a least-squares fit of the landmarks to
  themselves leaves 4.27 px (OHRC) / 3.49 px (TMC-2) residual. The benchmark
  is therefore **comparative**: every method is scored against the same
  independent points. Do not present a ChandraBench RMSE as an absolute
  sub-pixel accuracy claim.
- Reference behavior on release: our own pipeline scores 5.65 px (OHRC,
  LightGlue) and 27.35 px (TMC-2) forward RMSE against these landmarks —
  much worse than its inlier-based held-out RMSE (1.54 / 1.80 px). That
  disagreement is the point of an independent benchmark.
- No absolute (LROC/SLDEM) controls yet; see the tech note for the roadmap.

## v0.2 additions (39 landmarks, human-verified 2026-10-24)

- **ohrc_01** (21 points): SIFT+RANSAC candidates (NCC≥0.96 gate,
  spread-enforced, independent of the pipeline fit), every point verified
  by a human in the landmark clicker; ncc column blank (the clicker export
  carries no NCC values). Identity disparity 387.89 px; GT self-fit
  residual 1.40 px. **Coverage caveat: LT quadrant holds 2 points, below
  the ≥3/quadrant protocol floor — every eval report flags
  `floor_pass=false` for this pair.** Also the labeled-good pair for the
  strip-guard calibration (landmark RMSE 1.8869 px).
- **ohrc_04** (18 points): same protocol. Identity disparity 126.90 px;
  GT self-fit residual 1.51 px. **Coverage caveats: LB quadrant holds 2
  points (floor_pass=false); thin coverage, 18 points below the ~20
  target; UNLABELED for strip-guard calibration (landmark RMSE 2.2591 px
  fails the <2.0 px bar).**
- Both pairs are scored in **2048×2048 benchmark-crop coordinates**, not
  full-resolution coordinates — see "Inputs" above. The v0.1 pairs are
  unchanged and still score in their original spaces.
- Self-test change (documented, not hidden): the self-test now gates on
  evaluator arithmetic only (identity RMSE matches a direct computation;
  GT self-fit beats identity and matches a brute-force recomputation).
  Quadrant coverage is a property of the GT set, not of the evaluator, so
  it is **reported** per pair (with an explicit COVERAGE CAVEAT line for
  below-floor pairs) instead of gating the self-test. The per-pair
  `floor_pass` flag in scoring reports is unchanged.

## Versioning

v0.1 is frozen. The sha256 of each landmark CSV is pinned in `eval.py`; any
future version gets new filenames, new pins, and a changelog entry here.

**v0.1 pin correction (found while building v0.2, disclosed here): the
sha256 pins shipped in v0.1 never matched the frozen CSVs.** The CSVs
themselves are byte-identical to the `chandrabench-v0.1` tag — the frozen
content is genuine — but the v0.1 evaluator aborted on its own
circularity guard for both pairs. v0.2 corrects both pins to the true
hashes of the frozen files (verified programmatically); the CSVs are
untouched. If you ran the v0.1 evaluator and hit a sha256 mismatch, this
is why; it was a pin error, not tampering.

## Citation

See `docs/chandrabench-tech-note.md` for the tech note. Zenodo DOI:
`[reserved at upload — see ZENODO_UPLOAD_CHECKLIST.md]`.
