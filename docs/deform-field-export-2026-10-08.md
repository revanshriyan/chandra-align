# Dense-remap export: the fitted field ships in the download — 2026-10-08

## Problem

The deform-field stage improved the *scored* residuals and the verdict, but
the exported `warped_sec` (preview, checkerboard, blend, downloads) was still
produced by `cv2.warpAffine(sec, affine_matrix, ...)` -- the field never left
the telemetry. A SUCCESS_SUBPIXEL verdict whose download is affine-only is a
half-shipped feature.

## Design

When the stage applied (and was not voided by the fallback check), the export
path builds dense sampling maps from the *scored* forward map
F(p) = M_hook @ p + d(p) and remaps with `cv2.remap`:

- `apply_deform_field_stage` now also returns the fitted artifacts: `"field"`
  (the TPS tuple) and `"M_hook"` (the 2x3 the field corrects). Additive keys;
  the validated fit itself is untouched.
- `build_field_remap_maps(sec_shape, ref_shape, M_hook, field)` in
  `chandra_align/deform_field.py`:
  - evaluates d(p) once on the secondary grid (tiled; the (T, n_ctrl)
    kernel is shared between components in float32, 65536-point tiles,
    deterministic row order). Because the regularized TPS field is smooth,
    it is evaluated on a 2x-decimated grid and bilinearly upsampled; the
    total approximation vs the exact path (`_exact=True`: float64, full
    grid) is measured at <= 0.011 px max map diff and bounded by unit test
    (< 0.05 px). A 2048^2 build takes ~8 s at ~0.6 GB peak RSS.
  - inverts F per output pixel by fixed-point iteration on the affine
    sampling map S(q) = A^-1 (q - t): p_{k+1} = S(q - d(p_k)), 5 iterations.
    Converges geometrically (validated fields are smooth and fold-free, so
    ||A^-1 grad d|| << 1); iterations only bilinearly resample the
    precomputed displacement grid -- no new fitting, no new validation.
  - raises ValueError on any degenerate input (malformed/non-finite field,
    singular M_hook, bad shapes).
- `app.py` export (single production point of `warped_sec`, so preview,
  checkerboard, blend and downloads all carry it): flag-gated (`applied`
  must be true), try/except around the whole remap -- any failure falls back
  to the affine warp, bit-identical to before. Telemetry records
  `judge_metrics["warp_export_kind"]` = `"affine"` | `"field_remap"` |
  `"affine (field remap failed: ...)"`.
- `_jsonable_deform_field_info` excludes the raw `"field"`/`"M_hook"` keys
  (large arrays); the export reads them from the raw dict, not the JSON.

## Convention verification (empirical, not by reasoning)

`build_field_remap_maps` with a zero field reproduces `cv2.warpAffine`
to max 1 graylevel on random 256px imagery -- the sampling-map convention
S(q) = A^-1 (q - t) matches the pipeline's validated warpAffine behavior
(`tests/test_deform_field_export.py::test_zero_field_reproduces_warp_affine`).

## Inversion accuracy

Synthetic smooth field (3px/2px sinusoidal displacement, TPS fit, rotation +
scale + translation base): for 60 random secondary points, q = F(p), the
built maps recover p with max inversion residual < 0.05 px (bilinear map
sampling, no pixel-quantization noise). The export therefore reproduces the
scored geometry to ~1e-2 px, not just "better than affine".

## Fail-closed behavior

| Condition | Result |
|---|---|
| Flag off | `build_field_remap_maps` never called; `warp_export_kind="affine"`; raster bit-identical |
| Stage declined / voided | affine warp (hook never sets `applied`) |
| Malformed / non-finite field, singular M_hook | ValueError -> affine fallback + kind note |
| remap returns degenerate output | affine fallback + kind note |
| Any exception | affine fallback, never a crash |

## Tests

`tests/test_deform_field_export.py`: 5 unit tests run everywhere (convention,
inversion accuracy < 0.05 px, degenerate inputs raise, determinism,
stage returns field+M_hook); 3 app-level tests skip locally and run in CI
(flag-off untouched incl. monkeypatched-raiser proof, flag-on exports
`field_remap` differing from the affine fallback, byte-identical determinism).

## Independent photometric verification

ohrc_01/02/03, flag on: field-remapped export vs the affine-only warp of the
*identical* run (same matching/matrices; only the final warp differs).
Metric: patch NCC on a 20x20 check grid (15x15 patches, flat patches
skipped) -- independent of the TPS fit (which never saw intensities) and of
the stage's internal stride split.

| pair | gate RMSE (px) | patches | mean NCC field | mean NCC affine | frac field wins | mean MAD field | mean MAD affine |
|---|---|---|---|---|---|---|---|
| ohrc_01 | 0.4312 | 334 | 0.3868 | 0.0850 | 0.793 | 21.767 | 32.612 |
| ohrc_02 | 0.4687 | 396 | 0.5886 | 0.3275 | 0.818 | 13.951 | 21.561 |
| ohrc_03 | 0.3427 | 398 | 0.4727 | 0.1831 | 0.761 | 21.217 | 31.297 |

(`results/table_export_photometric.csv`.) The field-remapped export wins
76-82% of independent check patches with 2-4x higher mean NCC and ~35%
lower mean absdiff. Absolute NCC is modest (15px patches on lunar terrain
with inter-observation illumination change); the relative comparison is the
signal, and it is decisive: the exported raster really does align better,
not just score better. Gate RMSEs reproduce the A19 validation numbers.

## Caveats

- The remap inverts the *scored* map M_hook + d, not the final `affine_matrix`
  (a linear re-summary fit downstream on corrected points). The export is
  faithful to the verdict the gate scored.
- Fixed-point inversion assumes the field is fold-free; the stage's folding
  fuse (min Jacobian det >= 0.5) guarantees the contraction. A field that
  passed the fuse cannot break the inverter -- and any non-finite iterate
  still fails closed to affine.
- Cost: ~8 s for a 2048^2 export build (~0.6 GB peak); the photometric
  verification below ran against the exact (_exact) implementation, whose
  maps differ from the shipped fast path by <= 0.011 px.
- Known nuance: the error-vector overlay still draws affine-based vectors
  (`affine_matrix`), so on field-remapped exports it can overstate the
  residual arrows; the raster itself, the diff map, the checkerboard and the
  blend all carry the field warp. Left as-is (diagnostic-only).
