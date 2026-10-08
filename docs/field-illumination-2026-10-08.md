# Deformation Field × Illumination: Clean Negative — 2026-10-08

Worker A24. **The deform-field stage does not help cross-sun pairs.** This is a
clean negative against pre-registered bars, and it clarifies the stage's scope:
it corrects spatially-varying *geometric* distortion, but cross-sun failure is
*photometric* (the matcher finds no correspondences), so there is nothing for
the stage to refine.

## Setup

Five synthetic cross-sun cases, reconstructed from the documented recipe in
`docs/sunangle-full-sweep-2026-10-08.md`:

- **DTM:** 500×500 crop (rows 7697–8197, cols 3584–4084) of
  `NAC_DTM_VIKRAMSITE1.TIF` (3 m/px, Vikram site). The crop contains a small
  nodata patch (566 px, 0.23%); inpainted before rendering.
- **Render:** Lambertian (surface normal · sun vector, clamped ≥ 0) with albedo
  variation. **Deviation from the sweep recipe:** the sweep's "1 + 0.12 ×
  Gaussian-smoothed white noise (σ=6px)" yields only ~±1% effective albedo
  variation after smoothing, which on this smooth terrain (mean slope 4.4°)
  produces renders too weak for ANY matcher (raw SIFT: 3–4 Lowe matches at
  60° azimuth). This experiment uses 0.30 amplitude to create matchable texture
  while keeping illumination-dependent shading dominant. The sweep's exact
  difficulty was not reproduced; the cases here are harsher photometrically.
- **Reference:** az=0°, el=30° for all cases. Truth is identity (same DEM grid).
- **Pipeline:** true `app.py::_align_core`, flag OFF vs flag ON
  (`CHANDRA_DEFORM_FIELD=1`), `pixel_scale_m=0.25`, seed 7, deterministic.
- **Truth check:** recovered transform `H` applied to an independent 9-point
  grid (0.15/0.5/0.85 of each axis), RMSE vs identity — the Phase 11 lesson.

## Results

| Case (el/az) | Sweep verdict | Flag OFF | Flag ON | Stage applied? |
|---|---|---|---|---|
| 30° / 60° | COARSE (58 inl) | DEGENERATE, 0 inl (fail-closed) | DEGENERATE, 0 inl (fail-closed) | No — no correspondences |
| 30° / 90° | COARSE (16 inl) | DEGENERATE, 0 inl (fail-closed) | DEGENERATE, 0 inl (fail-closed) | No — no correspondences |
| 30° / 105° | DEGENERATE (0 inl) | DEGENERATE, 0 inl (fail-closed) | DEGENERATE, 0 inl (fail-closed) | No — no correspondences |
| 50° / 90° | COARSE (49 inl) | DEGENERATE, 0 inl (fail-closed) | DEGENERATE, 0 inl (fail-closed) | No — no correspondences |
| 10° / 30° | DEGENERATE (0 inl) | SUCCESS, 29 inl, 0.4px, truth 0.1px | SUCCESS, 20 inl, 0.3px, truth 0.1px | No (`df_applied=False`) |

Full table: `results/table_field_illum.csv`. Script: `scripts/field_x_illumination.py`.

## Bar verdicts

**(a) Win requires verdict upgrade on ≥2 cases, zero downgrades: CLEAN NEGATIVE.**
Zero upgrades, zero downgrades. Flag-on never changes any verdict. The stage
does not help cross-sun pairs in this test.

**(b) Overfit check: PASS.** On the one case with a fit (el10_az030), truth RMSE
is 0.1px on both flags — the transform is correct, not confident-but-wrong.
On the four DEGENERATE cases there is no transform (fail-closed, correct).
The stage did not produce any confident-but-wrong geometry.

**(c) Determinism: PASS.** el10_az030 flag-on re-run identical.

## Why the stage cannot help here

The deform-field stage operates on the affine fit's inliers: it models
*residual geometric distortion* after a successful match. Cross-sun failure is
not residual distortion — it is *absence of correspondences*. At 60–105°
azimuth difference, SIFT descriptors computed under one illumination do not
match descriptors under another (shadows move, gradients invert). With zero
inliers, the pipeline fails closed before the stage is even considered.

On the one matchable case (el10_az030), the stage declined (`df_applied=False`):
the 0.3–0.4px residual is photometric noise, not the smooth spatial distortion
the TPS field models. The flag-on SUCCESS (vs flag-off SUCCESS) comes from the
flag-gated quota (uncapped keypoints), not the stage.

## Does the stage make anything worse? (task point 4)

No. Zero downgrades across all five cases. The stage never applied, so it
could not fit shadow-induced false matches. Where it might have (el10_az030),
it correctly declined.

## Interpretation

This negative is informative, not disappointing. It draws the stage's boundary:

- **Stage helps:** same-sun pairs where the matcher succeeds but a single
  affine cannot represent spatially-varying distortion (the OHRC/TMC-2 wins).
- **Stage cannot help:** cross-sun pairs where the matcher fails photometrically.
  That failure needs an illumination-invariant front-end (e.g. the
  phase-congruency work), not geometric refinement.

The honest next lever for sun-angle robustness is not the deform field — it is
the matcher front-end. The stage's five sub-pixel wins stand; its scope is now
better defined.

## Files

- `scripts/field_x_illumination.py` — render + experiment (deterministic, seed 7)
- `results/table_field_illum.csv` — 10 rows (5 cases × 2 flags)
- `docs/field-illumination-2026-10-08.md` — this file

No `app.py` / `chandra_align/` changes. No push (per instructions).
