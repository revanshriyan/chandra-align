# Full Sun-Angle Sweep (96 cases) — 2026-10-08

## Setup and deviation note

The pilot script that produced `results/table_sunangle_pilot.csv` was **never
committed** (commit `c1b8552` added the CSV alone; no render/match script exists
in the repo). This sweep therefore uses a documented fallback that reproduces
the pilot's *described* setup as closely as possible:

- **DTM:** center 500×500 crop of `NAC_DTM_VIKRAMSITE1.TIF` (3 m/px, Vikram
  site), window rows 7697–8197, cols 3584–4084 of the 15895×7668 float32 grid.
- **Render:** Lambertian (surface normal · sun vector, clamped ≥ 0) plus subtle
  albedo variation (1 + 0.12 × Gaussian-smoothed white noise, σ = 6 px). Renders
  are converted to uint8 with one common 1st–99th percentile stretch across all
  96 renders (relative brightness preserved).
- **Reference:** az = 0°, el = 30° (same as pilot).
- **Grid:** azimuth 0°–345° in 15° steps (24) × elevation {10°, 30°, 50°, 70°} (4)
  = **96 cases**, deterministic order (azimuth ascending, elevation ascending).
- **Matcher:** SIFT `nfeatures=4000`, BF-L2, Lowe ratio 0.75, partial-affine
  RANSAC 3.0 px — the pilot's stated settings. CPU only.
- **Verdicts:** the repo's own frozen gate (`compute_quadrant_metrics` +
  `validate_registration_gate`, read-only import): ACCEPT needs RMSE ≤ 0.50 px,
  ≥ 8 inliers, entropy ≥ 0.75, ≥ 3 quadrants; COARSE needs RMSE ≤ 2.50 px,
  ≥ 8 inliers, entropy ≥ 0.50, ≥ 2 quadrants; anything else is DEGENERATE.
- **Truth:** every render shares the same DEM grid, so exact truth is the
  identity transform. Truth RMSE is computed on an **independent 9-point grid**
  (0.15/0.5/0.85 of each axis) mapped by the RANSAC-estimated transform —
  never on the inliers used to fit it (Phase 11 lesson: inlier self-consistency
  ≠ correctness).

## Seeds (deterministic)

- `np.random.seed(7)`; albedo field from `np.random.default_rng(7)`
- `cv2.setRNGSeed(7)` before every RANSAC call (match and RMSE refit)

## Verdict counts

| Verdict | Count |
|---|---|
| SUCCESS_SUBPIXEL (ACCEPT) | 1 (identity self-match, az=0/el=30, 363 inliers, 0.0000 px) |
| COARSE_ADVISORY | 65 |
| DEGENERATE_FAILURE | 30 |

## Failure curve (azimuth difference from reference vs elevation)

Inliers / inlier RMSE / truth RMSE / verdict:

| az diff | el=10 | el=30 | el=50 | el=70 |
|---|---|---|---|---|
| 0° | 14 / 0.61 / 0.42 COARSE | 363 / 0.00 / 0.00 **ACCEPT** | 194 / 0.86 / 0.10 COARSE | 100 / 1.11 / 0.13 COARSE |
| 15° | 8 / 1.71 / 1.40 COARSE | 266 / 0.67 / 0.03 COARSE | 178 / 0.90 / 0.10 COARSE | 84 / 1.09 / 0.36 COARSE |
| 30° | 0 DEGEN | 178 / 0.95 / 0.15 COARSE | 139 / 0.95 / 0.22 COARSE | 95 / 1.16 / 0.27 COARSE |
| 45° | 0 DEGEN | 107 / 1.17 / 0.13 COARSE | 121 / 1.06 / 0.16 COARSE | 88 / 1.20 / 0.42 COARSE |
| 60° | 0 DEGEN | 58 / 1.16 / 0.22 COARSE | 95 / 1.10 / 0.21 COARSE | 79 / 1.17 / 0.22 COARSE |
| 75° | 0 DEGEN | 25 / 1.13 / 0.37 COARSE | 70 / 1.10 / 0.13 COARSE | 67 / 1.22 / 0.32 COARSE |
| 90° | 0 DEGEN | 16 / 1.28 / 0.30 COARSE | 49 / 1.17 / 0.36 COARSE | 64 / 1.28 / 0.28 COARSE |
| 105° | 0 DEGEN | 0 DEGEN | 35 / 1.12 / 0.29 COARSE | 53 / 1.37 / 0.13 COARSE |
| 120° | 0 DEGEN | 0 DEGEN | 30 / 1.12 / 0.24 COARSE | 45 / 1.27 / 0.37 COARSE |
| 135° | 0 DEGEN | 0 DEGEN | 26 / 1.39 / 0.64 COARSE | 36 / 1.26 / 0.21 COARSE |
| 150° | 0 DEGEN | 0 DEGEN | 23 / 1.36 / 0.64 COARSE | 33 / 1.32 / 0.26 COARSE |
| 165° | 0 DEGEN | 0 DEGEN | 24 / 1.28 / 0.52 COARSE | 33 / 1.18 / 0.21 COARSE |
| 180° | 0 DEGEN | 0 DEGEN | 20 / 1.31 / 0.37 COARSE | 33 / 1.41 / 0.52 COARSE |

### Supplemental fine probe (az = 5°, 10°, same setup, not in the CSV)

| az | el=10 | el=30 | el=50 | el=70 |
|---|---|---|---|---|
| 5° | 15 / 0.77 / 0.46 COARSE | 334 / 0.35 / 0.03 **ACCEPT** | 195 / 0.88 / 0.07 COARSE | 96 / 1.05 / 0.19 COARSE |
| 10° | 15 / 1.14 / 0.76 COARSE | 298 / 0.50 / 0.04 COARSE | 195 / 0.93 / 0.06 COARSE | 96 / 1.11 / 0.32 COARSE |

## Findings

1. **The pilot's "15°+ DEGENERATE" claim does NOT reproduce here.** At 15°
   azimuth difference every elevation is COARSE (8–266 inliers, truth RMSE
   0.03–1.40 px — transforms verified correct on the independent grid). The
   fine probe shows 5° even reaches ACCEPT at el=30 (0.35 px) and 10° is COARSE
   everywhere. The true azimuth failure boundary in this setup is much wider:
   at el=30 it lies between 90° and 105°; at el=50/70 matching never fails
   across the full 180° (inliers decline 194 → 20–33, but transforms stay
   correct, truth RMSE 0.13–0.64 px).
2. **Low elevation is the fragile axis, not elevation change per se.**
   Elevation changes are harmless when the azimuth difference is small (az=0
   row: all COARSE; 5°/10° probes: all COARSE/ACCEPT). But at el=10°,
   azimuth differences ≥ 30° are fully DEGENERATE (0 correspondences — long
   shadows wipe out texture). The pilot's own CSV row az=0/el=15 was
   `DEGENERATE_NO_FEATURES`, which already contradicted its doc's "tolerated to
   ±30°" line; this sweep confirms low elevation is only safe near the
   reference azimuth.
3. **Phase 11 lesson confirmed again.** Inlier RMSE (typically 1.0–1.4 px) is
   systematically larger than truth RMSE (0.03–0.6 px) on accepted COARSE cases:
   the gate's residual metric is conservative relative to true transform error,
   and self-consistency alone would mis-rank these fits. Every verdict above is
   backed by the independent identity-grid check.
4. The identity self-match (az=0, el=30) is a clean deterministic sanity check:
   363/363 correspondences as inliers, 0.0000 px — the pipeline and seed
   handling are bit-stable.

## Interpretation and caveats

- The pilot reported total failure at ≥15° azimuth on its (uncommitted) render
  setup; this wider envelope likely comes from differences in the render model
  (albedo amplitude/texture content of the chosen DTM window, normalization).
  The envelope width is sensitive to how much illumination-independent texture
  the scene has. Both results agree on the *direction*: azimuth hurts, and low
  elevation combined with azimuth change is fatal. Neither should be quoted as
  a universal SIFT limit without naming the render setup.
- This is simulated Lambertian shading on one 500×500 DTM window — not orbital
  imagery. The operational conclusion for real pairs stands: small azimuth
  differences (a few degrees, e.g. same-day OHRC) are safe; opposite-sun pairs
  need an illumination-invariant front-end.

## Data

- `results/table_sunangle_full.csv` — 96 rows, one per (azimuth, elevation),
  columns: azimuth_deg, elevation_deg, correspondences, inliers,
  inlier_rmse_px, truth_rmse_px, entropy, quadrants, verdict, runtime_s.
- Render + match script: `/tmp/sunangle_full_sweep.py` (fallback; pilot script
  was missing from the repo).
- Supplemental 8-case fine probe: `/tmp/sunangle_supp.log` (numbers in the
  table above).
