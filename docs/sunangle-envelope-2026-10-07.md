# Sun-Angle Robustness Envelope — 2026-10-07

## Method
Systematic illumination sweep on LROC DTM (500x500 px at 3m/px, Vikram site).
Lambertian renders at grid of solar positions; SIFT matching between fixed
reference (az=0, el=30) and each variant. CPU-only, deterministic (seed 7).
All outcomes reported including DEGENERATE.

## Key Findings

### 1. Elevation is harmless; azimuth is fatal
- Same azimuth, elevation 30→45 or 30→60: ACCEPT at 0.15-0.25px RMSE (sub-pixel!)
- Any azimuth change ≥15°: DEGENERATE (0 inliers, complete failure)
- Azimuth tolerance threshold: ~10-12°

### 2. Precise azimuth degradation curve (el=30 fixed)
| Azimuth diff | Correspondences | Inliers | RMSE | Verdict |
|--------------|-----------------|---------|------|---------|
| 0° (identity)| 12 | 12 | 0.000px | ACCEPT |
| 5° | 11 | 11 | 0.518px | COARSE |
| 10° | 11 | 11 | 1.005px | COARSE |
| 15° | 8 | 0 | — | DEGENERATE |
| 20°+ | 0-7 | 0 | — | DEGENERATE |

### 3. Why our real OHRC pairs work
The two OHRC products (2024-04-25 12:09 and 14:06) are 2 hours apart.
Solar azimuth changes ~7-8° in 2 hours at this latitude — inside the 10°
tolerance. Pairs with larger time separation would fail.

### 4. Implication
SIFT descriptors encode gradient direction, which rotates with sun azimuth.
This is a fundamental limitation, not a tuning problem. Illumination-
invariant front-ends (phase congruency) are required for wider envelopes.

## Operational Envelope (SIFT, lunar imagery)
- Solar azimuth difference: must be <10° for COARSE, <5° for ACCEPT
- Solar elevation difference: tolerated to at least ±30° (sub-pixel)
- Outside envelope: fail-closed DEGENERATE (honest, not wrong)

## Data
- DTM: NAC_DTM_VIKRAMSITE1, 500x500 window at center
- Renders: Lambertian + subtle albedo variation
- SIFT: nfeatures=4000, Lowe 0.75, RANSAC 3.0px
- Full grid results: results/table_sunangle_pilot.csv
