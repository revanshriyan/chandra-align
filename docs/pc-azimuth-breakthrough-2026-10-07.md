# Phase Congruency Cracks the Azimuth Problem — 2026-10-07

## Problem
Sun-angle pilot showed SIFT fails completely when solar azimuth differs
by >10° between images. This limits real-pair matching to same-day
acquisitions.

## Solution Tested
Phase-congruency front-end (existing repo code:
`chandra_align/xmodal/phase_congruency.py`) + SIFT on PC maps instead
of raw intensities.

## Results (azimuth difference, el=30 fixed)

| Az diff | Raw SIFT | PC+SIFT | Transform verified |
|---------|----------|---------|-------------------|
| 10° | 11 inl, 1.01px COARSE | 588 inl, 0.65px COARSE | ✓ near-identity |
| 30° | DEGENERATE (0 inl) | 18 inl, 0.90px COARSE | ✓ near-identity |
| 90° | DEGENERATE | DEGENERATE | — |
| 180° | DEGENERATE | 85 inl, 0.91px COARSE | ✓ near-identity |

## Verification
All PC+SIFT transforms checked against known identity (same DTM patch):
translation <0.4px, scale error <0.05%, rotation <0.05°. All correct —
not false positives.

## Impact
- Azimuth tolerance: 10° → 30°+ (3x improvement)
- Inlier count at 10°: 11 → 588 (53x improvement)
- Even handles 180° (opposite sun) — shadows invert but structure persists
- 90° remains a failure mode (specific geometry, under investigation)

## Why it works
Phase congruency detects structural features (ridges, crater rims) from
local phase alignment, independent of illumination direction. SIFT
descriptors on PC maps encode structure, not shading.

## Path forward
Wire PC front-end as an opt-in arm in the pipeline (Phase 14). Gate
thresholds unchanged. This directly addresses the #1 gap: real pairs with
larger time separation become matchable.

## Data
- DTM: NAC_DTM_VIKRAMSITE1, 500x500 window
- PC: Kovesi Log-Gabor, 4 scales × 6 orientations (repo implementation)
- SIFT: nfeatures=4000, Lowe 0.75, RANSAC 3.0px, seed 7
