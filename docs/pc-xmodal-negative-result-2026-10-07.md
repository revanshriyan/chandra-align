# Cross-Modal Phase-Congruency Test — Negative Result 2026-10-07

## Setup
OHRC optical (downsampled 12x to 3m/px, 500x500) vs VIKRAMSITE1 DTM hillshade
(500x500, az 315/el 45). Phase-congruency maps computed via
chandra_align/xmodal/phase_congruency.py (Kovesi Log-Gabor, nscale=4, norient=6),
SIFT on PC maps.

## Results
| Stage | OHRC keypoints | Hillshade keypoints | Lowe matches | RANSAC inliers |
|---|---|---|---|---|
| Raw images | 675 | 54 | 7 | 3 (degenerate) |
| PC maps | 5190 | 6630 | 3 | n/a (<4) |

Phase congruency dramatically increases detectable structure (54→6630
keypoints on hillshade) but SIFT descriptors on PC maps still do not match
across the optical↔shaded-relief modality gap. Only 3 Lowe-ratio matches —
below the RANSAC minimum.

## Interpretation
PC marks structure invariant to brightness/contrast, but the *local descriptor
neighborhoods* remain modality-specific: an optical crater rim and its hillshade
rendition produce different gradient distributions even when the underlying
structure aligns. Detector fires; descriptor fails.

## Implication for LROC adjudication
Off-the-shelf SIFT (±PC front-end) is insufficient for OHRC↔DTM-hillshade.
Options, hardest first:
1. Learned cross-modal matcher (needs GPU + training data) — research project.
2. Match OHRC against the 1m LROC orthophoto instead (optical↔optical) —
   blocked on obtaining the orthophoto GeoTIFF.
3. Manual tie-point transfer (~10-15 crater rims) for coarse adjudication —
   labor-intensive but reliable at the ~12-24px level, sufficient for the
   5.65px question.

This negative result is recorded per the project's honest-reporting norm:
not every experiment works, and the failure mode is informative.
