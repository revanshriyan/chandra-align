# Chandra-Align: Sub-Pixel Planetary Raster Alignment & Feature Matching Engine

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-brightgreen.svg)]()
[![Accuracy](https://img.shields.io/badge/RMSE-%3C0.50px%20Sub--Pixel-success)]()
[![Demo Access](https://img.shields.io/badge/Live%20Demo-Private%20%2F%20By%20Request-orange)](#)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

> **Note on Live Demo:** The interactive deployment runs on a private instance for evaluation and review. For access tokens or demo credentials, please contact the repository maintainers directly.

The accuracy badge reflects the calibrated synthetic benchmark below; it is not a guarantee for every image pair. The public repository and its documentation intentionally omit private demo endpoints.

**SIH 2026 Problem Statement ID26166**
"Multi-modal, Sun angle and scale invariant image correspondence using Chandrayaan-2 optical images (OHRC, TMC and IIRS)"

**Team:** ChandraVision

CHANDRA-ALIGN registers lunar imagery across sensor, illumination, and scale differences. It supports Chandrayaan-2 OHRC, TMC-2, and IIRS products, with PDS4 labels used to interpret supported source rasters.

## How It Works

1. **Preprocessing** prepares image data for cross-sensor matching while respecting bounded image-size limits.
2. **RIFT2 (non-functional, under investigation; not the primary matcher):** the Sept. 30 CPU audit and issue #1 RTX 5070 GPU validation returned 0–1 correspondences per pair, with no valid fits. See [`results/table_issue01_gpu_validation.csv`](results/table_issue01_gpu_validation.csv).
3. **LightGlue/ALIKED GPU matching** is the validated deep-learning path: synthetic sub-pixel ACCEPT at 0.3695 px / 43 inliers; competitive real-pair results at 1.7961 px (OHRC) and 1.5558 px (TMC-2).
4. **SIFT + Brute-Force RANSAC CPU fallback** estimates the transform as the validated CPU fallback.
5. **Sub-pixel refinement** attempts NCC/parabolic refinement of the RANSAC estimate. If too few correspondences survive refinement, the pipeline retains the RANSAC model rather than promoting an unsupported refined fit.
6. **Fail-safe spatial validation gate** classifies the result as **ACCEPT**, **COARSE ADVISORY**, or **REJECT** before export.

### Spatial Validation Gate

- **ACCEPT (sub-pixel):** RMSE ≤ 0.50 px, configured minimum inliers, spatial entropy ≥ 0.75, and support in at least 3 of 4 quadrants.
- **COARSE ADVISORY:** RMSE ≤ 2.50 px, configured minimum inliers, entropy ≥ 0.50, and support in at least 2 quadrants. This is a regional fit advisory, not a sub-pixel acceptance.
- **REJECT:** any result that does not meet either tier. Non-finite metrics fail closed; rejected fits do not expose transform telemetry, and invalid RMSE is reported as unavailable rather than a misleading zero.

## High-Resolution Handling

PDS4 `.IMG` products require their associated XML labels so the raster layout and metadata can be interpreted safely. Source products are processed through bounded previews/crops rather than loading every full-resolution raster into the matching pipeline. The verified OHRC run used a 4096-pixel cap; the measured TMC-2 result used browse-aligned 2048 × 2048 crops. A larger 4096 × 4000 TMC-2 crop exceeded approximately 5.8 GB of working memory in the tested CPU environment and was stopped. These measurements do not establish that arbitrary full-frame inputs fit within a fixed memory budget.

## Benchmark Status

The matcher measurements below are recorded in [`results/table_issue01_gpu_validation.csv`](results/table_issue01_gpu_validation.csv). RIFT2 returned 0, 1, and 0 correspondences for `synthetic_gentle`, `OHRC_pair`, and `TMC2_fore_nadir`, respectively; each RIFT2 row is rejected.

| Pair | Matcher | Correspondences | RMSE (px) | Inliers | Entropy | Quadrants | Gate | Runtime (s) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| synthetic_gentle | RIFT2 | 0 | N/A | 0 | 0.0000 | 0/4 (0,0,0,0) | REJECTED: Only 0 correspondences | 22.6662 |
| synthetic_gentle | LightGlue_ALIKED | 1276 | 0.3695 | 43 | 1.9988 | 4/4 (11,10,11,11) | ACCEPTED (Sub-Pixel Precision) | 17.7092 |
| synthetic_gentle | SIFT_RANSAC | 150 | 0.3980 | 54 | 1.9513 | 4/4 (17,8,15,14) | ACCEPTED (Sub-Pixel Precision) | 2.0875 |
| OHRC_pair | RIFT2 | 1 | N/A | 0 | 0.0000 | 0/4 (0,0,0,0) | REJECTED: Only 1 correspondences | 42.0472 |
| OHRC_pair | LightGlue_ALIKED | 553 | 1.7961 | 34 | 1.9367 | 4/4 (6,6,11,11) | COARSE ALIGNMENT (Regional Fit Advisory) | 2.2815 |
| OHRC_pair | SIFT_RANSAC | 104 | 1.5425 | 9 | 0.9864 | 3/4 (7,0,1,1) | COARSE ALIGNMENT (Regional Fit Advisory) | 4.3942 |
| TMC2_fore_nadir | RIFT2 | 0 | N/A | 0 | 0.0000 | 0/4 (0,0,0,0) | REJECTED: Only 0 correspondences | 78.0995 |
| TMC2_fore_nadir | LightGlue_ALIKED | 1537 | 1.5558 | 13 | 1.6692 | 4/4 (2,3,1,7) | COARSE ALIGNMENT (Regional Fit Advisory) | 3.1617 |
| TMC2_fore_nadir | SIFT_RANSAC | 106 | 2.2259 | 24 | 1.7296 | 4/4 (4,6,2,12) | COARSE ALIGNMENT (Regional Fit Advisory) | 7.1537 |

The independent ground-truth protocol is documented in [`docs/ground-truth-protocol.md`](docs/ground-truth-protocol.md), and its split-held-out versus independent-RMSE comparison is recorded in [`results/table_issue02_ground_truth.csv`](results/table_issue02_ground_truth.csv); OHRC and TMC-2 independent landmarks remain pending human picking.

### Example Results

<p align="center">
	<img src="docs/images/gate-checklist-accepted.png" width="700" alt="LightGlue/ALIKED accepted synthetic-pair checklist" />
</p>

LightGlue/ALIKED on the calibrated synthetic pair — RMSE 0.3695 px, 43 inliers, entropy 1.9988, 4/4 quadrants: ACCEPTED (sub-pixel), measured on RTX 5070.

<p align="center">
	<img src="docs/images/gate-checklist-rejected.png" width="700" alt="RIFT2 rejected synthetic-pair checklist" />
</p>

RIFT2 produced 0 correspondences on the same input — rejected with transform telemetry masked (N/A). The failed primary is reported, not hidden.

<p align="center">
	<img src="docs/images/viz-lightglue-synthetic-checkerboard.png" width="700" alt="LightGlue/ALIKED calibrated synthetic registration checkerboard with inlier points" />
</p>

LightGlue/ALIKED, calibrated synthetic pair — checkerboard blend of the accepted registration (RMSE 0.3695 px, 43 inliers, 4/4 quadrants). Measured on RTX 5070.

<p align="center">
	<img src="docs/images/viz-lightglue-ohrc-overlay.png" width="700" alt="LightGlue/ALIKED OHRC reference raster with 34 inlier points across four quadrants" />
</p>

LightGlue/ALIKED, OHRC pair — 34 inliers across all four quadrants at 1.7961 px: COARSE advisory, not sub-pixel. Measured on RTX 5070.

## Run Locally

Requires Python 3.10 or newer. Full-resolution Chandrayaan-2 source imagery is not included; obtain products and their labels through ISRO PRADAN.

```powershell
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
python app.py
```

## Private Demo Access

The interactive deployment is private and available for evaluation/review by request. No Space address, demo endpoint, access token, or credential is published in this repository.

## Repository Layout

```text
chandra_align/          Core registration package
app.py                  Gradio application
tests/                  Automated tests
docs/                   Project and workflow documentation
scripts/                Data and evaluation utilities
notebooks/              Experiment notebooks
examples/benchmarks/    Example benchmark images
data/                   No source imagery ships here; obtain products via ISRO PRADAN
```

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE).
