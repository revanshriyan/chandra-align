[![MIT License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)

# CHANDRA-ALIGN

**SIH 2026 Problem Statement ID26166**
"Multi-modal, Sun angle and scale invariant image correspondence using
Chandrayaan-2 optical images (OHRC, TMC and IIRS)"

**Team:** ChandraVision

CHANDRA-ALIGN is a lunar image registration application for matching
Chandrayaan-2 OHRC, TMC and IIRS imagery across changes in sensor, sun angle and
scale. The pipeline preprocesses the images, attempts RIFT2/LightGlue/ALIKED,
falls back to SIFT when needed, estimates correspondence with RANSAC, applies
sub-pixel refinement and a validation gate, then exports accepted results.

## How It Works

1. **Preprocessing** normalizes the input images for cross-sensor, sun-angle,
	and scale differences.
2. **RIFT2** is the primary feature-matching path.
3. **LightGlue/ALIKED** provide deep feature matching.
4. **SIFT + Brute-Force RANSAC** is the CPU fallback path.
5. **Sub-pixel refinement** improves the estimated correspondence positions.
6. **Validation gate** labels the result **ACCEPT**, **COARSE ADVISORY**, or
	**REJECT** before export.

## Benchmark Status

Current measured results use the SIFT/RANSAC CPU fallback. RIFT2/LightGlue GPU
validation is pending; no performance claim is made for that path.

### Measured Results

- **Synthetic pair:** RMSE 0.3840 px, 62 inliers, entropy 1.9532, 4/4 quadrants — **ACCEPTED**
- **Real OHRC crop:** 1.5425 px, 9 inliers, 3/4 quadrants — **COARSE ADVISORY**
- **Real TMC-2 pair:** 1.3312 px (~5.95 m at 4.47 m/px), 9 inliers, 2/4 quadrants — **COARSE ADVISORY**

## Run Locally

Requires Python 3.10 or newer. Full-resolution source data is not included;
obtain the required Chandrayaan-2 products through ISRO PRADAN.

```powershell
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
python app.py
```

## Live Demo

Demo available on request.

## Repository Layout

```text
chandra_align/          Core registration package
app.py                  Gradio application
tests/                  Automated tests
docs/                   Project and workflow documentation
scripts/                Data and evaluation utilities
notebooks/              Experiment notebooks
examples/benchmarks/    Example benchmark images
data/                   No imagery ships here; obtain Chandrayaan-2 products via ISRO PRADAN
```

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE).
