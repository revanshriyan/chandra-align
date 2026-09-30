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

## Benchmark Status

Current measured results use the SIFT/RANSAC CPU fallback. RIFT2/LightGlue GPU
validation is pending; no performance claim is made for that path.

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

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE).
