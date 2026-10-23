# Chandra-Align: Sub-Pixel Planetary Raster Alignment & Feature Matching Engine

[![Python 3.12+](https://img.shields.io/badge/python-3.12%2B-brightgreen.svg)]()
[![Gate threshold](https://img.shields.io/badge/gate-inlier%20RMSE%20%E2%89%A4%200.50%20px-blue)]()
[![Demo Access](https://img.shields.io/badge/Live%20Demo-Private%20%2F%20By%20Request-orange)](#)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.23217163.svg)](https://doi.org/10.5281/zenodo.23217163)

> **Note on Live Demo:** The interactive deployment runs on a private instance for evaluation and review. For access tokens or demo credentials, please contact the repository maintainers directly.

The gate badge states the frozen sub-pixel gate threshold (inlier RMSE ≤ 0.50 px); it is a threshold, not a measured result, and not a guarantee for every image pair. The public repository and its documentation intentionally omit private demo endpoints.

**SIH 2026 Problem Statement ID26166**
"Multi-modal, Sun angle and scale invariant image correspondence using Chandrayaan-2 optical images (OHRC, TMC and IIRS)"

**Team:** ChandraVision

CHANDRA-ALIGN registers lunar imagery across sensor, illumination, and scale differences. It supports Chandrayaan-2 OHRC, TMC-2, and IIRS products, with PDS4 labels used to interpret supported source rasters.

## Results at a glance

| Claim | Measured | Basis |
| --- | --- | --- |
| Real-pair registration residual (self-consistency, not independent truth) | **0.27–0.45 px** on 9 pairs — 5 OHRC + 4 TMC-2 (17–24 inliers; rescue pairs use 740–2,162 admitted points). Non-circular 80/20 holdout verification on all 9 pairs: **0.25–0.36 px**, admitted fractions 8–32% of held-out matches (unconditional median residual 1.7–11.3 px, stated alongside; [`results/table_rescue_noncircular.csv`](results/table_rescue_noncircular.csv)). Stricter spatial-extrapolation score: **1.99–3.88 px** (harder benchmark, stated alongside). Independent landmark RMSE is OHRC 5.6510 px and TMC-2 27.3519 px (ChandraBench v0.1, canonical table). | Frozen gate, opt-in deformation-field stage + L1 field-guided rescue, held-out scoring |
| Absolute accuracy vs LROC truth | **1.7–2.3 m** (1 m grid; 6.5–8.8 OHRC px at 0.26 m/px), **2.0–2.5 m** (3 m grid; 7.7–9.6 OHRC px) | Independent held-out check points, no leakage |
| Synthetic control | 0.31–0.40 px, transforms recovered to ~0.1 px / 0.01° | Exact ground truth |
| Cross-modal (IIRS↔TMC-2) | 0.98 px COARSE_ADVISORY (XoFTR, W5, 23 inliers); 16/19 pairs COARSE_ADVISORY | Best single-window and broadest coverage to date |
| Failure behavior | Fail-closed: rejects report reasons, never a transform | 45-run robustness battery + sun-angle envelope |

Every number links to its run output below. Caveats are stated with the claims, not buried. Backend and hardware for each result are recorded in its section below and per entry in [`results/table_canonical.json`](results/table_canonical.json); where a run's backend or hardware is not recorded anywhere, the section says "not recorded".

## How It Works

![CHANDRA-ALIGN end-to-end data flow: PDS4 products to verdict and exports](https://raw.githubusercontent.com/revanshriyan/chandra-align/main/docs/images/pipeline-overview.svg)

**The pipeline in one picture:** Chandrayaan-2 products (OHRC, TMC-2, IIRS) enter with their PDS4 labels, get preprocessed into bounded crops, run through a matcher cascade, get a RANSAC fit, face a fail-closed gate, and leave as verdict + telemetry + exports. Every number below is measured — the single source of truth is [`results/table_canonical.json`](results/table_canonical.json), generated from raw run outputs by [`scripts/generate_results_table.py`](scripts/generate_results_table.py) (issue #8), and [`docs/model-card.md`](docs/model-card.md) is the honest benchmark card generated from it.

### 1. Matcher cascade

![Matcher cascade: LightGlue/ALIKED primary, RIFT2 opt-in only, SIFT/RANSAC CPU fallback](https://raw.githubusercontent.com/revanshriyan/chandra-align/main/docs/images/pipeline-cascade.svg)

- **LightGlue/ALIKED (GPU primary)** — the validated deep-learning path: synthetic ACCEPTED (Sub-Pixel Precision) at inlier RMSE 0.3695 px / 43 inliers; competitive real-pair results at inlier RMSE 1.7961 px (OHRC) and 1.5558 px (TMC-2).
- **RIFT2 (non-functional, under investigation; not the primary matcher):** RIFT2 (vendored NumPy, CPU) returned 0 to 1 correspondences per pair on the issue #1 comparison machine, with no valid fits. Vendored but disabled unless `CHANDRA_ENABLE_RIFT2=1`. See [`results/table_issue01_gpu_validation.csv`](results/table_issue01_gpu_validation.csv).
- **SIFT + Brute-Force RANSAC (CPU fallback)** — estimates the transform as the validated CPU fallback.
- **Sub-pixel refinement** attempts NCC/parabolic refinement of the RANSAC estimate. If too few correspondences survive refinement, the pipeline retains the RANSAC model rather than promoting an unsupported refined fit.
- **Opt-in matcher arms (default-off):** MatchAnything (`CHANDRA_MATCHANYTHING=1`), WildMatch (`CHANDRA_WILDMATCH=1`), AnyMatch (`CHANDRA_ANYMATCH=1`), XoFTR (`CHANDRA_XOFTR=1`) and the phase-congruency front-end (`CHANDRA_PC=1`) run ahead of the cascade above only when their flag is set; with no flag set, the cascade is exactly as listed.

### 2. Fail-closed spatial validation gate

![Fail-closed gate: ACCEPTED, COARSE ALIGNMENT, and REJECTED tiers with exact thresholds](https://raw.githubusercontent.com/revanshriyan/chandra-align/main/docs/images/pipeline-gates.svg)

| Tier | Inlier RMSE | Inliers | Entropy | Quadrants | Meaning |
| --- | --- | --- | --- | --- | --- |
| **ACCEPTED** (sub-pixel) | ≤ 0.50 px | ≥ 8 | ≥ 0.75 | ≥ 3/4 | trusted registration |
| **COARSE ALIGNMENT** | ≤ 2.50 px | ≥ 8 | ≥ 0.50 | ≥ 2/4 | regional fit advisory, not sub-pixel |
| **REJECTED** | anything else | — | — | — | telemetry masked, reasons reported |

Gate verdict names — the display string and status code are exactly what the code emits (`chandra_align/metrics/quadrant.py::validate_registration_gate`):

| Display string | Status code | Meaning |
| --- | --- | --- |
| ACCEPTED (Sub-Pixel Precision) | `SUCCESS_SUBPIXEL` | passed the sub-pixel gate |
| COARSE ALIGNMENT (Regional Fit Advisory) | `COARSE_ADVISORY` | regional fit advisory, not sub-pixel |
| REJECTED (reasons listed) | `DEGENERATE_FAILURE` | no valid registration; transform telemetry masked |

Non-finite metrics fail closed; rejected fits do not expose transform telemetry, and an invalid inlier RMSE is reported as unavailable rather than a misleading zero. Gate verdicts are final — confidence scores never override them.

An opt-in post-fit stage (`CHANDRA_DEFORM_FIELD=1`, default off) can fit a regularized deformation field on the affine inliers and re-run this same gate on field-corrected residuals — see the deformation-field benchmark section below. The gate thresholds themselves never change.

## High-Resolution Handling

PDS4 `.IMG` products require their associated XML labels so the raster layout and metadata can be interpreted safely. Source products are processed through bounded previews/crops rather than loading every full-resolution raster into the matching pipeline. The verified OHRC run used a 4096-pixel cap; the measured TMC-2 result used browse-aligned 2048 × 2048 crops. A larger 4096 × 4000 TMC-2 crop exceeded approximately 5.8 GB of working memory in the tested CPU environment and was stopped. These measurements do not establish that arbitrary full-frame inputs fit within a fixed memory budget.

## Benchmark Status

The matcher measurements below are recorded in [`results/table_issue01_gpu_validation.csv`](results/table_issue01_gpu_validation.csv). RIFT2 returned 0, 1, and 0 correspondences for `synthetic_gentle`, `OHRC_pair`, and `TMC2_fore_nadir`, respectively; each RIFT2 row is rejected.

| Pair | Matcher | Correspondences | Inlier RMSE (px) | Inliers | Entropy | Quadrants | Gate | Runtime (s) | Backend | Hardware |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| synthetic_gentle | RIFT2 | 0 | N/A | 0 | 0.0000 | 0/4 (0,0,0,0) | REJECTED: Only 0 correspondences | 22.6662 | CPU | CPU (model not recorded); machine has an RTX 5070 |
| synthetic_gentle | LightGlue_ALIKED | 1276 | 0.3695 | 43 | 1.9988 | 4/4 (11,10,11,11) | ACCEPTED (Sub-Pixel Precision) | 17.7092 | GPU | NVIDIA RTX 5070 |
| synthetic_gentle | SIFT_RANSAC | 150 | 0.3980 | 54 | 1.9513 | 4/4 (17,8,15,14) | ACCEPTED (Sub-Pixel Precision) | 2.0875 | CPU | CPU (model not recorded); machine has an RTX 5070 |
| OHRC_pair | RIFT2 | 1 | N/A | 0 | 0.0000 | 0/4 (0,0,0,0) | REJECTED: Only 1 correspondences | 42.0472 | CPU | CPU (model not recorded); machine has an RTX 5070 |
| OHRC_pair | LightGlue_ALIKED | 553 | 1.7961 | 34 | 1.9367 | 4/4 (6,6,11,11) | COARSE ALIGNMENT (Regional Fit Advisory) | 2.2815 | GPU | NVIDIA RTX 5070 |
| OHRC_pair | SIFT_RANSAC | 104 | 1.5425 | 9 | 0.9864 | 3/4 (7,0,1,1) | COARSE ALIGNMENT (Regional Fit Advisory) | 4.3942 | CPU | CPU (model not recorded); machine has an RTX 5070 |
| TMC2_fore_nadir | RIFT2 | 0 | N/A | 0 | 0.0000 | 0/4 (0,0,0,0) | REJECTED: Only 0 correspondences | 78.0995 | CPU | CPU (model not recorded); machine has an RTX 5070 |
| TMC2_fore_nadir | LightGlue_ALIKED | 1537 | 1.5558 | 13 | 1.6692 | 4/4 (2,3,1,7) | COARSE ALIGNMENT (Regional Fit Advisory) | 3.1617 | GPU | NVIDIA RTX 5070 |
| TMC2_fore_nadir | SIFT_RANSAC | 106 | 2.2259 | 24 | 1.7296 | 4/4 (4,6,2,12) | COARSE ALIGNMENT (Regional Fit Advisory) | 7.1537 | CPU | CPU (model not recorded); machine has an RTX 5070 |

Backend and hardware for this table: the issue #1 comparison ran locally on an NVIDIA RTX 5070 (PyTorch 2.11.0+cu128; [`docs/judge-qa-compute.md`](docs/judge-qa-compute.md)). LightGlue/ALIKED ran on the GPU; SIFT and the vendored NumPy RIFT2 ran on the CPU of the same machine (CPU model not recorded).

The independent ground-truth protocol is documented in [`docs/ground-truth-protocol.md`](docs/ground-truth-protocol.md), and its split-held-out versus independent-RMSE comparison is recorded in [`results/table_issue02_ground_truth.csv`](results/table_issue02_ground_truth.csv). The independent landmarks landed as ChandraBench v0.1: independent GT RMSE (OHRC 5.6510 px, TMC-2 27.3519 px in `table_canonical.json`) sits far above split-held-out RMSE (OHRC 1.5375 px, TMC-2 1.7950 px) — the gap between inlier self-consistency and true transform error remains an open, tracked limitation, not a closed one.

Backend: not recorded · Hardware: not recorded.

The Issue #3 crop benchmark recorded 12 browse-mapped south-polar pairs (six OHRC and six TMC-2) in [`results/table_issue03_benchmark.csv`](results/table_issue03_benchmark.csv): LightGlue/ALIKED produced 7 COARSE_ADVISORY and 5 DEGENERATE_FAILURE verdicts; SIFT/RANSAC produced 8 COARSE_ADVISORY and 4 DEGENERATE_FAILURE verdicts. No row was ACCEPTED; `gt_rmse` remains empty pending independent landmarks, and these windows do not cover mare terrain.

Backend: not recorded · Hardware: not recorded.

Issue #4 measured three-level coarse-to-fine matching on the same 12 pairs: 5 matcher-pair verdicts improved, 16 were unchanged, and 3 regressed, so it remains opt-in; paired inlier RMSE and runtime measurements are in [`results/table_issue04_coarse2fine.csv`](results/table_issue04_coarse2fine.csv). A native-resolution 8192×8192 OHRC tiled run used 2,482 MiB peak RSS versus 5,123 MiB for the eager resized run (51.6% lower) and returned REJECTED; details are in [`results/issue04_tiled_memory.json`](results/issue04_tiled_memory.json). Ground-truth RMSE is unavailable for this demonstration.

Backend: not recorded · Hardware: not recorded.

### Cross-modal breakthrough: first IIRS↔TMC-2 registration

SIFT-family matchers failed on every IIRS↔TMC-2 attempt (1–25 Lowe matches, none surviving the gate; see [`docs/phase9-report.md`](docs/phase9-report.md)). A detector-free dense matcher (LoFTR, `chandra_align/xmodal/loftr_arm.py`) broke through on IIRS vs common-GSD TMC-2: **2,432 correspondences → 997 inliers after sub-pixel refinement → inlier RMSE 1.35 px → COARSE_ADVISORY**, all four quadrants active (VNIR composite of the 256-band cube; single-band best was 1.37 px / 748). Full account with caveats in [`docs/phase9-loftr-addendum.md`](docs/phase9-loftr-addendum.md). COARSE_ADVISORY, not sub-pixel — reported as measured.

| Input | Matcher | Correspondences | Inlier RMSE (px) | Inliers | Quadrants | Gate |
| --- | --- | --- | --- | --- | --- | --- |
| IIRS 2852nm ↔ TMC-2 | SIFT_RANSAC | 25 | N/A | 0 | 0/4 | DEGENERATE_FAILURE |
| IIRS 2852nm ↔ TMC-2 | LoFTR_outdoor | 2242 | 1.83 | 639 | 4/4 | COARSE_ADVISORY |
| IIRS 2852nm ↔ TMC-2 | LoFTR_outdoor + LK refine | 1881 | 1.37 | 748 | 4/4 | COARSE_ADVISORY |
| IIRS VNIR composite ↔ TMC-2 | LoFTR_outdoor + LK refine | 2432 | 1.35 | 997 | 4/4 | COARSE_ADVISORY |
| IIRS 2852nm ↔ TMC-2 | MINIMA_LoFTR + LK refine | 173 | 1.74 | 52 | 3/4 | COARSE_ADVISORY |

All cross-modal runs behind the frozen gate; full per-arm account in [`docs/phase9-report.md`](docs/phase9-report.md) and [`docs/phase9-loftr-addendum.md`](docs/phase9-loftr-addendum.md).

Backend: CPU (torch 2.14.1+cpu, per the addendum) · Hardware: not recorded.

**XoFTR update (2026-10-15):** XoFTR-640 (opt-in `CHANDRA_XOFTR=1`) scores **0.98 px / 23 inliers on W5** (RANSAC inlier RMSE, fit — not held-out), beating AnyMatch's 1.33 px on the same refined window (fair re-run). Breadth: 16/19 pairs COARSE_ADVISORY, 11/19 outright wins, 4 DEGENERATE_FAILURE→COARSE_ADVISORY conversions. Full trial with caveats in [`docs/xoftr-trial-2026-10-15.md`](docs/xoftr-trial-2026-10-15.md); raw results in [`results/table_xoftr_w5.csv`](results/table_xoftr_w5.csv) and [`results/table_xoftr_breadth.csv`](results/table_xoftr_breadth.csv). Still COARSE_ADVISORY, not sub-pixel.

Backend: not recorded · Hardware: not recorded.

### Scale-ratio robustness (synthetic exact truth)

[`scripts/run_scale_ratio.py`](scripts/run_scale_ratio.py) measures CPU/SIFT against **exact ground truth** at downsampling ratios 1:1 through 16:1 (deterministic seed 7; full data in [`results/table_scale_ratio_sift_pilot.csv`](results/table_scale_ratio_sift_pilot.csv)):

| Ratio | Inliers | Truth RMSE (px) | Inlier RMSE (px) | Gate |
| --- | --- | --- | --- | --- |
| 1:1 | 1711 | 0.0097 | 0.2593 | SUCCESS_SUBPIXEL |
| 2:1 | 1372 | 0.0083 | 0.3580 | SUCCESS_SUBPIXEL |
| 4:1 | 993 | 0.0097 | 0.5276 | COARSE_ADVISORY |
| 8:1 | 701 | 0.0502 | 0.9416 | COARSE_ADVISORY |
| 16:1 | 143 | 0.8868 | 1.5871 | COARSE_ADVISORY |

Backend: CPU · Hardware: not recorded.

Graceful degradation to 16:1 — no catastrophic failure anywhere. The honest finding is at 4:1: the recovered transform was truth-perfect (0.0097 px) while inlier RMSE (0.5276 px) alone tripped ACCEPTED → COARSE ALIGNMENT. **The gate is conservative, not the matcher** — in this synthetic test, inlier self-consistency RMSE overestimates true transform error, so here the reported inlier RMSE is a pessimistic upper bound. That does not carry over to real pairs: against independent landmarks the true error is larger than inlier RMSE (OHRC 5.6510 px, TMC-2 27.3519 px), so on real pairs inlier RMSE must not be read as an upper bound. This is why absolute-truth adjudication (below) is the path to a legitimate real-data claim.

### Sun-angle envelope & phase-congruency breakthrough

A systematic illumination sweep on the LROC DTM (Vikram site) maps the SIFT operational envelope — first as a pilot ([`docs/sunangle-envelope-2026-10-07.md`](docs/sunangle-envelope-2026-10-07.md)), then as a full deterministic 96-case grid ([`docs/sunangle-full-sweep-2026-10-08.md`](docs/sunangle-full-sweep-2026-10-08.md), [`results/table_sunangle_full.csv`](results/table_sunangle_full.csv)). The full sweep revises the pilot's harsher numbers: a 15° azimuth difference is COARSE_ADVISORY at all elevations (truth RMSE 0.03–1.40 px, transforms verified correct); at el=30° the true failure boundary lies between 90° and 105°; at el=50°/70° matching never fails across the full 180°. Grazing sun is the real killer — at el=10°, azimuth differences ≥30° go fully DEGENERATE_FAILURE (long shadows). Envelope width is render-setup-sensitive (the pilot's render script was never committed), so quote the setup with the number. Either way, our real OHRC pairs sit comfortably inside every measured envelope — the two products are 2 hours apart, a ~7–8° azimuth change.

Backend: CPU (full sweep, per [`docs/sunangle-full-sweep-2026-10-08.md`](docs/sunangle-full-sweep-2026-10-08.md)) · Hardware: not recorded.

The repo's phase-congruency front-end (`chandra_align/xmodal/phase_congruency.py`) cracked the azimuth limit ([`docs/pc-azimuth-breakthrough-2026-10-07.md`](docs/pc-azimuth-breakthrough-2026-10-07.md)): PC+SIFT vs raw SIFT at 10° azimuth difference went from 11 to **588 inliers (53×)** and 1.01→0.65 px; at 30° it recovered from DEGENERATE_FAILURE to 18 inliers / 0.90 px COARSE_ADVISORY; even at 180° (opposite sun) it found 85 inliers / 0.91 px — all transforms verified near-identity, not false positives. 90° remains a failure mode.

The honest caveat ([`docs/pc-realpair-honest-result-2026-10-07.md`](docs/pc-realpair-honest-result-2026-10-07.md)): on the same-day real OHRC pair PC+SIFT was **worse** (1.70 vs 1.61 px, 7× fewer inliers) — PC discards the intensity texture SIFT needs when illumination is already similar. PC is a specialist tool for cross-illumination pairs (>10° azimuth difference), not a general improvement. Gate thresholds are unchanged for both paths.

Backend: not recorded · Hardware: not recorded.

### Absolute-truth adjudication against LROC orthophotos (re-derived, corrected geometry)

Independent absolute-truth adjudication against LROC orthophotos of the Vikram site (VIKRAMSITE1, 3 m and 1 m grids) instead of OHRC↔OHRC — both runs re-derived after finding two errors (wrong 1 m file offset, 0.25 m/px assumption vs the PDS4 label's 0.26 m/px). **1 m**: 96/164 inliers, 1.41 m inlier residual, scale 1.0156, rotation +2.12° ([`docs/lroc-adjudication-1m-result-2026-10-08.md`](docs/lroc-adjudication-1m-result-2026-10-08.md)). **3 m**: 46/46 inliers, 2.58 m inlier residual, scale 1.0129, rotation +2.68° ([`docs/lroc-adjudication-first-result-2026-10-08.md`](docs/lroc-adjudication-first-result-2026-10-08.md)). With the deform-field stage applied as a library under a strict no-leakage spatial-split protocol ([`docs/lroc-field-adjudication-2026-10-08.md`](docs/lroc-field-adjudication-2026-10-08.md), [`results/table_lroc_field.csv`](results/table_lroc_field.csv)), the independent held-out absolute bounds tighten to **1 m: 1.7–2.3 m** (6.5–8.8 OHRC px; was 2.6–3.5 m) and **3 m: 2.0–2.5 m** (7.7–9.6 OHRC px; was 6.4–7.1 m) — no folding, no overfit, transform agreement qualified as subset-sampling variance in the doc. The old ~6% scale discrepancy is resolved to **1.6% at both resolutions**; the rotation-sign dispute is settled (+2.1–2.7°, three independent methods agree). These are cross-registration residuals against LROC absolute truth — an honest bound, not a calibrated accuracy spec. The sub-pixel real-pair verdict in the next section comes from the deformation-field stage, scored on held-out matches drawn from the same fit population, not on independent landmarks.

Backend: not recorded · Hardware: not recorded.

### Deformation-field stage: the 1.6 px floor was a model limit (opt-in)

**The discovery.** Four matcher families (SIFT, SIFT+ECC, LoFTR, LoFTR+ECC) and two refinement methods all sat on one floor (~1.4–1.7 px) on the real OHRC pair — then the coverage study proved why: one quadrant's matches obey a coherent *different* local transform, so no single partial-affine can represent the frame ([`docs/spatial-coverage-2026-10-08.md`](docs/spatial-coverage-2026-10-08.md)). A regularized thin-plate-spline residual field fit on the affine inliers breaks it: fit on three quadrants, it predicts the held-out fourth at **1.02 px held-out RMSE** where the global affine gives 19.33 px — better than that quadrant's own local fit (1.75 px). Full sweep in [`results/table_deformation.csv`](results/table_deformation.csv), account in [`docs/deformation-field-2026-10-08.md`](docs/deformation-field-2026-10-08.md).

**The stage.** Wired into the pipeline as an opt-in stage (`CHANDRA_DEFORM_FIELD=1`, default off; [`chandra_align/deform_field.py`](chandra_align/deform_field.py), [`docs/deform-field-pipeline-stage-2026-10-08.md`](docs/deform-field-pipeline-stage-2026-10-08.md)): affine → field (λ chosen per-pair by internal held-out) → the *same frozen gates* re-run on field-corrected residuals. Two safety findings shaped the design:

- The keypoint cap that starved the stage is lifted **only when the stage flag is on** ([`docs/matcher-cap-sweep-2026-10-08.md`](docs/matcher-cap-sweep-2026-10-08.md)): quotas 128/256 let the *default* pipeline lock onto catastrophically wrong transforms, so the cap stays at 64 for the default path and flag-off remains bit-identical.
- 12-pair validation with the flag-gated quota ([`docs/quota-12pair-validation-2026-10-08.md`](docs/quota-12pair-validation-2026-10-08.md), [`results/table_quota_12pair.csv`](results/table_quota_12pair.csv)): **zero verdict downgrades, 3 upgrades** — the cratered-rim, massif-slope, and hummocky-relief OHRC windows reach **SUCCESS_SUBPIXEL** at **0.43/0.47/0.34 px** with 17–23 inliers. Re-measured at the correct TMC-2 pixel scale (5.0 m/px; [`docs/tmc2-correct-scale-rerun-2026-10-08.md`](docs/tmc2-correct-scale-rerun-2026-10-08.md), [`results/table_tmc2_correct_scale.csv`](results/table_tmc2_correct_scale.csv)): **two more SUCCESS_SUBPIXEL** — the large-shadowed-crater-walls and isolated-crater TMC-2 windows at **0.46/0.42 px** with 23 inliers each — plus two DEGENERATE_FAILURE→COARSE_ADVISORY upgrades, **zero downgrades**. The stage's smoothing grid now reaches λ=0.01/0.03 ([`docs/ohrc-blocker-levers-2026-10-08.md`](docs/ohrc-blocker-levers-2026-10-08.md)): the mixed-crater low-sun OHRC window crosses to **SUCCESS_SUBPIXEL at 0.45 px / 24 inliers** through the true pipeline, **zero downgrades**, flag-off bit-identical. Six SUCCESS_SUBPIXEL gate verdicts in-pipeline (inlier RMSE of field-corrected residuals on 17 to 24 inliers per window; not independently verified), all reproducible. (A prototype field-guided rescue also takes ohrc_04 to SUCCESS_SUBPIXEL at 0.36 px / 4 quadrants, but it is an experiment, not a wired pipeline stage — documented honestly in the same doc.)

**The export.** When the stage applies, the download carries the field: a dense `cv2.remap` of the fitted field (fixed-point-inverted, fail-closed to affine; [`docs/deform-field-export-2026-10-08.md`](docs/deform-field-export-2026-10-08.md)), independently verified photometrically — patch NCC on a check grid the fit never saw: the field-remapped export wins 76–82% of patches at 2–4× mean NCC vs affine-only.

**Corrections, kept visible.** An earlier tmc2_04 0.47 px SUCCESS_SUBPIXEL did not reproduce in a fresh independent run (harness artifact) and is **withdrawn**, not footnoted. The 0.41 px rich-set number came from a standalone 842-inlier set the true pipeline never produces — documented in [`docs/deform-stage-multipair-2026-10-08.md`](docs/deform-stage-multipair-2026-10-08.md), not hidden.

**Honest caveats, stated plainly.** The stricter spatial-extrapolation score is 1.99–3.88 px and stands as the harder benchmark — the verdicts above are scored on interpolation-like held-out. The LROC field result was applied as a library (the adjudication is a standalone script), not through the deployed pipeline. Three OHRC windows remain COARSE_ADVISORY with known blockers ([`docs/blocker-diagnosis-2026-10-08.md`](docs/blocker-diagnosis-2026-10-08.md)). **Correction, resolved:** the six TMC-2 DEGENERATE_FAILURE verdicts in the 12-pair benchmark were measured with a misconfigured harness (`pixel_scale_m` defaulted to 0.25 instead of 5.0, forcing a ~20× asymmetric downsample of the reference; [`docs/tmc2-matching-collapse-2026-10-08.md`](docs/tmc2-matching-collapse-2026-10-08.md)). Re-run at the correct scale: 2 SUCCESS_SUBPIXEL, 3 COARSE_ADVISORY, 1 DEGENERATE_FAILURE (honest coverage refusal — 22 inliers in one quadrant). The withdrawn numbers are superseded, not footnoted.

Backend: not recorded · Hardware: not recorded.

### Example Results

Backend and hardware: the issue #1 validation runs — see the Benchmark Status table (LightGlue/ALIKED on GPU, SIFT and RIFT2 on CPU, all on an NVIDIA RTX 5070 host).

![LightGlue/ALIKED accepted synthetic-pair checklist](https://raw.githubusercontent.com/revanshriyan/chandra-align/main/docs/images/gate-checklist-accepted.png)

LightGlue/ALIKED on the calibrated synthetic pair — inlier RMSE 0.3695 px, 43 inliers, entropy 1.9988, 4/4 quadrants: ACCEPTED (Sub-Pixel Precision), measured on RTX 5070.

![RIFT2 rejected synthetic-pair checklist](https://raw.githubusercontent.com/revanshriyan/chandra-align/main/docs/images/gate-checklist-rejected.png)

RIFT2 produced 0 correspondences on the same input — rejected with transform telemetry masked (N/A). The failed matcher is reported, not hidden.

![LightGlue/ALIKED calibrated synthetic registration checkerboard with inlier points](https://raw.githubusercontent.com/revanshriyan/chandra-align/main/docs/images/viz-lightglue-synthetic-checkerboard.png)

LightGlue/ALIKED, calibrated synthetic pair — checkerboard blend of the accepted registration (inlier RMSE 0.3695 px, 43 inliers, 4/4 quadrants). Measured on RTX 5070.

![LightGlue/ALIKED OHRC reference raster with 34 inlier points across four quadrants](https://raw.githubusercontent.com/revanshriyan/chandra-align/main/docs/images/viz-lightglue-ohrc-overlay.png)

LightGlue/ALIKED, OHRC pair — 34 inliers across all four quadrants at inlier RMSE 1.7961 px: COARSE ALIGNMENT (Regional Fit Advisory), not sub-pixel. Measured on RTX 5070.

### Trust, robustness & calibration

Backend: not recorded · Hardware: not recorded.

| Layer | Result |
| --- | --- |
| Independent pixel verification (Phase 10) | 100% cells verified on true transforms, ≤18.75% on wrong ones; synthetic triplet loop closure 0 px |
| Robustness battery (Phase 11) | 45 deterministic runs: 34 SUCCESS_SUBPIXEL / 10 COARSE_ADVISORY / 1 DEGENERATE_FAILURE |
| Confident-but-wrong detection (Phase 11) | `sun_flip` case: 0.87 px inlier RMSE yet 485.8 px off truth — inlier self-consistency ≠ correctness |
| Confidence calibration (Phase 12) | Fitted Brier 0.1198 vs 0.2197 heuristic baseline; verdict NOT CALIBRATED — calibration never touches gate decisions |
| Systematic window tiling (Phase 13) | 84 deterministic windows, every one reported: OHRC 41/42 COARSE (median 2158 inliers, 1.49 px among COARSE windows), TMC-2 fore/nadir 21/42 COARSE (median 514 inliers, 1.64 px among COARSE windows) — full table in [`results/table_phase13_windows.csv`](results/table_phase13_windows.csv) |
| Joint pose-graph (Phase 15) | 2D affines solved simultaneously (TRF+Huber, image 0 pinned), gates run first: synthetic loop misclosure 0.36 px → 0.00 px; no real triplet exists so real data ABSTAINs honestly |
| ChandraBench v0.1 (Phase 16) | 40 human-verified landmarks + one-command evaluator + tech note, CC-BY-4.0, DOI [10.5281/zenodo.23217163](https://doi.org/10.5281/zenodo.23217163) — release [chandrabench-v0.1](https://github.com/revanshriyan/chandra-align/releases/tag/chandrabench-v0.1) — an open benchmark for lunar correspondence |
| Spatial uniformity metrics (diagnostic) | `compute_spatial_uniformity_metrics` (`chandra_align/metrics/quadrant.py`): 8×8 grid occupancy + nearest-neighbor spread stats reported alongside quadrant entropy. **Diagnostic only — gate thresholds stay frozen.** |
| ISIS3 comparison (Phase 17) | `coreg` on identical crops: flawless on synthetic control, but **fails open** on real pairs (57 "successful" chips, 4% consensus) where our pipeline fails closed |

Full accounts in [`docs/phase12-report.md`](docs/phase12-report.md) and the phase result files under [`results/`](results/).

### Standout artifacts

- [Failure gallery](docs/failure-gallery.md) — "How Not to Register the Moon": 8 real failures with diagnoses, including the 485.8 px confident-but-wrong case and the incumbent that fails open.
- [Behavioral test matrix](chandra_align/eval/behavioral.py) — 15 CheckList-style capability × test-type cells, all runnable.
- [Trust maps](scripts/trust_map.py) — per-cell verification heatmaps: 36/36 on true transforms, 0/36 on wrong ones.
- [Frozen evidence](results/EVIDENCE_SHA256.txt) — sha256 manifest of every results file.

## Run Locally

Requires Python 3.12 or newer. Full-resolution Chandrayaan-2 source imagery is not included; obtain products and their labels through ISRO PRADAN.

```powershell
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements-cpu.txt
python app.py
```

### Docker

**Canonical entry point:** the CPU Docker image — this is what CI builds, tests, and verifies on every push:

```bash
docker build -f Dockerfile.cpu -t chandra-align:cpu .
docker run --rm chandra-align:cpu --help
```

- **CPU slim (canonical):** `Dockerfile.cpu` — no CUDA, runs the SIFT/RANSAC CPU path and the batch CLI (`python -m chandra_align.cli.batch`).
- **GPU (full pipeline):** `docker build -t chandra-align .` — CUDA 12.1, runs the full pipeline including the Gradio app.

## Private Demo Access

The interactive deployment is private and available for evaluation/review by request. No Space address, demo endpoint, access token, or credential is published in this repository.

## Canonical Results Table

All headline numbers in this README are generated from raw run outputs by [`scripts/generate_results_table.py`](scripts/generate_results_table.py) — the single source of truth (issue #8). The machine-readable sidecar is [`results/table_canonical.json`](results/table_canonical.json); `tests/test_canonical_table.py` fails loudly if any number drifts from its source. Quote only from this table.

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
