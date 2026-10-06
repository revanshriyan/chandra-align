# CHANDRA-ALIGN — Honest Benchmark Card

*Generated 2026-10-06T21:39:59+00:00 UTC from `results/table_canonical_v1.csv` by*
`scripts/gen_model_card.py`. *Numbers are measured, not claimed.*

## What this pipeline is

A lunar image registration pipeline for Chandrayaan-2 OHRC, TMC-2, and IIRS
products. Matcher cascade: LightGlue/ALIKED (GPU, validated path) → SIFT +
Brute-Force RANSAC (CPU fallback). A fail-closed spatial gate classifies every
result as ACCEPT (sub-pixel), COARSE ADVISORY (regional fit), or REJECT /
DEGENERATE_FAILURE. Rejected fits mask transform telemetry.

## Measured results (canonical table v1)

| Benchmark | Pair | Matcher | Corr | Inliers | RMSE (px) | Held-out (px) | GT RMSE (px) | Verdict |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| gpu_validation | synthetic_gentle | RIFT2 | 0 | 0 | — | — | — | REJECTED |
| gpu_validation | synthetic_gentle | LightGlue_ALIKED | 1276 | 43 | 0.3694 | — | — | ACCEPTED |
| gpu_validation | synthetic_gentle | SIFT_RANSAC | 150 | 54 | 0.3979 | — | — | ACCEPTED |
| gpu_validation | OHRC_pair | RIFT2 | 1 | 0 | — | — | — | REJECTED |
| gpu_validation | OHRC_pair | LightGlue_ALIKED | 553 | 34 | 1.7960 | — | — | COARSE ALIGNMENT |
| gpu_validation | OHRC_pair | SIFT_RANSAC | 104 | 9 | 1.5425 | — | — | COARSE ALIGNMENT |
| gpu_validation | TMC2_fore_nadir | RIFT2 | 0 | 0 | — | — | — | REJECTED |
| gpu_validation | TMC2_fore_nadir | LightGlue_ALIKED | 1537 | 13 | 1.5558 | — | — | COARSE ALIGNMENT |
| gpu_validation | TMC2_fore_nadir | SIFT_RANSAC | 106 | 24 | 2.2258 | — | — | COARSE ALIGNMENT |
| ground_truth | synthetic_gentle | RIFT2 |  |  | — | — | — | UNMEASURED_NO_FIT |
| ground_truth | synthetic_gentle | LightGlue_ALIKED |  |  | — | 0.4231 | 0.1191 | YES |
| ground_truth | synthetic_gentle | SIFT_RANSAC |  |  | — | 0.4735 | 0.0533 | YES |
| ground_truth | OHRC_pair | RIFT2 |  |  | — | — | — | UNMEASURED_NO_FIT |
| ground_truth | TMC2_fore_nadir | RIFT2 |  |  | — | — | — | UNMEASURED_NO_FIT |
| ground_truth | OHRC_pair | LightGlue_ALIKED |  |  | — | 1.5374 | 5.6510 | NO |
| ground_truth | OHRC_pair | SIFT_RANSAC |  |  | — | 1.9797 | 5.4965 | NO |
| ground_truth | TMC2_fore_nadir | LightGlue_ALIKED (SIFT fallback used) |  |  | — | 1.7950 | 27.351 | NO |
| ground_truth | TMC2_fore_nadir | SIFT_RANSAC |  |  | — | 1.7950 | 27.351 | NO |
| iirs_first_contact | iirs2_fore_tmc2 | LightGlue/ALIKED | 19 | 4 | 1.1748 | — | — | DEGENERATE_FAILURE |
| iirs_first_contact | iirs2_fore_tmc2 | SIFT + Brute-Force (RANSAC fallback) | 25 | 10 | 0.0 | 0.0 | — | DEGENERATE_FAILURE |

### 12-pair south-polar crop benchmark

- **LightGlue_ALIKED**: COARSE_ADVISORY: 7, DEGENERATE_FAILURE: 5
- **SIFT_RANSAC**: COARSE_ADVISORY: 8, DEGENERATE_FAILURE: 4

## What the numbers mean — and don't

- **Synthetic pairs ACCEPT sub-pixel** (0.37–0.40 px). The pipeline works when
  correspondences exist.
- **Real OHRC/TMC-2 pairs are COARSE at best** (1.5–2.2 px in-sample; held-out
  1.5–2.6 px). Independent human landmarks disagree with the fitted transforms
  (GT RMSE 5.7–27.4 px): the pipeline finds *a* consistent fit, not necessarily
  *the* true one. Treat real-pair outputs as advisories, not measurements.
- **IIRS↔TMC-2 cross-modal: no successful registration yet.** Both matchers hit
  DEGENERATE_FAILURE on the verified-overlap crop.
- **RIFT2 is non-functional** in this codebase (0–1 correspondences per pair).
  It is vendored but disabled by default.
- Held-out RMSE is computed on 3–25 check points depending on the pair; small
  held-out sets are noisy — see the per-table `n_heldout_points` columns.

## Provenance

- GPU validation: NVIDIA RTX 5070.
- Full per-run tables: `results/table_issue01_gpu_validation.csv`,
  `results/table_issue02_ground_truth.csv`, `results/table_issue03_benchmark.csv`,
  `results/table_issue05_baselines.csv`, `results/table_issue12_iirs.csv`.
- Canonical rollup: `results/table_canonical_v1.csv` (+ `.meta.json`).
- README numbers are machine-checked by `scripts/check_readme_numbers.py`.
