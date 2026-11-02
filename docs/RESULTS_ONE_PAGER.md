# Results one-pager (generated)

Produced by `scripts/generate_results_one_pager.py` from the committed
evidence files named on every line below. Do not hand-edit: the drift test
(`tests/test_results_one_pager_drift.py`) regenerates this file and fails on
any difference. Real pairs first; synthetic rows are sanity checks.

## Independent landmark scoring (ChandraBench v0.1 landmarks)

- OHRC_pair / RIFT2: no RMSE on this row (verdict-only); verdict_agrees: UNMEASURED_NO_FIT; source: results/table_canonical_v1.csv (ground_truth); hardware: not recorded
- TMC2_fore_nadir / RIFT2: no RMSE on this row (verdict-only); verdict_agrees: UNMEASURED_NO_FIT; source: results/table_canonical_v1.csv (ground_truth); hardware: not recorded
- OHRC_pair / LightGlue_ALIKED: held-out RMSE 1.5374719216 px; independent landmark RMSE 5.651034 px; verdict_agrees: NO; source: results/table_canonical_v1.csv (ground_truth); hardware: not recorded
- OHRC_pair / SIFT_RANSAC: held-out RMSE 1.9797391828 px; independent landmark RMSE 5.496597 px; verdict_agrees: NO; source: results/table_canonical_v1.csv (ground_truth); hardware: not recorded
- TMC2_fore_nadir / LightGlue_ALIKED (SIFT fallback used): held-out RMSE 1.7950233624 px; independent landmark RMSE 27.351923 px; verdict_agrees: NO; source: results/table_canonical_v1.csv (ground_truth); hardware: not recorded
- TMC2_fore_nadir / SIFT_RANSAC: held-out RMSE 1.7950233624 px; independent landmark RMSE 27.351923 px; verdict_agrees: NO; source: results/table_canonical_v1.csv (ground_truth); hardware: not recorded

Human-verified calibration landmarks (strip-guard calibration set):

- ohrc_01: independent landmark RMSE 1.8869 px over 21 verified landmarks; source: results/table_strip_guard_calibration_2026-10-24.csv; hardware: not recorded
- ohrc_04: independent landmark RMSE 2.2591 px over 18 verified landmarks; source: results/table_strip_guard_calibration_2026-10-24.csv; hardware: not recorded
- TMC2_fore_nadir_anchor: independent landmark RMSE 28.031705 px over 20 verified landmarks; source: results/table_strip_guard_calibration_2026-10-24.csv; hardware: not recorded
- OHRC_pair_anchor: independent landmark RMSE 5.471325 px over 20 verified landmarks; source: results/table_strip_guard_calibration_2026-10-24.csv; hardware: not recorded

## Held-out verification (non-circular 80/20 splits)

- held-out RMSE across 9 pairs (ohrc_01, ohrc_02, ohrc_03, ohrc_04, ohrc_05, tmc2_01, tmc2_02, tmc2_04, tmc2_06): 0.2459 to 0.3631 px; source: results/table_rescue_noncircular.csv
- admitted fraction of held-out matches: 0.0816 to 0.3225; source: results/table_rescue_noncircular.csv
- unconditional held-out median residual: 1.6629 to 11.3288 px (stated alongside, not hidden); source: results/table_rescue_noncircular.csv
- per-pair rows carry their own inlier (fit) RMSE in the same file; the split-held-out versus independent-landmark gap is an open, tracked limitation, not a closed one.

## Matcher tables, real pairs

Canonical table (issue #1 validation harness):

- OHRC_pair / RIFT2: no RMSE on this row (verdict-only); verdict "REJECTED: Only 1 correspondences"; source: results/table_canonical_v1.csv (gpu_validation); hardware: NVIDIA RTX 5070
- OHRC_pair / LightGlue_ALIKED: inlier (fit) RMSE 1.796071957641206 px; verdict "COARSE ALIGNMENT (Regional Fit Advisory)"; source: results/table_canonical_v1.csv (gpu_validation); hardware: NVIDIA RTX 5070
- OHRC_pair / SIFT_RANSAC: inlier (fit) RMSE 1.5425123316884326 px; verdict "COARSE ALIGNMENT (Regional Fit Advisory)"; source: results/table_canonical_v1.csv (gpu_validation); hardware: NVIDIA RTX 5070
- TMC2_fore_nadir / RIFT2: no RMSE on this row (verdict-only); verdict "REJECTED: Only 0 correspondences"; source: results/table_canonical_v1.csv (gpu_validation); hardware: NVIDIA RTX 5070
- TMC2_fore_nadir / LightGlue_ALIKED: inlier (fit) RMSE 1.5558487006656572 px; verdict "COARSE ALIGNMENT (Regional Fit Advisory)"; source: results/table_canonical_v1.csv (gpu_validation); hardware: NVIDIA RTX 5070
- TMC2_fore_nadir / SIFT_RANSAC: inlier (fit) RMSE 2.2258678123173246 px; verdict "COARSE ALIGNMENT (Regional Fit Advisory)"; source: results/table_canonical_v1.csv (gpu_validation); hardware: NVIDIA RTX 5070

2026-11-02 GPU bench (inlier (fit) RMSE; backend torch 2.11.0+cu128):

- OHRC_pair / RIFT2: inlier (fit) RMSE 2.0185234135945245 px, 48 inliers, gate "COARSE ALIGNMENT (Regional Fit Advisory)"; source: results/table_gpu_bench_2026-11-02.csv; hardware: Tesla T4
- OHRC_pair / LightGlue_ALIKED: inlier (fit) RMSE 1.5381289525290243 px, 34 inliers, gate "COARSE ALIGNMENT (Regional Fit Advisory)"; source: results/table_gpu_bench_2026-11-02.csv; hardware: Tesla T4
- OHRC_pair / SIFT_RANSAC: inlier (fit) RMSE 1.9865294733902066 px, 36 inliers, gate "COARSE ALIGNMENT (Regional Fit Advisory)"; source: results/table_gpu_bench_2026-11-02.csv; hardware: Tesla T4
- TMC2_fore_nadir / RIFT2: inlier (fit) RMSE UNMEASURED px, 6 inliers, gate "REJECTED (Insufficient Inlier Yield (6 < 8) | Degenerate Spatial Cluster (2/4 Active Quadrants))"; source: results/table_gpu_bench_2026-11-02.csv; hardware: Tesla T4
- TMC2_fore_nadir / LightGlue_ALIKED: inlier (fit) RMSE UNMEASURED px, 13 inliers, gate "REJECTED (STRIP_COLLAPSE: inlier spatial support minor_ratio=0.075 major_ratio=0.777 < 0.16 of frame)"; source: results/table_gpu_bench_2026-11-02.csv; hardware: Tesla T4
- TMC2_fore_nadir / SIFT_RANSAC: inlier (fit) RMSE UNMEASURED px, 13 inliers, gate "REJECTED (STRIP_COLLAPSE: inlier spatial support minor_ratio=0.075 major_ratio=0.777 < 0.16 of frame)"; source: results/table_gpu_bench_2026-11-02.csv; hardware: Tesla T4

- 12 south-polar pairs (6 OHRC + 6 TMC-2) / LightGlue_ALIKED: no RMSE on this row (verdict-only); verdict "COARSE_ADVISORY: 7, DEGENERATE_FAILURE: 5"; source: results/table_canonical_v1.csv (crop_benchmark_12pair); hardware: not recorded
- 12 south-polar pairs (6 OHRC + 6 TMC-2) / SIFT_RANSAC: no RMSE on this row (verdict-only); verdict "COARSE_ADVISORY: 8, DEGENERATE_FAILURE: 4"; source: results/table_canonical_v1.csv (crop_benchmark_12pair); hardware: not recorded
- iirs2_fore_tmc2 / LightGlue/ALIKED: inlier (fit) RMSE 1.1748296022415161 px; verdict "DEGENERATE_FAILURE"; source: results/table_canonical_v1.csv (iirs_first_contact); hardware: not recorded
- iirs2_fore_tmc2 / SIFT + Brute-Force (RANSAC fallback): inlier (fit) RMSE 0.0 px; held-out RMSE 0.0 px; verdict "DEGENERATE_FAILURE"; source: results/table_canonical_v1.csv (iirs_first_contact); hardware: not recorded

External baselines (39 canonical rows, per-pair windows):

- COARSE_ADVISORY: 24 rows; source: results/table_canonical_v1.csv (external_baselines)
- DEGENERATE_FAILURE: 9 rows; source: results/table_canonical_v1.csv (external_baselines)
- REJECTED: 3 rows; source: results/table_canonical_v1.csv (external_baselines)

## Sanity checks (synthetic — not real-pair evidence)

- synthetic_gentle / RIFT2: inlier (fit) RMSE 0.13978389984846726 px, 28 inliers, gate "ACCEPTED (Self-Consistent, Sub-Pixel)"; source: results/table_gpu_bench_2026-11-02.csv; hardware: Tesla T4
- synthetic_gentle / LightGlue_ALIKED: inlier (fit) RMSE 0.5060550010010166 px, 37 inliers, gate "COARSE ALIGNMENT (Regional Fit Advisory)"; source: results/table_gpu_bench_2026-11-02.csv; hardware: Tesla T4
- synthetic_gentle / SIFT_RANSAC: inlier (fit) RMSE 0.3556843780045989 px, 35 inliers, gate "ACCEPTED (Self-Consistent, Sub-Pixel)"; source: results/table_gpu_bench_2026-11-02.csv; hardware: Tesla T4
- synthetic_gentle / RIFT2: no RMSE on this row (verdict-only); verdict "REJECTED: Only 0 correspondences"; source: results/table_canonical_v1.csv (gpu_validation); hardware: NVIDIA RTX 5070
- synthetic_gentle / LightGlue_ALIKED: inlier (fit) RMSE 0.3694768305346762 px; verdict "ACCEPTED (Sub-Pixel Precision)"; source: results/table_canonical_v1.csv (gpu_validation); hardware: NVIDIA RTX 5070
- synthetic_gentle / SIFT_RANSAC: inlier (fit) RMSE 0.39797883074596385 px; verdict "ACCEPTED (Sub-Pixel Precision)"; source: results/table_canonical_v1.csv (gpu_validation); hardware: NVIDIA RTX 5070
- synthetic_gentle / RIFT2: no RMSE on this row (verdict-only); verdict_agrees: UNMEASURED_NO_FIT; source: results/table_canonical_v1.csv (ground_truth); hardware: not recorded
- synthetic_gentle / LightGlue_ALIKED: held-out RMSE 0.4231709230 px; independent landmark RMSE 0.119166 px; verdict_agrees: YES; source: results/table_canonical_v1.csv (ground_truth); hardware: not recorded
- synthetic_gentle / SIFT_RANSAC: held-out RMSE 0.4735171094 px; independent landmark RMSE 0.053374 px; verdict_agrees: YES; source: results/table_canonical_v1.csv (ground_truth); hardware: not recorded

## ChandraBench v0.1 (frozen landmark sets)

- ohrc_pair: 20 human-verified landmarks; source: chandrabench/landmarks_ohrc_pair_v0.1.csv
- tmc2_fore_nadir: 20 human-verified landmarks; source: chandrabench/landmarks_tmc2_fore_nadir_v0.1.csv
- both sets are sha256-pinned inside chandrabench/eval.py and guarded fail-closed by its CIRCULARITY_GUARD (run `python chandrabench/eval.py --self-test`)
- landmark licence: CC-BY-4.0 (chandrabench/LICENSE)

## How to read this page

- inlier (fit) RMSE: residuals of the RANSAC-accepted matches on the fit set; self-consistency, not independent truth.
- held-out RMSE: residuals on check points held out of the fit (80/20 non-circular splits where the file says so).
- independent landmark RMSE: residuals against human-verified landmark coordinates that no fit stage saw.
- hardware is quoted from the source file; "hardware: not recorded" is preserved as-is.
- gate thresholds and verdict logic are frozen (chandra_align/metrics/quadrant.py::validate_registration_gate) and are not restated or re-measured here.
