# CHANDRA-ALIGN planetary ingestion and pipeline verification

**Date:** 2026-09-29

**Runtime:** local Windows environment, `.venv` Python 3.14, CPU matcher/fallback
**Scope:** synthetic calibration pairs, a real LROC NAC pair, and full-resolution Chandrayaan-2 OHRC/TMC-2 products.

## Summary

The repaired local environment imports the app and executes the focused engine and app test suites. The synthetic acceptance pair passes the sub-pixel quality gate after shadow suppression was made opt-in. PDS4 OHRC and TMC-2 full-resolution products are readable with their XML product labels and produce registration results on tested crops. Those real full-resolution pair results are **regional/coarse advisories**, not full `REGISTRATION ACCEPTED`; the measured residual and spatial spread do not meet the strict sub-pixel gate. This is reported as measured, rather than treating a coarse fit as full acceptance.

| Pair | Resolution / crop | Result | Inliers / ratio | RMSE (px) | Entropy | Active quadrants | Runtime |
|---|---:|---|---:|---:|---:|---:|---:|
| Synthetic gentle calibrated | 900×900 | Accepted, sub-pixel | 62 / 43.26% | 0.3840 | 1.9532 | 4 (16, 10, 21, 15) | 24.02 s |
| Synthetic strong calibrated | 900×900 | Accepted, sub-pixel | 61 / 43.26% | 0.4141 | 1.9343 | 4 (13, 19, 20, 9) | 29.01 s |
| LROC NAC real stereo pair | 4096-pixel cap from 5000×5000 sources | Rejected: spatially degenerate | 8 / 66.67% (12 matches) | N/A (degenerate fit; placeholder 0.0 not valid) | 0.0000 | 1 (0, 0, 8, 0) | 167.12 s |
| Chandrayaan-2 OHRC pair | Full-res PDS4 inputs, 4096-pixel cap | Coarse alignment / regional advisory | 9 / 8.65% | 1.5425 | 0.9864 | 3 (7, 0, 1, 1) | 55.04 s |
| Chandrayaan-2 TMC-2 fore/nadir | 2048×2048 local crops from full-res PDS4 | Coarse alignment / regional advisory | 9 / 8.33% | 1.3312 | 0.9911 | 2 (4, 0, 0, 5) | 102.05 s |

## Ingestion and coverage

- The supplied lite QA bundle and full-resolution PDS4 products were used. The full-res TMC-2 fore/nadir files are about 1.52 GB each; OHRC products are about 1.12 GB each. XML product labels are stored beside the detached PDS4 `.img` files. The loader uses the labels to resolve raster layout/metadata; bare PDS4 binary files do not carry enough metadata for a generic raster driver to infer their layout safely.
- PDS4 TMC-2 and OHRC ingestion was checked with Rasterio and the application loader. Decoded pixels were finite, returned in the uint8 engine format, and resized under the configured image cap.
- A 4096×4000 TMC fore/nadir region was too costly for this CPU-only run (about 5.8 GB working set and more than 214 CPU seconds); it was stopped. The recorded TMC pipeline run therefore uses a browse-aligned 2048×2048 pair of local windows. Its source windows were fore lines 93329–95376 / samples 362–2409 and nadir lines 101612–103659 / samples 417–2464 (1-based extents).
- Running the two TMC products at the same raw array indices yielded no matches because the fore/nadir sensors have a substantial acquisition footprint offset. The local crop was positioned using matches from their 19,000×400 browse quicklooks.

## Matcher regression finding

The regression was caused by Otsu shadow suppression being enabled by default on lunar scenes. Its masks discarded useful dark-surface texture before matching. On the provided gentle synthetic pair, enabling that mask produced only 7 inliers (RMSE 0.2415 px; four active quadrants) and failed the minimum-inlier gate. Disabling it produced 62 inliers, RMSE 0.3840 px, entropy 1.9532, and support in all four quadrants; the registration passed. The default is now opt-in, while explicit shadow masking remains available to the user. The companion regression test exercises the default behavior.

## Full-resolution real-pair interpretation

- **OHRC:** the paired orbital images yield a regional advisory with 9 inliers. Residual RMSE (1.5425 px) exceeds the sub-pixel limit; entropy is above the configured minimum. The gate and report avoid calling this full acceptance.
- **TMC-2:** the browse-derived 2048-pixel crop produces 9 inliers, but only two quadrants are active and RMSE is 1.3312 px. It is correctly labeled coarse advisory and does not meet full acceptance.
- **LROC NAC:** the 4096-capped real-pair run returned 8 inliers from 12 matches, all in one quadrant. The gate rejects it for zero measured spatial entropy and only one active quadrant. The metrics object reports RMSE `0.0`, but the telemetry correctly marks sub-pixel RMSE as `N/A`; with this degenerate support, `0.0` is a sentinel and must not be read as a valid zero-residual solution. The imagery comes from the NASA Ames Stereo Pipeline LROC NAC tutorial example, sourced from NASA/PDS LROC products.

Reported transform telemetry for the OHRC regional advisory was ΔX −47.5248 px, ΔY −58.3440 px, rotation −0.9596°, and scale 0.99722871. The LROC fit was rejected before a usable global transform was accepted. For TMC-2, the browse quicklook matching step estimated a coarse browse-space affine transform with approximately ΔX −25.80 px / ΔY −160.74 px at its 3800×400 comparison resolution; that is only used to locate the full-resolution windows and is not a georeferenced full-resolution transform.

## Automated checks

Command: `.venv\Scripts\python.exe -m py_compile app.py chandra_align\metrics\quadrant.py`
Result: **PASS**.

Command: `.venv\Scripts\python.exe -m pytest tests\test_app.py tests\test_feature_regression.py -q`
Result: **15 passed** in 87.28 s. Warnings were non-fatal: optional `pyfftw` is absent (RIFT falls back to slower FFTPack), Matplotlib tight-layout warning, existing UTC timestamp deprecation, and numeric warnings exercised by invalid-input tests. Pytest also warned that its cache path collided with an existing file/path; test assertions still passed.

Command: `.venv\Scripts\python.exe -m pytest -q -p no:cacheprovider --basetemp=.pytest-tmp`
Result: **220 passed, 3 skipped** in 146.18 s after resolving a missing `LOW_TRUST` enum referenced by dossier export. The workspace-local pytest temp directory avoids the Windows temp-directory permission errors seen on the first broad-suite attempt. Remaining warnings are non-fatal dependency deprecations, expected numeric/guard warnings from edge-case tests, and Pytest return-value warnings from the standalone readiness script.

The tests cover raster normalization, 14-output callback arity and rejection handling, synthetic registration, visualization functions, export package generation, gate diagnostics, the calibrated sub-pixel regression, and end-to-end pipeline edges. The readiness script now checks the current four-example native Gradio configuration instead of a stale three-example/pitch-demo label.

## Attribution

- LROC NAC data: NASA Planetary Data System / Lunar Reconnaissance Orbiter Camera (LROC), via the NASA Ames Stereo Pipeline tutorial example.
- Chandrayaan-2 data: “We acknowledge the use of data from the Chandrayaan-II, second lunar mission of the Indian Space Research Organisation (ISRO), archived at the Indian Space Science Data Centre (ISSDC).”

## Limitations

These runs establish local ingestion and pipeline behavior for the listed inputs/crops. They do not establish a 100% crash-free guarantee for every possible upload or every runtime. The real OHRC/TMC fits did not satisfy the sub-pixel acceptance thresholds in these tested regions, and the TMC full-frame registration was not completed due to CPU/memory cost. The metric readout must not be represented to reviewers as a scientific validation of georeferencing without independent ground control or truth data.
