# Review-only ground-truth landmark candidates

These CSVs and chip sheets are **proposals, not ground truth**. Every row is marked `PENDING_HUMAN`; do not compute or publish ground-truth RMSE from these points until a reviewer accepts or corrects each landmark and the correspondence coordinates.

## Inputs and reproducibility

- `OHRC_pair`: the two canonical products `ch2_ohr_nrp_20240425T1209509264_d_img_d18` and `ch2_ohr_nrp_20240425T1406019344_d_img_d18`, loaded with the same `app.load_lunar_raster(..., max_dimension=4096)` call used by `results/run_issue01_gpu_validation.py`.
- `TMC2_fore_nadir`: the canonical fore/nadir products from that runner, with its audited windows: fore `Window(col_off=361, row_off=93328, width=2048, height=2048)` and nadir `Window(col_off=416, row_off=101611, width=2048, height=2048)`. PDS rasters are normalized with the same app loader after the window read.
- Generator: `scripts/propose_ground_truth_candidates.py`.

The generator independently downsamples each pair, finds Harris corners in the reference, searches each corner patch in the source using normalized cross-correlation (NCC), rejects duplicate source peaks, and applies a homography RANSAC consensus filter to propose a reviewable set. It does not call LightGlue/ALIKED or SIFT. NCC score and the coarse-model residual help triage proposals, but repeated lunar texture, shadow changes, resampling, and the coarse fit can still produce false matches. Human visual confirmation is required.

Each chip-sheet cell shows the reference chip on the left and proposed source chip on the right. CSV coordinates are in the canonical full-resolution input coordinate systems (x is column, y is row); `model_error_small_px` is measured on the 640-pixel proposal images, not a ground-truth accuracy estimate.

## External control-data search

The official LROC NAC Chandrayaan-3 pre-landing orthophoto product `NAC_DTM_VIKRAMSITE1_M1443025251_3M` covers 31.31–33.37°E and 68.51–70.07°S, overlapping the OHRC pair's region. Its 116.36 MB GeoTIFF download is linked from the [official product page](https://data.lroc.im-ldi.com/lroc/view_rdr_product/NAC_DTM_VIKRAMSITE1_M1443025251_3M). The product was not fetched in this run: outbound socket access to the NASA PDS mirror is denied in the execution environment. Therefore no control points were derived from it. The TMC-2 crop is from the product's south-polar track; no matching georeferenced LROC NAC/SLDEM control raster for the exact crop was available locally or downloaded in this run. The existing public south-pole mosaic is not evidence of coverage for this crop. See the official [LROC NAC processing guide](https://www.lroc.asu.edu/data/support/downloads/LROC_NAC_Processing_Guide.pdf) for SLDEM's stated coverage limits.

## Human review still needed

For each CSV row, accept or reject the visual correspondence, adjust either coordinate if needed, and record reviewer/date/method. Then independently establish absolute/control coordinates from the LROC orthophoto for OHRC and an appropriate georeferenced product for TMC-2. Until then neither pair has an accepted independent ground-truth set.
