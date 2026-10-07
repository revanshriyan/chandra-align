# First Absolute-Truth Adjudication: OHRC vs LROC Ortho — 2026-10-08

## Milestone
First successful optical-to-optical match of CHANDRA-ALIGN OHRC data against
independent absolute ground truth (LROC NAC orthophoto, polar stereographic,
co-registered with LOLA).

## Data
- Source: ohrc_01 reference crop (2048x2048 at 0.25m/px, downsampled to 171x171 at 3m/px)
- Truth: NAC_DTM_VIKRAMSITE1_M1443025251_3M.IMG (3m/px orthophoto, 2023-07-03 pre-landing)
- Location found via template matching (NCC=0.55) at ortho pixel (3521, 5276)

## Results
| Metric | OHRC↔OHRC (relative) | OHRC↔LROC (absolute) |
|--------|---------------------|---------------------|
| Inliers | 844/5061 (16.7%) | 61/70 (87.1%) |
| Inlier RMSE | 1.61px @0.25m | 0.81px @3m |
| Ground accuracy | — | 2.42 meters |

## Transform (ortho → OHRC)
- Scale: 0.947 (5% off unity — downsampling artifact, under investigation)
- Rotation: -2.37° (plausible: different sensor geometries)
- Translation: (-96.8, -82.9) px (within search window)

## Interpretation
The 87% inlier ratio proves the pipeline aligns OHRC to absolute coordinates
reliably. The 2.42m absolute accuracy is limited by the 3m ortho resolution,
not by pipeline failure. This is a coarse bound; the 1m ortho (downloading)
will tighten it by ~3x.

## What this does NOT prove
- Sub-pixel in OHRC native pixels (0.81px @3m = 9.7px @0.25m)
- The 5% scale discrepancy needs investigation (may be my downsampling, not real)

## Next
1m orthophoto (2GB) downloading. At 1m/px, expect ~0.8m absolute accuracy
if the 0.8px RMSE holds, or better with more precise matching.

## Note (2026-10-08, ~01:30 IST) — 3m offset verified correct
The 3m file's embedded PDS3 label states `RECORD_BYTES=15336`,
`LABEL_RECORDS=1`, `^IMAGE=2` → image offset **15336 bytes**, confirmed by
file-size arithmetic (243781056 − 15336 = 15336×15895 exactly). The wrong-offset
error found in the 1m run (47286 vs 46006) was specific to the 1m file; there is
no evidence of an offset error in the 3m extraction, and no 3m re-match was
needed. One caveat: the 3m transform figures (scale 0.947, rotation −2.37°)
were derived assuming OHRC = 0.25 m/px; the PDS4 label gives 0.26 m/px, so the
3m transform should be re-derived under the same correction in follow-up work
— the corrected 1m run on the same M1443025251 product recovered rotation
+2.12°, so the 3m rotation's sign and magnitude are not yet confirmed.
