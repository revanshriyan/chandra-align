# Q3 trial — SELENE TC morning/evening pair — 2026-11-03

Trial run 2026-11-02. A cross-illumination pair test (Phase 2 T4
candidate): the same terrain imaged by SELENE (Kaguya) Terrain Camera
under near-opposite Sun azimuths. All RMSE values below are in pixels
unless a ground figure is explicitly derived; the primary arm's inlier
figure is descriptive only, for the reason stated with it.

## Provenance

- Code: repo main content at `fea8696d` (consolidated tree), run
  through the app's alignment core, `app._align_core` — the same
  entry point the earlier benches used. One run per arm,
  sequentially, CPU only. No repo files were changed by the run.
- Pair (JAXA DARTS public PDS3 listings, staged 2026-11-02):
  - Reference: `TCO_MAPm04_N12E339N09E342SC` (SELENE TC morning MAP,
    dataset SLN-L-TC-5-MORNING-MAP-V4.0)
  - Source: `TCO_MAPe04_N12E339N09E342SC` (SELENE TC evening MAP,
    dataset SLN-L-TC-5-EVENING-MAP-V4.0)
  - Terrain: Copernicus crater region, lat 9–12N, lon 339–342E. Per
    the dataset definitions, the morning tile is the eastward
    low-Sun mapping cycle and the evening tile the westward low-Sun
    cycle — near-opposite illumination azimuths over the same
    terrain.
- Grid verification: both PDS3 labels were compared field by field
  before any run. The two products share one map grid: LINES =
  LINE_SAMPLES = 12288, Simple Cylindrical, MAP_RESOLUTION
  4096 px/deg, MAP_SCALE 0.0074031617246699 km/px (7.403 m/px),
  MAP_PROJECTION_ROTATION 0, LINE_PROJECTION_OFFSET 49152.0,
  SAMPLE_PROJECTION_OFFSET 86016.0, identical footprint corners
  (lat 9.000244–12.0, lon 339.0–341.999756). Identical pixel windows
  therefore cover identical ground; no cross-product window
  adjustment was needed.
- Byte order: the tiles are MSB_UNSIGNED_INTEGER with DUMMY = 0
  (read big-endian uint16); the staging README's original int16
  little-endian description ("signed reflectance", "fill DN −32768")
  was a byte-swap artifact and is corrected there (2026-11-03).
- Window protocol (pre-registered, followed as written): centre
  2048×2048 of the shared grid, same pixel window in both products
  (row_off = 5120, col_off = 5120). Fill check: DN == 0 fraction in
  the centre window is 0.0% in both products (well under the 5%
  threshold), so the centre window stands and the fallback-window
  rule was not invoked. No protocol deviation, and no second window
  was tried.
- Conversion: 8-bit PNG, percentile stretch at 1–99 over valid
  (DN > 0) pixels. Morning: DN range 1–3636, stretch 1.0–2276.0.
  Evening: DN range 1–6117, stretch 1.0–2890.0.
- Call, exactly as run:

  ```python
  app._align_core(
      ref, sec,
      pixel_scale_m=10.0,
      reference_sensor_name="KAGUYA_TC",
      secondary_sensor_name="KAGUYA_TC",
  )
  ```

  The 10.0 m/px is the pipeline's own KAGUYA_TC table value. Passing
  it for both images makes the common-GSD resample step a no-op —
  correct here because both products already share one grid at
  7.403 m/px. All metrics are in pixels.
- Strip guard: default ON (env unset) in both arms, threshold
  `STRIP_SUPPORT_RATIO_MIN` 0.159.

## Results

| Arm | Matcher that ran | Corr. | Inliers | Inlier (fit) RMSE | Gate RMSE | Entropy | Quadrants | Strip minor | Verdict |
|---|---|---|---|---|---|---|---|---|---|
| Primary (default pipeline) | LightGlue/ALIKED (CPU) | 59 | 4 | 1.0462 px (descriptive only) | — (null on rejected path) | 0.8113 | 2/4 | invalid (n = 4) | DEGENERATE_FAILURE |
| Secondary (`CHANDRA_PC=1`) | PhaseCongruency/SIFT (CPU) | 118 | 68 | 1.9141 px | 2.4313 px (held-out) | 1.8936 | 4/4 | 1.150 (valid) | COARSE_ADVISORY |

Runtimes: primary 51.0 s, secondary 118.1 s.

### Primary arm — default pipeline, exactly as shipped

All opt-in flags unset.

- Gate verdict: **DEGENERATE_FAILURE** — REJECTED (High Residual
  RMSE (sub-pixel limit exceeded: 1.0462 px > 0.50 px) |
  Insufficient Inlier Yield (4 < 8) | Degenerate Spatial Cluster
  (2/4 Active Quadrants)).
- Inlier (fit) RMSE: 1.0462 px — in-sample fallback basis on only
  4 inliers; the gate RMSE is null on the rejected path and held-out
  RMSE is unavailable. Treat the 1.0462 figure as descriptive only;
  the rejection rests on inlier yield and spatial distribution, not
  on that figure.
- Inliers / correspondences: 4 / 59 (inlier ratio 6.8%).
- Quadrant entropy: 0.8113; active quadrants 2/4 (per-quadrant
  counts are not returned on the rejected path).
- Strip support: invalid (insufficient points, n = 4); minor_ratio
  reported as 0.0 with valid = false, threshold 0.159 — no strip
  verdict can be drawn from 4 points.
- Matcher that actually ran: LightGlue/ALIKED, on CPU
  (execution_device = CPU; the cascade's internal fallback triggered
  on low inlier ratio and re-ran LightGlue/ALIKED).

### Secondary arm — phase-congruency front-end (`CHANDRA_PC=1`, opt-in)

Same crops, same everything else; strip guard default ON.

- Gate verdict: **COARSE_ADVISORY** — COARSE ALIGNMENT (Regional
  Fit Advisory).
- Gate RMSE: 2.4313 px (held-out basis). Inlier (fit) RMSE:
  1.9141 px. At the label scale of 7.403 m/px, the held-out gate
  figure corresponds to ≈ 18.0 m on the ground.
- Inliers / correspondences: 68 / 118.
- Quadrant entropy: 1.8936; quadrants 4/4 active (Q1 26, Q2 17,
  Q3 8, Q4 17 inliers).
- Strip support: valid, minor_ratio 1.150, major_ratio 1.064 —
  not collapsed.
- Matcher that actually ran: PhaseCongruency/SIFT, on CPU.

## Independent replication

An independent second-environment replication (a different machine,
GPU environment) re-ran both arms from the same crops:

- The secondary (phase-congruency) arm reproduced **exactly**:
  COARSE_ADVISORY, held-out gate RMSE 2.4313 px digit-identical,
  68 inliers, entropy 1.8936, 4/4 quadrants, strip minor_ratio
  1.1502. The classical path is deterministic across machines and
  backends.
- The primary arm reached the same verdict, DEGENERATE_FAILURE,
  with drift-level differences (58 vs 59 correspondences). The
  conclusion — the default pipeline fails closed on this pair — is
  backend-robust.

## What this trial settles

- **Default pipeline on this cross-illumination pair: clean
  negative.** At the pre-registered centre window the default
  LightGlue/ALIKED path cannot assemble a gateable fit (4 inliers)
  under near-opposite Sun azimuths. No second window was tried; a
  bad verdict is a result, not a cue to go window shopping.
- **The opt-in phase-congruency arm rescues the pair to a coarse
  advisory** (held-out 2.4313 px, 68 inliers, 4/4 quadrants) — the
  regime that arm exists for, consistent with its earlier behaviour
  at large illumination differences. It is a coarse rescue, not a
  sub-pixel result. The arm remains opt-in; nothing here argues for
  changing the default path or any gate threshold.

## Caveats

- Both products are map-projected and co-registered to the same
  Kaguya DTM grid, so the geometric truth is near-identity by
  construction. This trial measures matcher survival under
  illumination reversal on real data; it is not an independent
  geolocation accuracy measurement, and no landmark check was
  performed.
- Illumination is qualitative only: the labels carry no acquisition
  times (START_TIME and STOP_TIME are UNK in both; the product
  creation time is a 2013 processing date, not an acquisition time),
  so no numeric Sun angles are computed. The morning/evening
  characterisation rests on the dataset definitions.
- Backend note: this run's default arm used LightGlue/ALIKED on
  CPU (torch 2.14.1+cpu is importable on the run machine, so the
  default path did not fall through to SIFT). On an install without
  the LightGlue/torch stack the same default path would take its
  SIFT route instead; the primary verdict above is specific to the
  backend named.
