---
title: Chandra-Align
emoji: 🌖
colorFrom: blue
colorTo: indigo
sdk: gradio
sdk_version: 4.44.1
app_file: app.py
pinned: false
---

# chandra_align

Config-driven engine that registers Chandrayaan-2 optical products (OHRC, TMC, IIRS) to
geo-referenced lunar reference maps (LROC NAC/WAC, Kaguya TC) at sub-pixel RMSE
(target) with uniformly distributed tie points and an explicit trust flag.

Problem statement SIH26166 (ISRO): *Multi-modal, Sun angle and scale invariant image
correspondence using Chandrayaan-2 optical images (OHRC, TMC and IIRS).*

## Status: Stage-1 PROTOTYPE — synthetic-data only so far

All metrics in this repo so far come from **synthetic fixtures** with known
ground-truth transforms. **No real lunar pair has been run.** Real-data numbers are
`UNMEASURED` until a real OHRC/TMC pair plus reference map is ingested (see
`data/download_log.csv` — currently header-only). Sub-pixel accuracy is a **target**
until measured on real data.

## Setup

```bash
python -m venv .venv
.venv/Scripts/pip install -r requirements.txt   # Windows; .venv/bin on POSIX
pytest                                          # synthetic fixtures only
uvicorn api.main:app --port 8000                # demo service
```

## Layout

```
config/           modality profiles: ohrc.yaml, tmc.yaml, iirs.yaml (all magic numbers live here)
chandra_align/    the engine
  ingest/         PDS4/QUB readers, CRS unification + projection-trap guard, tiling
  photometry/     Lommel-Seeliger / lunar-Lambert normalisation, shadow/saturation masking
  matcher/        Tier 1 RIFT2 (vendored port) wrapper; Tier 2 LightGlue+ALIKED escalation; SIFT baseline
  refine/         grid bucketing + ANMS + quotas, MAGSAC++ verification, NCC sub-pixel refinement
  warp/           piecewise-affine tile warping + blending, COG export, GeoJSON/metrics sidecars
  trust/          calibrated trust flag + circularity guard (fit/held-out split)
  metrics/        RMSE, inlier stats, uniformity score, residual map
  testing/        synthetic fixture generator
api/              FastAPI demo service
web/              React viewer (roadmap item)
vendored/rift2/   RIFT2 Python port, pinned + cited (see VENDOR_NOTE.md)
tests/            unit tests on synthetic fixtures
data/             local only, never commit; download_log.csv lives here
```

## Pipeline

Stage 1: ingest (PDS4/QUB, CRS unification, IIRS band selection, tiling) -> Stage 2:
two-tier matcher cascade (Tier 1 RIFT2 phase congruency always; Tier 2
LightGlue+ALIKED only on escalation) -> Stage 3: grid bucketing + ANMS + per-tile
quotas + empty-cell re-match, MAGSAC++ verification FIRST, then NCC-window +
2D-parabolic sub-pixel refinement -> Stage 4: piecewise-affine tile warping with
overlap blending, COG export, match-point GeoJSON sidecar, metrics JSON.

Run it:

```bash
.venv/Scripts/python scripts/run_pipeline.py \
  config/ohrc.yaml fixtures/pair_shift_a.tiff fixtures/pair_shift_b.tiff \
  outputs/demo metrics.json
```

## Honest claims

Prototype demonstrating generic multimodal lunar image correspondence and
registration to a reference base map under illumination, viewpoint and scale
variation, with sub-pixel RMSE (target) and verified uniform tie-point distribution,
validated against classical and learned baselines and two independent reference
archives (LROC and Kaguya).

Every leg exists (per-source registration studies); the engine does not. This
prototype is the engine claim, at Stage-1 scope, on synthetic data. Not: 100%
accurate, not real-time ISRO operational, not fully autonomous lunar mapping, no
permanently-shadowed-region claims (robustness covers polar imagery under low/oblique
sun only), no RCM/S success-rate numbers without manual labels.

## License table

| Component | License | Notes |
|---|---|---|
| RIFT2 (vendored port: canyagmur/RIFT2-multimodal-matching-rotation-python @ 7d92f00) | no explicit OSI license; research code, attribution-required | Li et al. 2020/2023, RIFT/RIFT2. Cited in VENDOR_NOTE.md. Re-check before any commercial use. |
| LightGlue + ALIKED (escalation path; not exercised on synthetic runs) | Apache-2.0 (LightGlue) / BSD-3-Clause (ALIKED) | pretrained weights; escalation-only |
| DISK | Apache-2.0 | Tier-2 alternative extractor |
| OpenCV (SIFT/AKAZE/MAGSAC++) | Apache-2.0 | baselines + geometric verification |
| SuperPoint | Magic Leap non-commercial research | fallback only; disclose, never "open source" |
| RoMa / EfficientLoFTR | MIT+Apache-2.0 / research | offline benchmarking only, never wired live |
| Chandrayaan-2 data | ISRO copyright, non-profit scientific use | attribute; no commercial claims |

## Download traceability

Every external number must trace to a run ID in `data/download_log.csv`
(product ID, source URL, acquisition date, solar geometry, scale ratio, local path,
run ID). The CSV ships header-only: no downloads have happened. No entry = no claim.
