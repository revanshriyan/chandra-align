# Matcher keypoint-cap sweep — does raising the cap let the stage engage in-pipeline? — 2026-10-08

## Question

The sub-pixel story has one structural blocker: the true pipeline's `match_pair_hf`
caps SIFT keypoints at 64 per cell on a 4x4 grid (`select_detector_keypoints(...,
64)` in `app.py`; SIFT itself allows `nfeatures=10000`), while the standalone
SUCCESS_SUBPIXEL numbers came from an 842-inlier set that never exists in-pipeline
(see `docs/stage-prepruning-2026-10-08.md`). The deform stage needs a rich inlier
set to see distortion. Does raising the cap fix that?

## Method

`app.py` imported via the throwaway `/tmp/stubs` (gradio/spaces/matplotlib — never
committed; same trick as A15/A16). **Only** `app.select_detector_keypoints` is
monkeypatched with a parameterized `quota_per_cell`; everything else — matcher,
hook RANSAC, stage, balance/bucket/NCC chain, frozen gate — is the true
`_align_core`. The single difference between runs is the quota (and the
`CHANDRA_DEFORM_FIELD` flag). No repo file was modified.

Quotas: 64 (current), 128, 256, 10**9 (== uncapped; bounded in practice by SIFT
`nfeatures=10000`, observed ~10,000 keypoints/image). Pairs: ohrc_01/02/03/05,
tmc2_04 (`data/benchmark_crops/`). 40 runs: 5 pairs x 4 quotas x flag off/on.
Seed 7; determinism re-check (ohrc_01/64/off re-run) **PASS** — byte-identical
outputs. Frozen gates throughout.

Full data: `results/table_cap_sweep.csv` (script: `scripts/cap_experiment.py`).
Transform agreement below is vs each pair's (64, off) baseline
(`affine_telemetry` rotation_deg/scale_s); pre-registered bars: drot > 0.05 deg
or dscale > 0.001 flags a row.

## Results

### Keypoint yield (first SIFT variant, mean over runs)

| Quota | kp/image | Lowe matches |
|---|---:|---:|
| 64 | ~1,024 | ~662 |
| 128 | ~2,044 | ~1,370 |
| 256 | ~4,034 | ~2,816 |
| uncapped | ~10,000 | ~7,047 |

### Flag-on verdicts and gate RMSE by quota (stage applied at lam=0.1 everywhere below)

| Pair | 64 | 128 | 256 | uncapped |
|---|---|---|---|---|
| ohrc_01 | COARSE 0.61 | COARSE 0.53 | **SUCCESS 0.48** | **SUCCESS 0.43** |
| ohrc_02 | COARSE 0.82 | COARSE 0.58 | **SUCCESS 0.46** | **SUCCESS 0.46** |
| ohrc_03 | COARSE 0.58 | **SUCCESS 0.40** | **SUCCESS 0.36** | **SUCCESS 0.34** |
| ohrc_05 | COARSE 0.79 | COARSE 1.12 | COARSE 0.58 | COARSE 0.53 |
| tmc2_04 | **SUCCESS 0.46** | COARSE 0.44 | COARSE 0.42 | COARSE 0.24 |

### Pre-registered clause (a): no flag-off downgrades vs quota 64

- **quota 128: KILLED.** ohrc_01 COARSE -> DEGENERATE_FAILURE. The transform is
  catastrophically wrong (drot 0.99 deg, dscale 1.002 — i.e. scale ~2 vs ~1):
  more keypoints let RANSAC lock onto a wrong local solution. Genuine trap hit.
- **quota 256: KILLED.** ohrc_05 COARSE -> DEGENERATE_FAILURE, same mechanism
  (dscale 1.007, wrong transform).
- **uncapped: PASS.** No flag-off downgrade on any pair (ohrc_01/02/03/05 stay
  COARSE; tmc2_04 upgrades DEGENERATE -> COARSE). Gate RMSE worsens on some
  pairs (ohrc_01 1.64->2.25, ohrc_03 1.10->2.01) but tiers hold.

The non-monotonicity (128 bad on ohrc_01, 256 bad on ohrc_05, uncapped clean)
shows the effect is *which* ambiguous matches enter, not how many — RANSAC
variant-selection luck, not a smooth trend.

### Pre-registered clause (b): SUCCESS_SUBPIXEL via stage, zero downgrades

**MET at uncapped**: ohrc_01/02/03 reach SUCCESS_SUBPIXEL (0.43/0.46/0.34 px)
through the true ordered pipeline, zero verdict downgrades anywhere (flag-off
vs 64, and flag-on vs flag-off at every quota), determinism PASS.

**With agreement-flag caveats** (reported per the pre-registered trap rule):
- ohrc_01 uncapped/on: drot 0.009 deg, dscale 0.0006 — **clean**, within bars.
- ohrc_02 uncapped/on: drot 0.035 deg (ok), dscale 0.0024 — trips the scale bar.
- ohrc_03 uncapped/on: drot 0.075 deg, dscale 0.0059 — trips both bars.
- (ohrc_01/02 at 256/on also trip bars.)

Interpretation, stated carefully: the bars (0.05 deg / 0.001) are *tighter than
the frame's own distortion signature* — A12 measured per-quadrant affine spread
of ~0.018 in scale, so different point samples of a distorted frame legitimately
yield affines differing by ~0.05-0.14 deg / ~0.002-0.006 scale. Moreover the
stage-on affine is fit on *field-corrected* points while the baseline absorbs
distortion, so some difference is expected and even desirable. What a REAL trap
looks like is calibrated by the killed rows: dscale ~1.0 (100x the bar) with a
DEGENERATE verdict. Against that calibration, and with the gate itself scoring
0.34-0.46 px on held-out (not in-sample), the flagged SUCCESS rows are not
wrong-fit events — but only ohrc_01's is bar-clean, and that distinction is
preserved here rather than smoothed over.

For tmc2_04 the agreement baseline (64/off) is itself DEGENERATE (7 inliers),
so trap flags there are vacuous — noted, not interpreted.

### Cost

Mean wall time: 17.6 s (64) / 17.4 s (128) / 18.9 s (256) / 29.1 s (uncapped,
max 49.7 s). Peak process RSS 1,233 MB. Uncapped costs ~1.6x time; acceptable.

## Recommendation

**Do not raise the cap globally.** Quotas 128/256 demonstrably let the default
(flag-off) pipeline lock onto catastrophically wrong transforms on 2/5 pairs;
uncapped was clean on this sample but the mechanism is pair-dependent luck, and
flag-off gate RMSEs degrade on some pairs even when tiers hold.

**If the stage is to get its rich set, gate the quota behind the same
`CHANDRA_DEFORM_FIELD=1` flag**: flag off -> quota 64 (bit-identical to today,
zero robustness risk); flag on -> uncapped. The opt-in user gets 3 ohrc
SUCCESS_SUBPIXEL verdicts (ohrc_01's bar-clean); the default pipeline never
sees the ambiguity risk. The stage's internal held-out already compares
field-vs-affine within the run, so the comparison stays fair.

That is a pipeline-design decision for the maintainer; this experiment changed
nothing in the repo.
