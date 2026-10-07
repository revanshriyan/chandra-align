# Phase 17 — ISIS3 `coreg` incumbent comparison

## Purpose

Credibility against the industry standard: run USGS ISIS3 `coreg` on the
**identical** OHRC/TMC-2 crops our pipeline measured in Phase 13, and compare
residuals and failure behavior. Framed as a robustness/failure-behavior
comparison, **not** as a replacement exercise. Scope is OHRC/TMC-2 only —
ISIS3 ships no IIRS camera model, so IIRS is out of scope by construction.

## Setup

- **ISIS3 10.0.0** (`usgs-astrogeology` conda channel; `isis 10.0.0
  h1f94ec8_1`), installed locally via micromamba. No SPICE kernels needed:
  crops were imported with `raw2isis` (pixel space, no camera geometry), so
  both tools operate in the same pixel-residual frame.
- **coreg parameters**: `MaximumCorrelation` algorithm, tolerance 0.7,
  pattern chip 20x20 px. Two search configurations: the stock template
  (`coreg.maxcor.p2020.s5050.def`, search 50x50 px) and a wide-search variant
  (search 140x140 px, +/-60 px capture range). 9x9 chip grid (81 chips).
  Transforms tested: `TRANSLATE` and `WARP DEGREE=1` (affine, comparable to
  our per-window affine fit).
- **Crops** (1024x1024, cut from the local PRADAN products with the measured
  inter-product affines from `scripts/make_benchmark_crops.py` — the same
  mapping Phase 13 used):
  - `ohrc_w0000` — OHRC 12:09/14:06, Phase 13: 1692 unique inliers,
    1.53 px, COARSE_ADVISORY.
  - `tmc2_w0000` — TMC-2 fore/nadir, Phase 13: 1048 unique inliers,
    1.41 px, COARSE_ADVISORY.
  - `tmc2_w0001` — TMC-2 fore/nadir, Phase 13: 1664 inliers, 2.92 px,
    DEGENERATE_FAILURE (residual straddles the 2.5 px gate).
  - `synth_12m7` — self-pair of the TMC-2 base crop shifted by exactly
    (dy=+12, dx=-7) px; validates the coreg invocation before judging it.
- Machine-readable results: `results/phase17_coreg.csv`. Repro driver:
  `scripts/run_phase17_coreg.sh`.

## Results

| Crop | coreg config | Chips passed | Consensus (within 5 px of median) | Coherent? | Ours (Phase 13) |
| --- | --- | --- | --- | --- | --- |
| synth (known shift) | TRANSLATE, stock | 81/81 | 100% | **yes** — recovered exactly (dx=+7, dy=-12) | 0.37 px ACCEPTED |
| tmc2_w0000 | TRANSLATE, stock | 4/81 | 0% | no | 1.41 px COARSE |
| tmc2_w0000 | TRANSLATE, wide | 33/81 | 3% | no | 1.41 px COARSE |
| tmc2_w0000 | WARP d1, wide | 35/81 | 3% | no | 1.41 px COARSE |
| tmc2_w0001 | TRANSLATE, wide | 57/81 | 4% | no | 2.92 px DEGENERATE |
| ohrc_w0000 | TRANSLATE, wide | 47/81 | 0% | no | 1.53 px COARSE |

"Consensus" = fraction of passing chips whose (dx, dy) falls within 5 px of
the median offset. On every real pair it is 0-4%: the passing chips scatter
across the full search range (e.g. dx in [-43, +51] px) — false locks on
repetitive lunar texture, not a transform.

## Headline finding

`coreg` is a **precision refinement tool, not a global registration tool**.
On the synthetic pair with a small known offset it is flawless (81/81 chips,
exact recovery, GoodnessOfFit 1.0). On real lunar pairs carrying the
~30-80 px residual offsets left by the measured inter-product affines, its
area-based chip matching cannot establish correspondence: with the stock
template 76/81 chips fail tolerance outright, and widening the search to
+/-60 px only converts outright failures into **false passes** — chips that
clear the 0.7 correlation tolerance yet disagree with each other completely.

Our SIFT-based pipeline, doing global feature matching before the gated
affine fit, returns COARSE registrations (1.4-1.5 px, 1000+ inliers) on the
same crops where coreg returns nothing coherent.

## Failure-behavior comparison (the point of this phase)

- **Ours fails closed.** On `tmc2_w0001` the residual (2.92 px) straddles the
  frozen 2.5 px gate: verdict DEGENERATE_FAILURE, transform telemetry masked,
  reasons reported. No number is presented as a registration.
- **coreg does not fail closed.** On the same hard pair it reports
  "Successful = 57" chips with GoodnessOfFit up to 0.84 — numbers that look
  like success but contain no consensus transform (4% agreement). A user
  trusting the success count would walk away with a confident-but-wrong
  registration. This is precisely the failure mode our Phase 8/10 gates were
  built to prevent, and it reproduces independently of our code.

## Honest limitations

1. No camera models / SPICE: both tools compared in pixel space. A full
   ISIS pipeline with `spiceinit` + map projection might behave differently;
   that is a different experiment (and needs kernels we did not fetch).
2. Only three real crops (+ one synthetic control). They were chosen to span
   our verdict range (COARSE x2, DEGENERATE x1), not to be a survey.
3. An expert ISIS operator hand-tuning per-pair templates, seed points, or
   iterative refinement might coax more out of coreg; we tested the tool as
   a straightforward user would run it: stock template, then one principled
   widening of the search range.
4. coreg was given no better initialization than our crops as cut. Better
   coarse alignment upstream would help it — which is itself the finding:
   it needs what our pipeline provides.
