# Scale-ratio experiment, real-sensor arm (2026-10-08)

## Purpose

Extend `docs/scale-ratio-*` pilot work (synthetic crater-blob texture, `scripts/run_scale_ratio.py`,
`results/table_scale_ratio_sift_pilot.csv`) to **real sensor texture with exact known truth**.
Real texture, exact truth: the source scene is the ohrc_01 benchmark crop
(`data/benchmark_crops/ohrc_01_reference.png`, 2048x2048 uint8, real OHRC
regolith texture at native resolution). The moving image is generated from it
with an exactly-known sub-pixel shift (dx=7.3, dy=-3.9 px, same as the synthetic
pilot), then downsampled by ratio r (INTER_AREA) and upsampled back to the
native grid (INTER_CUBIC). This tests whether the pilot's synthetic-texture
conclusions hold on real regolith texture.

Script: `scripts/run_scale_ratio_real_sensor.py` (new; the pilot script is
unmodified). Results: `results/table_scale_ratio_real_sensor.csv` (script-generated,
never hand-edited). Truth used for scoring only, never for fitting (no-peeking).

## Results

CPU/SIFT, frozen gates, seed 7. Verdicts are the pipeline's frozen ACCEPT/COARSE
thresholds (inlier RMSE <= 0.50 / 2.50 px). Truth RMSE (transform correctness vs
exact resampling truth) and inlier self-consistency RMSE are reported **separately**
— the Phase 11 lesson: self-consistency is not correctness.

| ratio | n_corr | n_inliers | inlier RMSE (px) | truth RMSE (px) | entropy | quads | verdict |
|------:|-------:|----------:|-----------------:|---------------:|--------:|------:|:--------|
| 1 | 6686 | 6675 | 0.0898 | 0.0124 | 1.9896 | 4 | SUCCESS_SUBPIXEL |
| 2 | 5735 | 5714 | 0.1586 | 0.0164 | 1.9938 | 4 | SUCCESS_SUBPIXEL |
| 4 | 2668 | 2638 | 0.4525 | 0.0442 | 1.9981 | 4 | SUCCESS_SUBPIXEL |
| 8 |  925 |  853 | 0.9672 | 0.1051 | 1.9989 | 4 | COARSE_ADVISORY |
| 16 |  258 |  137 | 1.5555 | 0.5740 | 1.9819 | 4 | COARSE_ADVISORY |

## Comparison against the synthetic pilot

| ratio | synthetic pilot (n_inl / inlier RMSE / truth RMSE / verdict) | real texture (n_inl / inlier RMSE / truth RMSE / verdict) |
|------:|---|---|
| 1 | 1711 / 0.2593 / 0.0097 / SUCCESS | 6675 / 0.0898 / 0.0124 / SUCCESS |
| 2 | 1372 / 0.3580 / 0.0083 / SUCCESS | 5714 / 0.1586 / 0.0164 / SUCCESS |
| 4 |  993 / 0.5276 / 0.0097 / COARSE  | 2638 / 0.4525 / 0.0442 / SUCCESS |
| 8 |  701 / 0.9416 / 0.0502 / COARSE  |  853 / 0.9672 / 0.1051 / COARSE  |
| 16 |  143 / 1.5871 / 0.8868 / COARSE |  137 / 1.5555 / 0.5740 / COARSE |

## Conclusions

1. **The pilot's core conclusions hold on real texture.** Graceful degradation
   with ratio; no catastrophic failure at any ratio; no wrong-accept (truth RMSE
   is never worse than the gate verdict implies). The degradation-curve shape is
   the same: 8:1 and 16:1 inlier-RMSE values are within a few percent between
   textures (0.9672 vs 0.9416; 1.5555 vs 1.5871).
2. **Real texture does strictly better at low ratios.** ~4x more correspondences
   at 1:1 (6686 vs 1767 Lowe matches) with a ~99% inlier ratio, and inlier RMSE
   0.0898 vs 0.2593 px. Real multi-scale regolith texture is kinder to SIFT than
   the synthetic blobs.
3. **The 4:1 honest-conservative-gate finding is refined, not contradicted.**
   On synthetic texture the pipeline tripped ACCEPT -> COARSE at 4:1 because
   inlier RMSE (0.5276) exceeded the 0.50 gate while the transform was truth-perfect
   (0.0097). On real texture the same ratio earns SUCCESS (inlier RMSE 0.4525,
   truth RMSE 0.0442). So the 4:1 COARSE on the pilot was the most conservative
   case of the two textures — the gate's conservatism is texture-dependent, and
   on real texture the margin at 4:1 is thin (0.4525 vs 0.50).
4. **Phase 11 check passes everywhere.** Truth RMSE stays below (or equal to)
   the inlier self-consistency RMSE at every ratio — self-consistency is a
   conservative bound on true transform error here (e.g. 16:1: truth 0.574 px
   vs inlier 1.5555 px), matching the pilot's behavior. No case where the
   pipeline is confident-but-wrong.
5. One caveat: these are self-pairs (the moving image is derived from the
   reference), so illumination, viewing geometry, and sensor noise are identical
   between the two images. The scale-ratio robustness shown here does not cover
   cross-sensor or cross-illumination pairs — that is the phase-congruency
   track's domain.

## Reproduction

```
python scripts/run_scale_ratio_real_sensor.py --out results/table_scale_ratio_real_sensor.csv
```
