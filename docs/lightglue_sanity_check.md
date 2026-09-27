# LightGlue Tier-2 Sanity Check — CONFIRMED HEALTHY

**Date**: 2026-09-14
**Run**: Synthetic fixture (mild illumination change: dx=5.0, dy=2.0, gamma=1.6, gain=1.25, gradient=0.35)

| Metric | Value |
|--------|-------|
| Raw matches | 155 |
| Verified inliers (RANSAC) | 155 |
| Inlier ratio | 1.0 |
| Status | **PASS — Tier-2 pipeline path is healthy and functional** |

## Notes
- LightGlue + ALIKED wrapper successfully executed on CPU (torch 2.14.0+cpu)
- Tensor shapes: input images (512, 512) → extracted keypoints → matched 155 pairs
- All 155 raw matches survived RANSAC verification (inlier ratio 1.0)
- No tensor-shape interface issues detected
- Tier-2 escalation path is operational and ready for real-data use