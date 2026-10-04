# Synthetic gentle ground-truth points

Status: measured independent calibration control set, 25 points.

The benchmark's known-warp record is `ppt_assets/data/01_known_vs_recovered_warp.csv`: `dx=+1.8 px`, `dy=-1.2 px`, `theta=+0.25°`. This CSV uses the project affine convention `x_src=cos(theta)*x_ref-sin(theta)*y_ref+dx`, `y_src=sin(theta)*x_ref+cos(theta)*y_ref+dy` on the canonical 900×900 pair. It places 25 deterministic checkpoints on a 5×5 grid at reference coordinates 90, 270, 450, 630, and 810 px. These points are generated from known parameters, not selected from any matcher output.

The measured synthetic transforms and exact fit-inlier coordinates are preserved in `results/issue02_fits/synthetic_gentle_*.npz`. The evaluator verified no checkpoint coincided with a fit inlier within 1e-6 px for LightGlue/ALIKED or SIFT/RANSAC. RIFT2 produced no valid transform, so no ground-truth RMSE is available for that row.
