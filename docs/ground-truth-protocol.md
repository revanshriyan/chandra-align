# Independent Ground-Truth RMSE Protocol

This protocol measures transform accuracy against coordinate pairs that were not used to estimate or refine that transform. The fit's split-based held-out RMSE remains a useful internal diagnostic, but it is not independent when its check points are drawn from the fit's own inlier set.

## Ground truth by pair type

- **Synthetic pairs:** use the exact known warp used to generate the target image. Record the warp parameters, coordinate convention, pivot, image dimensions, and generation source. Convert them into reference/source pixel correspondences over a regular grid. Do not select points from the matcher's inliers.
- **Real pairs:** use independent LROC NAC or SLDEM-derived control points covering the same surface region, transformed into each input image's pixel coordinates with documented product identifiers, projection/camera models, crop offsets, and software/version. If such products cannot be used, manually pick recognizable landmarks in both original inputs. Record the picker, date, image/crop identity, landmark description, coordinates, and any ambiguous picks; do not pick from the match overlay or after seeing residuals.

No local independent LROC NAC/SLDEM control set was found for the OHRC or TMC-2 benchmark pair regions in this checkout. The local `M181058717LE` / `M181073012LE` NAC files are a separate tutorial stereo pair and do not provide tie points for those regions. Their CSV templates are in `data/ground_truth/`; landmark collection remains pending human work.

## Minimum size and spread

For a headline RMSE, require **at least 20 valid checkpoints**, with at least **3 in each of the four image quadrants** and points spanning at least 60% of the reference and source crop width and height. Prefer 25 or more. Report the count and quadrant counts with every RMSE. A three-point set is not trustworthy: it gives little redundancy, can cluster in one small area, and can make a local or degenerate fit look globally accurate. Three points may define a 2D affine transform exactly, leaving no meaningful independent validation.

If the count or spatial floor is not met, report the measured value only as exploratory and mark it `INSUFFICIENT_GT`; do not present it as a trustworthy pair-level RMSE. Never pad a set with inferred or duplicated points.

## Computation and circularity guard

For each ground-truth row, `(x_ref, y_ref)` is a point in the reference image and `(x_src, y_src)` is the corresponding point in the source image. Apply the already-fitted source-to-reference 2×3 affine transform to each source coordinate. For residuals `e_i = T(x_src_i, y_src_i) - (x_ref_i, y_ref_i)`, report radial pixel RMSE as `sqrt(mean(e_x_i^2 + e_y_i^2))`.

Run `python scripts/eval_ground_truth.py PAIR --ground-truth GT.csv --fit-artifact FIT.npz`. The fit artifact must contain the transform and the exact inlier coordinate pairs used by that run. The evaluator fails closed if either coordinate from any proposed checkpoint matches a fit inlier within its documented tolerance. A ground-truth point set must be fixed before inspecting evaluation residuals.

The app's split-based held-out RMSE is computed by splitting that run's fit inliers into fit/check subsets and evaluating its transform on the check subset. Those check points were seen by the matcher as correspondences and came from the same image pair, so this guards against scoring the transform on its direct fitting residuals but is not independent ground truth. Independent ground-truth RMSE uses a separately sourced/picked point set that the matcher and fitting process never saw. Keep the two metrics labeled and reported separately; do not substitute one for the other.
