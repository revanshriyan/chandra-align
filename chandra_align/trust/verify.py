"""Phase 10 — independent verification runner.

verify_pair(img_a, img_b, M, ...): runs the pixel area-check on a fitted
transform. It uses ONLY the images and M — never the feature correspondences
that produced M — so it is pipeline-independent evidence.

The area-check is the per-pair check; loop_closure (triplets) and
HardenedGT (human landmarks) are the other two legs of the trust layer.
"""

from .area_check import area_check, calibrate_thresholds
from .loop_closure import loop_closure, compose_affine
from .gt_hardened import HardenedGT


def verify_pair(img_a, img_b, M, grid=(6, 6), pair_id=""):
    """Independent pixel-level verification of transform M (A->B).

    Returns the area-check dict plus a plain-language summary. A "verified"
    overall means most grid cells show a sharp correlation peak at the
    expected location AFTER warping — evidence from the pixels, not the
    matcher.
    """
    r = area_check(img_a, img_b, M, grid=grid)
    r["pair_id"] = pair_id
    r["summary"] = (
        f"{pair_id or 'pair'}: pixel area-check {r['overall']} "
        f"({r['n_verified']}/{r['n_cells']} cells verified, "
        f"{r['n_weak']} weak, {r['n_no_evidence']} no evidence)"
    )
    return r
