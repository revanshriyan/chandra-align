"""Funnel diagnosis (Phase 9): find WHERE cross-modal matching starves.

A matcher pipeline is a funnel — detection -> description/matching ->
geometric verification — and a failed pair usually dies at one specific
stage. This module runs the three stages with injected callables and
reports per-stage counts, so a diagnosis says "detection starved" or
"descriptors matched but RANSAC found no model" instead of just "failed".

What it does NOT do: it does not fix anything, rank true correspondences,
or decide a verdict. It is a measuring instrument, not a matcher.
"""

import numpy as np


def _count_points(obj):
    if obj is None:
        return 0
    arr = np.asarray(obj)
    if arr.size == 0:
        return 0
    return int(arr.shape[0]) if arr.ndim >= 1 else 0


def diagnose_funnel(detect_fn, describe_match_fn, ransac_fn, img_a, img_b):
    """Run detection -> description/matching -> RANSAC with stage counts.

    Callable contracts (kept deliberately loose so any matcher plugs in):
      detect_fn(img) -> dict with 'points': (N,2) and optional
          'descriptors': (N,D) or None.
      describe_match_fn(det_a, det_b) -> dict with 'pairs': (M,2) index
          pairs into det_a['points'] / det_b['points'], and optional
          'scores': (M,) (e.g. Lowe ratios — smaller is better).
      ransac_fn(pa, pb) -> dict with 'inliers': boolean (M,) mask or
          integer (I,) indices, 'model' (fitted transform or None), and
          optional 'inlier_rank': caller-defined extra info, passed
          through untouched.

    Returns a dict with per-stage counts:
      n_detected_a, n_detected_b, n_descriptors_a, n_descriptors_b,
      n_raw_matches, n_ransac_inliers, inlier_rank (passthrough, may be
      None), plus 'model' and the raw stage outputs under 'stages' for
      deeper inspection.
    """
    det_a = detect_fn(img_a) or {}
    det_b = detect_fn(img_b) or {}
    pts_a = np.asarray(det_a.get("points", np.zeros((0, 2))), dtype=np.float64)
    pts_b = np.asarray(det_b.get("points", np.zeros((0, 2))), dtype=np.float64)
    desc_a = det_a.get("descriptors")
    desc_b = det_b.get("descriptors")

    match_out = describe_match_fn(det_a, det_b) or {}
    pairs = np.asarray(match_out.get("pairs", np.zeros((0, 2), dtype=int)))
    if pairs.size and pairs.ndim == 2 and pairs.shape[1] == 2:
        valid = (
            (pairs[:, 0] >= 0) & (pairs[:, 0] < len(pts_a))
            & (pairs[:, 1] >= 0) & (pairs[:, 1] < len(pts_b))
        )
        pairs = pairs[valid].astype(int)
    else:
        pairs = np.zeros((0, 2), dtype=int)

    ransac_out = {}
    n_inliers = 0
    model = None
    inlier_rank = None
    if len(pairs):
        pa = pts_a[pairs[:, 0]]
        pb = pts_b[pairs[:, 1]]
        ransac_out = ransac_fn(pa, pb) or {}
        model = ransac_out.get("model")
        inl = ransac_out.get("inliers")
        if inl is not None:
            inl = np.asarray(inl)
            if inl.dtype == bool:
                n_inliers = int(inl.sum()) if inl.size == len(pa) else 0
            else:
                n_inliers = int(len(np.unique(inl)))
        inlier_rank = ransac_out.get("inlier_rank")  # passthrough

    return {
        "n_detected_a": _count_points(pts_a),
        "n_detected_b": _count_points(pts_b),
        "n_descriptors_a": 0 if desc_a is None else _count_points(desc_a),
        "n_descriptors_b": 0 if desc_b is None else _count_points(desc_b),
        "n_raw_matches": int(len(pairs)),
        "n_ransac_inliers": n_inliers,
        "inlier_rank": inlier_rank,
        "model": model,
        "stages": {"detect_a": det_a, "detect_b": det_b,
                   "matches": match_out, "ransac": ransac_out},
    }
