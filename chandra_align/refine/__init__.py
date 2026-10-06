"""Stage 3 — uniformity control, verification, sub-pixel refinement.

Order is binding (project file §5): grid bucketing + ANMS/SSC + per-tile quotas +
empty-cell re-match -> MAGSAC++ (cv2.RANSAC) verification FIRST -> then
NCC-window + 2D-parabolic sub-pixel refinement. Refinement sharpens a verified match;
it never runs before verification and cannot rescue a wrong one.

The uniformity score formula is defined ONCE here and frozen:
    uniformity = 0.5 * occupied_cell_fraction + 0.5 * normalised_NNI
where occupied_cell_fraction = occupied cells / total cells over the `grid` layout
and NNI = mean nearest-neighbour distance / (0.5 / sqrt(n / area)) (Clark-Evans
index), clipped to [0, 1]. The formula does not change between runs; only the
config grid changes.
"""

import numpy as np


# ---------------- Stage 3a: grid bucketing + ANMS/SSC ----------------

def grid_bucket(points, shape, grid=(4, 4)):
    """Assign points to grid cells over the image extent. Returns {(row, col): [idx]}."""
    h, w = shape
    gh, gw = grid
    pts = np.asarray(points)
    if pts.size == 0:
        return {}
    out = {}
    cols = np.clip((pts[:, 0] / max(w, 1) * gw).astype(int), 0, gw - 1)
    rows = np.clip((pts[:, 1] / max(h, 1) * gh).astype(int), 0, gh - 1)
    for r, c, idx in zip(rows, cols, range(len(pts))):
        out.setdefault((int(r), int(c)), []).append(idx)
    return out


def anms(points, scores, shape, grid=(4, 2), per_tile_quota=8):
    """Grid bucketing + ANMS/SSC selection with a per-tile quota.

    points: (N, 2) array, scores: per-point strength (higher kept first).
    Within each occupied cell, keep up to per_tile_quota points spread by
    min-distance (ANMS: repeatedly take the point whose distance to the
    already-kept set is largest). Returns (selected (M, 2), indices).
    """
    pts = np.asarray(points, dtype=np.float64)
    n = len(pts)
    if n == 0:
        return np.zeros((0, 2)), np.zeros(0, dtype=int)
    if scores is None:
        scores = np.ones(n)
    scores = np.asarray(scores, dtype=np.float64)
    order = list(np.argsort(-scores))
    buckets = grid_bucket(pts, shape, grid)
    keep = []
    for cell, idxs in sorted(buckets.items()):
        idx_set = set(idxs)
        cell_order = [i for i in order if i in idx_set]
        if not cell_order:
            continue
        kept_cell = [cell_order[0]]
        remaining = cell_order[1:]
        while remaining and len(kept_cell) < per_tile_quota:
            best_i, best_d = None, -1.0
            for i in remaining:
                d = min(np.hypot(*(pts[i] - pts[j])) for j in kept_cell)
                if d > best_d:
                    best_d, best_i = d, i
            kept_cell.append(best_i)
            remaining.remove(best_i)
        keep.extend(kept_cell)
    keep = sorted(keep)
    return pts[keep], np.asarray(keep, dtype=int)


def rematch_empty_cells(points, shape, grid=(4, 4), per_tile_quota=8,
                        rematch_fn=None, selected_indices=None):
    """Re-match pass on empty cells: run `rematch_fn(cell_box)` for each empty cell.

    rematch_fn must return (points_in_cell, scores) or ([] , []).
    Returns (points (M, 2), indices, n_cells_filled).
    """
    pts = np.asarray(points, dtype=np.float64)
    h, w = shape
    gh, gw = grid
    if selected_indices is None:
        sel_idx = np.arange(len(pts), dtype=int)
    else:
        sel_idx = np.asarray(selected_indices, dtype=int)
    occupied = set(grid_bucket(pts[sel_idx], shape, grid).keys())
    n_filled = 0
    extra_pts, extra_idx = [], []
    base = len(pts)
    for r in range(gh):
        for c in range(gw):
            if (r, c) in occupied or rematch_fn is None:
                continue
            x0, x1 = c * w / gw, (c + 1) * w / gw
            y0, y1 = r * h / gh, (r + 1) * h / gh
            new_pts, new_scores = rematch_fn((x0, y0, x1, y1))
            if len(new_pts) == 0:
                continue
            sub, idx = anms(np.asarray(new_pts), np.asarray(new_scores),
                            (y1 - y0, x1 - x0), grid=(1, 1),
                            per_tile_quota=per_tile_quota)
            sub = sub + np.array([x0, y0])
            extra_pts.extend(sub.tolist())
            extra_idx.extend((idx + base + len(extra_pts) - len(sub)).tolist())
            base += len(new_pts)
            n_filled += 1
    all_pts = np.vstack([pts, np.asarray(extra_pts, dtype=np.float64)]) if extra_pts else pts
    all_idx = np.concatenate([sel_idx, np.asarray(extra_idx, dtype=int)]) if extra_idx else sel_idx
    return all_pts, all_idx, n_filled


# ---------------- Uniformity score (frozen formula) ----------------

def uniformity_score(points, shape, grid=(4, 4)):
    """Frozen formula: 0.5 * occupied_cell_fraction + 0.5 * normalised_NNI.

    normalised_NNI = mean-NN-distance / (0.5 / sqrt(n / area)), clipped to [0, 1].
    Deterministic; depends only on points, shape, grid.
    """
    pts = np.asarray(points, dtype=np.float64)
    h, w = shape
    if len(pts) < 2:
        return 0.0
    gh, gw = grid
    buckets = grid_bucket(pts, shape, grid)
    occupied_frac = len(buckets) / (gh * gw)

    area = float(h * w)
    n = len(pts)
    expected = 0.5 / np.sqrt(n / area)
    d = np.hypot(pts[:, 0][:, None] - pts[:, 0][None, :],
                 pts[:, 1][:, None] - pts[:, 1][None, :])
    np.fill_diagonal(d, np.inf)
    mean_nn = float(d.min(axis=1).mean())
    nni = float(np.clip(mean_nn / expected, 0.0, 1.0)) if expected > 0 else 0.0
    return 0.5 * occupied_frac + 0.5 * nni


# ---------------- Stage 3b: MAGSAC++ verification (FIRST) ----------------

def verify_magsac(points_a, points_b, cfg_verification: dict):
    """Geometric verification with cv2.RANSAC (MAGSAC++ fallback).

    OpenCV 5.0+ uses RANSAC for estimateAffinePartial2D; USAC_MAGSAC is not
    supported on that function. We use RANSAC with the same config params.
    Returns (inliers_a, inliers_b, model, inlier_ratio) where model is the fitted
    affine (None if verification failed). Runs BEFORE refinement.
    """
    import cv2

    pa = np.asarray(points_a, np.float32).reshape(-1, 2)
    pb = np.asarray(points_b, np.float32).reshape(-1, 2)
    n = len(pa)
    if n < 3:
        return (np.zeros((0, 2), np.float32), np.zeros((0, 2), np.float32),
                None, 0.0)
    thresh = float(cfg_verification.get("ransac_reproj_threshold", 3.0))
    max_iters = int(cfg_verification.get("max_iters", 2000))
    confidence = float(cfg_verification.get("confidence", 0.99))
    M, mask = cv2.estimateAffinePartial2D(
        pa, pb, method=cv2.RANSAC,
        ransacReprojThreshold=thresh, maxIters=max_iters, confidence=confidence)
    if M is None or mask is None:
        return (np.zeros((0, 2), np.float32), np.zeros((0, 2), np.float32), None, 0.0)
    mask = mask.ravel().astype(bool)
    inl_a, inl_b = pa[mask], pb[mask]
    ratio = float(mask.sum()) / n if n else 0.0
    return inl_a, inl_b, M, ratio


# ---------------- Stage 3c: NCC sub-pixel refinement (AFTER verification) ---------

def refine_subpixel_ncc(img_a, img_b, pts_a, pts_b, ncc_window=11, search_range_px=1):
    """11x11 (or 15x15) NCC window + 2D parabolic peak fit -> sub-pixel offsets.

    Runs AFTER MAGSAC++ verification. For each verified pair, take a (2r+1) window in
    A, search a (2r+1+2*search) window in B, compute normalised cross-correlation,
    find the integer peak, then fit a 2D parabola through the 3x3 neighbourhood of
    the peak for sub-pixel offset. Returns refined pts_b (float64) + residual stats.
    Points whose window falls outside the image or whose NCC peak is degenerate are
    dropped honestly (no interpolation across failure).
    """
    import cv2

    a = np.asarray(img_a, np.float32)
    b = np.asarray(img_b, np.float32)
    ha, wa = a.shape
    hb, wb = b.shape
    r = ncc_window // 2
    out_b = []
    kept_a = []
    for (xa, ya), (xb, yb) in zip(np.asarray(pts_a, np.float64),
                                  np.asarray(pts_b, np.float64)):
        xa_i, ya_i = int(round(xa)), int(round(ya))
        xb_i, yb_i = int(round(xb)), int(round(yb))
        if not (r <= xa_i < wa - r and r <= ya_i < ha - r):
            continue
        patch = a[ya_i - r:ya_i + r + 1, xa_i - r:xa_i + r + 1]
        if patch.std() < 1e-6:
            continue
        sx, sy = search_range_px, search_range_px
        x0, x1 = xb_i - r - sx, xb_i + r + sx + 1
        y0, y1 = yb_i - r - sy, yb_i + r + sy + 1
        if x0 < 0 or y0 < 0 or x1 > wb or y1 > hb:
            continue
        region = b[y0:y1, x0:x1]
        res = cv2.matchTemplate(region, patch, cv2.TM_CCOEFF_NORMED)
        _, _, _, peak = cv2.minMaxLoc(res)
        px, py = peak
        if px <= 0 or py <= 0 or px >= res.shape[1] - 1 or py >= res.shape[0] - 1:
            continue  # parabola needs the 3x3 neighbourhood; edge peaks dropped
        # 2D parabolic sub-pixel fit around the integer peak
        c = res[py, px]
        l, cc, rr = res[py, px - 1], res[py, px], res[py, px + 1]
        t, bo = res[py - 1, px], res[py + 1, px]
        denom_x = (rr - 2 * cc + l)
        denom_y = (bo - 2 * cc + t)
        # For a concave peak the denominator is negative; this sign convention
        # returns a positive offset when the sampled maximum lies to the right/down.
        dx = 0.5 * (l - rr) / denom_x if abs(denom_x) > 1e-12 else 0.0
        dy = 0.5 * (t - bo) / denom_y if abs(denom_y) > 1e-12 else 0.0
        if abs(dx) > 1.0 or abs(dy) > 1.0:
            continue  # parabola apex outside the ±1 px neighbourhood: degenerate
        # matchTemplate reports the patch's top-left corner; restore its center
        # before applying the fitted fractional offset.
        sub_x = x0 + px + r + float(np.clip(dx, -1, 1))
        sub_y = y0 + py + r + float(np.clip(dy, -1, 1))
        out_b.append((sub_x, sub_y))
        kept_a.append((xa, ya))
    if not out_b:
        return np.zeros((0, 2), np.float64), np.zeros((0, 2), np.float64), {}
    out_b = np.asarray(out_b, np.float64)
    kept_a = np.asarray(kept_a, np.float64)
    res = out_b - kept_a
    stats = {
        "refined_pairs": int(len(out_b)),
        "refine_residual_mean_px": float(np.hypot(res[:, 0], res[:, 1]).mean()),
        "refine_residual_std_px": float(np.hypot(res[:, 0], res[:, 1]).std()),
    }
    return kept_a, out_b, stats


# ---------------- Phase 8: guarded verification ----------------

def verify_guarded(points_a, points_b, cfg_verification: dict, image_shape=None):
    """RANSAC verification with Phase 8 guards.

    Pipeline: pre-fit ABSTAIN classification -> verify_magsac -> unique-inlier
    dedup (2 px) -> RANSAC span guard (<50% Y-span relaxes the threshold up to
    12 px and refits) -> Gate 3 transform-conditioning backstop.

    Returns a dict with keys: abstain_code (None when a fit was attempted),
    inliers_a, inliers_b, model, inlier_ratio, n_raw, n_unique, span_guard
    (whether the threshold was relaxed), gate3 (conditioning report), and
    ok (fit accepted by every guard).
    """
    from chandra_align.metrics.conditioning import (
        check_transform_conditioning,
        classify_prefit_abstain,
        dedup_correspondences,
    )

    pa = np.asarray(points_a, np.float32).reshape(-1, 2)
    pb = np.asarray(points_b, np.float32).reshape(-1, 2)
    out = {
        "abstain_code": None, "inliers_a": np.zeros((0, 2), np.float32),
        "inliers_b": np.zeros((0, 2), np.float32), "model": None,
        "inlier_ratio": 0.0, "n_raw": int(len(pa)), "n_unique": 0,
        "span_guard": False, "gate3": {}, "ok": False,
    }
    code = classify_prefit_abstain(pa, pb)
    if code is not None:
        out["abstain_code"] = code
        return out

    cfg = dict(cfg_verification)
    inl_a, inl_b, M, ratio = verify_magsac(pa, pb, cfg)
    if M is None:
        out["abstain_code"] = "NO_VALID_MODEL"
        return out

    # Span guard: inliers covering <50% of the Y extent get a relaxed
    # threshold (up to 12 px) and a refit instead of accepting collapse.
    if image_shape is not None and len(inl_a) >= 3:
        h = float(image_shape[0])
        y_span = float(inl_a[:, 1].max() - inl_a[:, 1].min()) if len(inl_a) else 0.0
        base_thresh = float(cfg.get("ransac_reproj_threshold", 3.0))
        if h > 0 and (y_span / h) < 0.5 and base_thresh < 12.0:
            cfg["ransac_reproj_threshold"] = min(12.0, base_thresh * 2.0)
            inl_a, inl_b, M, ratio = verify_magsac(pa, pb, cfg)
            out["span_guard"] = True
            if M is None:
                out["abstain_code"] = "NO_VALID_MODEL"
                return out

    # Unique-inlier dedup before gating; report raw + unique.
    u_a, u_b, _idx, n_raw, n_unique = dedup_correspondences(inl_a, inl_b, radius=2.0)
    out["n_raw"], out["n_unique"] = n_raw, n_unique
    out["inliers_a"] = u_a.astype(np.float32)
    out["inliers_b"] = u_b.astype(np.float32)
    out["inlier_ratio"] = float(n_unique) / len(pa) if len(pa) else 0.0

    # Gate 3 backstop on the fitted model.
    ok3, rep3 = check_transform_conditioning(M)
    out["gate3"] = rep3
    if not ok3:
        out["abstain_code"] = "NO_VALID_MODEL"
        out["model"] = None
        return out
    out["model"] = M
    out["ok"] = True
    return out
