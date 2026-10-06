"""Phase 11 — deterministic perturbation battery for robustness evaluation.

Every perturbation takes an explicit integer ``seed`` and builds its own
``numpy.random.Generator``: no global RNG state is touched, so a
(perturbation, seed) pair always yields byte-identical output. This is the
mechanical basis of the Phase 11 reproducibility claim.

Perturbations (applied to the moving image unless noted):
  identity          no-op baseline
  gaussian_noise    additive N(0, sigma) noise
  brightness_shift  additive offset, clipped to [0, 255]
  contrast_gain     (img - 127) * gain + 127
  gamma_curve       255 * (img / 255) ** gamma
  small_rotation    rotate about the image centre (BORDER_REPLICATE);
                    returns the warp matrix dM so the caller can compose the
                    ground truth as M_new = M_gt @ dM (homogeneous)
  jpeg_artifacts    JPEG encode/decode at the given quality (cv2, deterministic)
  sun_flip          255 - img (brightness inversion; shadow-reversal proxy for
                    Phase 14's phase-congruency work)

Also provided:
  overlap_coverage  grid coverage measured INSIDE the overlap polygon
                    (warped source footprint intersected with the reference
                    frame), not over the whole frame
  median_inlier_error
  independent_accuracy  probe-grid agreement with exact synthetic truth;
                    withheld ("no independent accuracy available") when the
                    transform is missing or supported by < 3 inliers
"""

import numpy as np

BATTERY_VERSION = "phase11-v1"


def _rng(seed):
    return np.random.default_rng(int(seed))


def perturb_identity(img, seed=0):
    """No-op baseline. Returns (img, {})."""
    return np.asarray(img, dtype=np.float64).copy(), {}


def perturb_gaussian_noise(img, sigma, seed):
    """Additive Gaussian noise with std ``sigma`` DN, seeded."""
    rng = _rng(seed)
    a = np.asarray(img, dtype=np.float64)
    out = a + rng.normal(0.0, float(sigma), a.shape)
    return np.clip(out, 0, 255), {"sigma": float(sigma)}


def perturb_brightness_shift(img, delta, seed=0):
    """Additive brightness offset ``delta`` DN, clipped to [0, 255]."""
    a = np.asarray(img, dtype=np.float64)
    return np.clip(a + float(delta), 0, 255), {"delta": float(delta)}


def perturb_contrast_gain(img, gain, seed=0):
    """Contrast about mid-grey: (img - 127) * gain + 127."""
    a = np.asarray(img, dtype=np.float64)
    return np.clip((a - 127.0) * float(gain) + 127.0, 0, 255), {"gain": float(gain)}


def perturb_gamma_curve(img, gamma, seed=0):
    """Gamma curve: 255 * (img / 255) ** gamma (input clipped to >= 0 first)."""
    a = np.asarray(img, dtype=np.float64)
    return np.clip(255.0 * (np.clip(a, 0, None) / 255.0) ** float(gamma),
                   0, 255), {"gamma": float(gamma)}


def perturb_small_rotation(img, angle_deg, seed=0):
    """Rotate ``angle_deg`` about the image centre (BORDER_REPLICATE).

    Returns (rotated, {"angle_deg": ..., "dM": [[..],[..]]}) where dM is the
    2x3 matrix passed to warpAffine (which inverts it: dst(x) = src(dM^{-1}x)).
    Ground-truth composition: if M_gt maps ref -> mov (the empirically
    verified convention of chandra_align.testing.make_pair_shift — its
    docstring says "moving->reference" but the recovered matrix is ref->mov),
    the rotated pair's truth is M_new = (dM_h @ M_gt_h)[:2, :] (homogeneous).
    """
    import cv2
    a = np.asarray(img, dtype=np.float64)
    h, w = a.shape[:2]
    dM = cv2.getRotationMatrix2D((w / 2.0, h / 2.0), float(angle_deg), 1.0)
    out = cv2.warpAffine(a, dM, (w, h), flags=cv2.INTER_CUBIC,
                         borderMode=cv2.BORDER_REPLICATE)
    return out, {"angle_deg": float(angle_deg), "dM": dM.tolist()}


def perturb_jpeg_artifacts(img, quality, seed=0):
    """JPEG encode/decode at ``quality`` (1-100). Deterministic for fixed input."""
    import cv2
    a = np.asarray(img, dtype=np.float64)
    u8 = np.clip(a, 0, 255).astype(np.uint8)
    ok, buf = cv2.imencode(".jpg", u8,
                           [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)])
    if not ok:
        raise RuntimeError("JPEG encode failed")
    dec = cv2.imdecode(buf, cv2.IMREAD_GRAYSCALE)
    return dec.astype(np.float64), {"quality": int(quality)}


def perturb_sun_flip(img, seed=0):
    """Brightness inversion: 255 - img.

    Shadow-reversal proxy: craters lit from one side become lit from the
    other. Phase 14's phase-congruency front-end is the principled answer;
    this perturbation is its acceptance test.
    """
    a = np.asarray(img, dtype=np.float64)
    return 255.0 - a, {}


PERTURBATIONS = {
    "identity": perturb_identity,
    "gaussian_noise": perturb_gaussian_noise,
    "brightness_shift": perturb_brightness_shift,
    "contrast_gain": perturb_contrast_gain,
    "gamma_curve": perturb_gamma_curve,
    "small_rotation": perturb_small_rotation,
    "jpeg_artifacts": perturb_jpeg_artifacts,
    "sun_flip": perturb_sun_flip,
}


def compose_gt_with_warp(M_gt, dM):
    """Compose ground truth through an image warp.

    M_gt: 2x3 mapping ref -> mov (empirically verified convention of
    chandra_align.testing.make_pair_shift). dM: 2x3 matrix passed to
    warpAffine, which inverts it (warped(x) = src(dM^{-1} @ x)).
    Returns 2x3 mapping ref -> warped-mov: (dM_h @ M_gt_h)[:2, :].
    """
    H_gt = np.vstack([np.asarray(M_gt, np.float64).reshape(2, 3), [0, 0, 1]])
    H_d = np.vstack([np.asarray(dM, np.float64).reshape(2, 3), [0, 0, 1]])
    return (H_d @ H_gt)[:2, :]


def overlap_coverage(inliers_b, M_ab, shape_b, grid=(8, 8)):
    """Fraction of overlap-polygon grid cells containing >= 1 inlier.

    The overlap polygon is the source footprint warped into frame B
    (via M_ab: A -> B) intersected with B's frame. Coverage is measured
    only inside that polygon — cells outside it do not count for or
    against the score. Returns a float in [0, 1].
    """
    import cv2
    inliers_b = np.asarray(inliers_b, np.float64).reshape(-1, 2)
    hb, wb = shape_b
    M = np.asarray(M_ab, np.float64).reshape(2, 3)
    # Overlap polygon in frame B: the footprint of A warped into B by M_ab,
    # intersected with B's frame. (Our synthetic pairs share one frame size,
    # so A's footprint is the full frame; warpAffine's output is B-sized.)
    ones = np.ones((hb, wb), np.float32)
    warped = cv2.warpAffine(ones, M.astype(np.float32), (wb, hb),
                           flags=cv2.INTER_NEAREST,
                           borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    overlap = warped > 0.5
    gh, gw = grid
    inside, hit = 0, 0
    for gy in range(gh):
        for gx in range(gw):
            y0, y1 = gy * hb // gh, (gy + 1) * hb // gh
            x0, x1 = gx * wb // gw, (gx + 1) * wb // gw
            cell = overlap[y0:y1, x0:x1]
            if cell.size == 0 or cell.mean() < 0.5:
                continue
            inside += 1
            sel = inliers_b[(inliers_b[:, 0] >= x0) & (inliers_b[:, 0] < x1)
                            & (inliers_b[:, 1] >= y0) & (inliers_b[:, 1] < y1)]
            if len(sel):
                hit += 1
    if inside == 0:
        return 0.0
    return float(hit) / float(inside)


def median_inlier_error(residuals):
    """Median of per-inlier residual magnitudes (robust companion to RMSE)."""
    r = np.asarray(residuals, np.float64).ravel()
    if r.size == 0:
        return None
    return float(np.median(r))


def measure_pair(img_a, img_b, M_gt, image_shape=None, sift_nfeatures=10000):
    """Run the CURRENT default pipeline and report robustness metrics.

    Pipeline (frozen; this function only measures it, never changes it):
    SIFT detect+compute -> BFMatcher kNN(k=2) + Lowe 0.75 ->
    chandra_align.refine.verify_guarded -> quadrant metrics ->
    validate_registration_gate (min_inliers=8).

    Returns a dict with detection counts, abstain/verdict codes, inlier RMSE
    AND median error, unique inlier count, entropy, overlap-aware coverage,
    and independent accuracy (probe-grid agreement with exact truth, withheld
    when unavailable).
    """
    import cv2
    from chandra_align.refine import verify_guarded
    from chandra_align.metrics import compute_quadrant_metrics, validate_registration_gate

    def _u8(a):
        a = np.asarray(a, dtype=np.float64)
        lo, hi = np.percentile(a, [1, 99])
        return np.clip((a - lo) * (255.0 / (hi - lo + 1e-9)), 0, 255).astype(np.uint8)

    A8, B8 = _u8(img_a), _u8(img_b)
    shape = image_shape or A8.shape
    sift = cv2.SIFT_create(nfeatures=int(sift_nfeatures),
                           contrastThreshold=0.005, edgeThreshold=15)
    ka, da = sift.detectAndCompute(A8, None)
    kb, db = sift.detectAndCompute(B8, None)
    rec = {"n_det_a": int(len(ka)), "n_det_b": int(len(kb)),
           "n_lowe": 0, "abstain": None, "verdict": "ABSTAIN",
           "rmse_px": None, "median_resid_px": None, "inliers": 0,
           "entropy": None, "overlap_coverage": None,
           "independent_accuracy_px": None, "accuracy_note": ""}
    if da is None or db is None or len(da) < 2 or len(db) < 2:
        rec["abstain"] = "ZERO_CANDIDATES"
        rec["accuracy_note"] = "no independent accuracy available"
        return rec
    matches = cv2.BFMatcher().knnMatch(da, db, k=2)
    good = [m for m, n in matches if m.distance < 0.75 * n.distance]
    rec["n_lowe"] = int(len(good))
    pa = np.float64([[ka[m.queryIdx].pt[0], ka[m.queryIdx].pt[1]] for m in good])
    pb = np.float64([[kb[m.trainIdx].pt[0], kb[m.trainIdx].pt[1]] for m in good])
    cfg = {"ransac_reproj_threshold": 3.0, "max_iters": 2000, "confidence": 0.99}
    g = verify_guarded(pa, pb, cfg, image_shape=shape)
    rec["abstain"] = g["abstain_code"]
    M = g["model"]
    if not g["ok"] or M is None:
        ia = g["inliers_a"]
        rec["inliers"] = int(g["n_unique"])
        rec["accuracy_note"] = "no independent accuracy available"
        return rec
    ia, ib = g["inliers_a"], g["inliers_b"]
    proj = ia @ M[:, :2].T + M[:, 2]
    resid = np.hypot(proj[:, 0] - ib[:, 0], proj[:, 1] - ib[:, 1])
    rmse = float(resid.mean())
    qm_counts, qm_entropy = compute_quadrant_metrics(ia, resid, shape)
    _msg, code = validate_registration_gate(rmse, len(ia), 8, qm_entropy,
                                            qm_counts, model=M)
    acc, note = independent_accuracy(M, M_gt, g["n_unique"], shape)
    rec.update({
        "verdict": code,
        "rmse_px": rmse,
        "median_resid_px": median_inlier_error(resid),
        "inliers": int(g["n_unique"]),
        "entropy": float(qm_entropy),
        "overlap_coverage": overlap_coverage(ib, M, shape),
        "independent_accuracy_px": acc,
        "accuracy_note": note,
    })
    return rec


def independent_accuracy(M_est, M_gt, n_inliers, shape, grid_n=12):
    """Probe-grid agreement between the estimated and true transform.

    Deterministic grid_n x grid_n probe points over the reference frame are
    mapped through M_est (A -> B, as returned by verify_guarded) and through
    M_gt (ref -> mov — the empirically verified convention of
    chandra_align.testing.make_pair_shift); the RMSE between the two mappings
    is the independent accuracy. Withheld (None + note) when the transform is
    missing or supported by < 3 inliers: accuracy is never faked from too few
    points.
    """
    if M_est is None or int(n_inliers) < 3:
        return None, "no independent accuracy available"
    h, w = shape
    ys = np.linspace(0, h - 1, grid_n)
    xs = np.linspace(0, w - 1, grid_n)
    yy, xx = np.meshgrid(ys, xs, indexing="ij")
    probe = np.stack([xx.ravel(), yy.ravel()], axis=1)
    Me = np.asarray(M_est, np.float64).reshape(2, 3)
    Mg = np.asarray(M_gt, np.float64).reshape(2, 3)
    p_est = probe @ Me[:, :2].T + Me[:, 2]
    p_gt = probe @ Mg[:, :2].T + Mg[:, 2]
    rmse = float(np.hypot(p_est[:, 0] - p_gt[:, 0],
                          p_est[:, 1] - p_gt[:, 1]).mean())
    return rmse, "probe-grid agreement with exact synthetic truth"
