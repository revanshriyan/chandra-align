"""Polarity-invariant detection + NMI reranking (Phase 9 cross-modal branch).

Cross-modal pairs (IIRS vs TMC-2) can invert contrast: a bright rim in one
sensor is a dark rim in the other. Gradient-based detectors see edges
wherever |gradient| is large regardless of sign, so detecting on the
gradient-magnitude image is polarity-invariant by construction.

Candidate pairs are then scored with normalized mutual information, which
measures statistical dependence between patches — not pixel equality —
so it tolerates the nonlinear intensity mappings between sensors.

What this does NOT do: NMI is a *similarity score*, not a geometric
verdict. High NMI on a candidate pair does not mean the pair is a true
correspondence; RANSAC and the spatial gates still decide that.
"""

import cv2
import numpy as np


def _to_gray_float(img):
    img = np.asarray(img, dtype=np.float64)
    if img.ndim == 3:
        img = img.mean(axis=2)
    return img


def gradient_magnitude_points(img, n=300):
    """Harris corners on the gradient-magnitude image (polarity-invariant).

    Edges appear in |gradient| whether the step goes up or down, so the
    same physical features are detected in both polarities.

    Parameters
    ----------
    img : (H, W) array-like
    n : int, maximum number of points to return.

    Returns
    -------
    (N, 2) float64 array of (x, y) points, N <= n. Empty (0, 2) if the
    image has no usable structure.
    """
    gray = _to_gray_float(img)
    h, w = gray.shape
    gx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    mag = np.hypot(gx, gy).astype(np.float32)

    if not np.any(mag > 0):
        return np.zeros((0, 2), dtype=np.float64)

    resp = cv2.cornerHarris(mag, blockSize=2, ksize=3, k=0.04)
    # Non-maximum suppression via dilation, then rank by response.
    kernel = np.ones((3, 3), np.uint8)
    dilated = cv2.dilate(resp, kernel)
    peaks = (resp == dilated) & (resp > 0)
    ys, xs = np.nonzero(peaks)
    if xs.size == 0:
        return np.zeros((0, 2), dtype=np.float64)
    scores = resp[ys, xs]
    # Keep peaks above the 90th percentile of positive responses, top-n.
    thresh = np.percentile(scores, 90)
    keep = scores >= thresh
    xs, ys, scores = xs[keep], ys[keep], scores[keep]
    order = np.argsort(-scores, kind="stable")[: max(int(n), 0)]
    pts = np.stack([xs[order], ys[order]], axis=1).astype(np.float64)
    # Defensive clip: Harris can return border pixels on tiny images.
    pts[:, 0] = np.clip(pts[:, 0], 0, w - 1)
    pts[:, 1] = np.clip(pts[:, 1], 0, h - 1)
    return pts


def nmi_score(patch_a, patch_b, bins=32):
    """Normalized mutual information of two patches, in [0, 1].

    NMI = 2 * MI / (H_a + H_b). Identical patches score ~1; statistically
    independent patches score ~0. Returns 0.0 when either patch is
    constant (no information to share).
    """
    a = np.asarray(patch_a, dtype=np.float64).ravel()
    b = np.asarray(patch_b, dtype=np.float64).ravel()
    if a.size == 0 or b.size == 0 or a.size != b.size:
        raise ValueError("patches must be non-empty and the same size")
    lo = min(a.min(), b.min())
    hi = max(a.max(), b.max())
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        return 0.0  # constant patches: nothing to share
    edges = np.linspace(lo, hi, int(bins) + 1)
    joint, _, _ = np.histogram2d(a, b, bins=[edges, edges])
    joint = joint / joint.sum()
    pa = joint.sum(axis=1)
    pb = joint.sum(axis=0)

    def _entropy(p):
        p = p[p > 0]
        return float(-np.sum(p * np.log(p)))

    h_a, h_b = _entropy(pa), _entropy(pb)
    if h_a <= 0 or h_b <= 0:
        return 0.0
    ia, ib = np.nonzero(joint)
    jnz = joint[ia, ib]
    mi = float(np.sum(jnz * (np.log(jnz) - np.log(pa[ia]) - np.log(pb[ib]))))
    return float(np.clip(2.0 * mi / (h_a + h_b), 0.0, 1.0))


def _patch(img, x, y, window):
    h, w = img.shape[:2]
    r = int(window) // 2
    xi, yi = int(round(x)), int(round(y))
    x0, x1 = xi - r, xi + r + 1
    y0, y1 = yi - r, yi + r + 1
    if x0 < 0 or y0 < 0 or x1 > w or y1 > h:
        return None
    return img[y0:y1, x0:x1]


def nmi_rerank(cands_a, cands_b, img_a, img_b, window=21):
    """Score candidate pairs by NMI of windowed patches.

    Parameters
    ----------
    cands_a, cands_b : (K, 2) arrays of candidate (x, y) points, paired
        by index (pair k is cands_a[k] <-> cands_b[k]).
    img_a, img_b : single-channel images the points live in.
    window : int, odd patch side length.

    Returns
    -------
    (order, scores): `order` is the K indices sorted by descending NMI,
    `scores` the K NMI values in original candidate order. Pairs whose
    window falls outside the image score -1.0 and sort last.
    """
    cands_a = np.asarray(cands_a, dtype=np.float64)
    cands_b = np.asarray(cands_b, dtype=np.float64)
    if cands_a.shape != cands_b.shape or cands_a.ndim != 2 or cands_a.shape[1] != 2:
        raise ValueError("cands_a and cands_b must be matching (K,2) arrays")
    ga = _to_gray_float(img_a)
    gb = _to_gray_float(img_b)
    k = cands_a.shape[0]
    scores = np.full(k, -1.0, dtype=np.float64)
    for i in range(k):
        pa = _patch(ga, cands_a[i, 0], cands_a[i, 1], window)
        pb = _patch(gb, cands_b[i, 0], cands_b[i, 1], window)
        if pa is None or pb is None:
            continue
        scores[i] = nmi_score(pa, pb)
    order = np.argsort(-scores, kind="stable")
    return order, scores
