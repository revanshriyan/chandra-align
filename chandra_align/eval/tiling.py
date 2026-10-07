"""Phase 13 — deterministic window tiling for the batch harness.

Generates non-overlapping windows on a fixed grid over a declared overlap
footprint, skipping windows that are mostly dark or flat. The module is
I/O-free: the caller supplies a ``read_window(y0, x0, size)`` callback
(typically backed by a numpy memmap), so the tiling logic itself is pure,
deterministic, and unit-testable.

Eligibility gates (documented, fixed — never tuned per pair):
  min_std:        windows with std below this are "flat" (featureless plains)
  max_dark_frac:  windows with more than this fraction at/below dark_value
                  are "dark" (missing data / shadow)
Both are reported per window so skips are auditable, not silent.
"""

from __future__ import annotations

import numpy as np


def window_stats(img):
    """Return (mean, std, dark_frac) for a window array.

    dark_frac is the fraction of pixels at or below 0 (missing/shadow).
    """
    a = np.asarray(img, dtype=np.float64).ravel()
    if a.size == 0:
        return 0.0, 0.0, 1.0
    finite = a[np.isfinite(a)]
    if finite.size == 0:
        return 0.0, 0.0, 1.0
    return (float(np.mean(finite)), float(np.std(finite)),
            float(np.mean(finite <= 0)))


def generate_windows(image_shape, window_size, region, seed=13,
                     min_std=8.0, max_dark_frac=0.5, read_window=None,
                     shuffle=False):
    """Deterministic non-overlapping window grid over ``region``.

    image_shape: (h, w) of the raster.
    window_size: square window edge in pixels.
    region: (y0, y1, x0, x1) overlap footprint, pixel coords, y1/x1 exclusive
        bounds for window origins (windows stay fully inside the raster).
    seed: accepted for interface stability; only used when shuffle=True.
    read_window: callable (y0, x0, size) -> ndarray, or None to skip the
        dark/flat eligibility gates (windows still get deterministic ids).
    shuffle: if True, window order is shuffled with the seed (default False:
        row-major order).

    Returns a list of dicts: window_id ("w0000"...), y0, x0, size, mean, std,
    dark_frac, eligible (bool), skip_reason ("" when eligible).
    """
    h, w = int(image_shape[0]), int(image_shape[1])
    size = int(window_size)
    ry0, ry1, rx0, rx1 = (int(v) for v in region)
    if not (0 <= ry0 < ry1 <= h and 0 <= rx0 < rx1 <= w):
        raise ValueError(f"region {region} outside raster {image_shape}")
    if size <= 0 or size > min(ry1 - ry0, rx1 - rx0):
        raise ValueError(f"window_size {size} does not fit region {region}")

    ys = list(range(ry0, ry1 - size + 1, size))
    xs = list(range(rx0, rx1 - size + 1, size))
    order = [(y, x) for y in ys for x in xs]
    if shuffle:
        rng = np.random.default_rng(int(seed))
        perm = rng.permutation(len(order))
        order = [order[i] for i in perm]

    windows = []
    for i, (y0, x0) in enumerate(order):
        rec = {"window_id": f"w{i:04d}", "y0": y0, "x0": x0, "size": size,
               "mean": None, "std": None, "dark_frac": None,
               "eligible": True, "skip_reason": ""}
        if read_window is not None:
            try:
                img = read_window(y0, x0, size)
            except Exception as exc:  # I/O failure is a skip, not a crash
                rec.update(eligible=False,
                           skip_reason=f"read_failed: {type(exc).__name__}")
                windows.append(rec)
                continue
            mean, std, dark_frac = window_stats(img)
            rec.update(mean=round(mean, 3), std=round(std, 3),
                       dark_frac=round(dark_frac, 4))
            if dark_frac > max_dark_frac:
                rec.update(eligible=False,
                           skip_reason=f"dark: {dark_frac:.2f} > {max_dark_frac}")
            elif std < min_std:
                rec.update(eligible=False,
                           skip_reason=f"flat: std {std:.2f} < {min_std}")
        windows.append(rec)
    return windows


def check_no_overlap(windows):
    """Return True iff no two windows' interiors intersect."""
    rects = [(r["y0"], r["x0"], r["y0"] + r["size"], r["x0"] + r["size"])
             for r in windows]
    for i in range(len(rects)):
        ay0, ax0, ay1, ax1 = rects[i]
        for j in range(i + 1, len(rects)):
            by0, bx0, by1, bx1 = rects[j]
            if ax0 < bx1 and bx0 < ax1 and ay0 < by1 and by0 < ay1:
                return False
    return True
