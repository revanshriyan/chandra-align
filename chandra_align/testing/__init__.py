"""Synthetic fixtures emulating the hard cases (build without real lunar data):

- Two overlapping crops with known sub-pixel translation/rotation (ground-truth
  transform known exactly).
- Same scene with simulated illumination change (gamma/gain/gradient shading) to
  emulate a Sun-angle difference.
- A 10x-20x downscaled reference pair (emulating OHRC<->NAC and TMC<->WAC ratios).
- A degenerate featureless pair that MUST trigger the "Not Trusted" flag.

All generators are deterministic (seeded) so tests are reproducible.
"""

import numpy as np


def _base_scene(h=512, w=512, seed=7):
    """Deterministic crater-and-ridge lunar-like scene with rich phase structure."""
    rng = np.random.default_rng(seed)
    y, x = np.mgrid[0:h, 0:w].astype(np.float64)
    img = np.zeros((h, w), np.float64)
    # ridge/ridge texture: sum of oriented sinusoids (phase congruency food)
    for freq, ang, amp in [(0.05, 0.4, 1.0), (0.09, 1.2, 0.7), (0.16, 2.1, 0.5),
                           (0.28, 0.9, 0.3), (0.045, 2.9, 0.8)]:
        img += amp * np.sin(freq * (x * np.cos(ang) + y * np.sin(ang)))
    # craters: dark rims + bright rims via radial profiles
    for _ in range(40):
        cy, cx = rng.uniform(10, h - 10), rng.uniform(10, w - 10)
        r = rng.uniform(3, 24)
        d = np.hypot(y - cy, x - cx)
        img -= 2.2 * np.exp(-((d - r) ** 2) / (2 * (r / 6) ** 2))   # dark rim
        img += 1.4 * np.exp(-((d - 1.4 * r) ** 2) / (2 * (r / 5) ** 2))  # bright ejecta
    # fine noise
    img += rng.normal(0, 0.08, (h, w))
    lo, hi = np.nanpercentile(img, [1, 99])
    img = np.clip((img - lo) / (hi - lo) * 255.0, 0, 255)
    return img


def make_pair_shift(shape=(512, 512), dx=7.3, dy=-3.9, angle_deg=0.0, seed=7):
    """Known ground-truth sub-pixel translation (and optional rotation) pair.

    Returns (img_ref, img_mov, gt_matrix) where gt_matrix maps moving->reference
    in pixel coords (2x3 affine, known exactly).
    """
    img = _base_scene(*shape, seed=seed)
    import cv2
    h, w = shape
    theta = np.radians(angle_deg)
    M = np.array([[np.cos(theta), -np.sin(theta), dx],
                  [np.sin(theta), np.cos(theta), dy]], np.float64)
    mov = cv2.warpAffine(img, M.astype(np.float32), (w, h),
                         flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    return img, mov, M


def make_pair_illumination(shape=(512, 512), dx=5.0, dy=2.0, gamma=1.6, gain=1.25,
                           gradient_strength=0.35, seed=7):
    """Same scene, simulated illumination change: gamma + gain + gradient shading
    (emulates a Sun-angle difference between acquisitions)."""
    ref, mov, M = make_pair_shift(shape, dx=dx, dy=dy, seed=seed)
    h, w = shape
    y, x = np.mgrid[0:h, 0:w].astype(np.float64)
    shade = 1.0 - gradient_strength * (x / w) - 0.5 * gradient_strength * (y / h)
    mov2 = gain * (np.power(np.clip(mov, 1, 255) / 255.0, gamma)) * 255.0 * shade
    mov2 = np.clip(mov2, 0, 255)
    return ref, mov2, M


def make_pair_scale(shape=(512, 512), ratio=10.0, dx=3.0, dy=3.0, seed=7):
    """Reference pair emulating OHRC<->NAC (2-4x) / TMC<->WAC (up to ~20x) ratios.

    Returns (img_ref_coarse, img_mov_fine, M) where the coarse reference is the fine
    scene downscaled by `ratio` then upscaled back to the same pixel grid
    (so 1 pixel in the coarse image = `ratio` pixels in the fine image).
    gt M maps fine->coarse-grid coords: scale then translate.
    """
    img = _base_scene(*shape, seed=seed)
    import cv2
    h, w = shape
    small = cv2.resize(img, (max(2, int(round(w / ratio))), max(2, int(round(h / ratio)))),
                       interpolation=cv2.INTER_AREA)
    coarse = cv2.resize(small, (w, h), interpolation=cv2.INTER_CUBIC)
    M = np.array([[1.0 / ratio, 0.0, dx / ratio],
                  [0.0, 1.0 / ratio, dy / ratio]], np.float64)
    mov = cv2.warpAffine(img, M.astype(np.float32), (w, h),
                         flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    return coarse, mov, M


def make_pair_degenerate(shape=(512, 512), seed=7):
    """Degenerate featureless pair (PSR-like: no optical texture). MUST fail."""
    rng = np.random.default_rng(seed)
    a = np.full(shape, 18.0) + rng.normal(0, 0.4, shape)
    b = np.full(shape, 21.0) + rng.normal(0, 0.4, shape)
    return a, b, None


def write_tiff(path, arr, crs="IAU_2015:30100", transform=None):
    """Write a float64 array as a GeoTIFF with the given lunar CRS."""
    import rasterio
    from rasterio.transform import from_origin

    if transform is None:
        transform = from_origin(0.0, float(arr.shape[0]), 1.0, 1.0)
    with rasterio.open(path, "w", driver="GTiff", height=arr.shape[0],
                       width=arr.shape[1], count=1, dtype="float64",
                       crs=crs, transform=transform) as dst:
        dst.write(arr, 1)
    return path
