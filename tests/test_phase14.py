"""Phase 14 tests — phase-congruency front-end (numpy/cv2 only, no torch)."""

import numpy as np
import pytest

from chandra_align.xmodal.phase_congruency import phase_congruency
from chandra_align.xmodal import pc_arm


def _textured(shape=(128, 128), seed=0):
    rng = np.random.default_rng(seed)
    x = np.linspace(0, 4 * np.pi, shape[1])
    y = np.linspace(0, 4 * np.pi, shape[0])
    xx, yy = np.meshgrid(x, y)
    img = (np.sin(xx) * np.cos(yy) + 0.5 * np.sin(3 * xx + yy)
           + 0.1 * rng.standard_normal(shape))
    return ((img - img.min()) / (img.max() - img.min()) * 255).astype(np.uint8)


def _corr(a, b):
    a = np.asarray(a, np.float64).ravel()
    b = np.asarray(b, np.float64).ravel()
    if a.std() < 1e-12 or b.std() < 1e-12:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])


def test_pc_identical_images_correlate():
    img = _textured()
    pc1, _ = phase_congruency(img)
    pc2, _ = phase_congruency(img)
    assert _corr(pc1, pc2) > 0.999


def test_pc_invariant_to_brightness_contrast_rescale():
    img = (_textured() * 0.35).astype(np.float64)  # headroom: no clipping
    rescaled = 2.0 * img + 30.0                    # stays in [30, ~108]
    assert rescaled.max() < 255
    pc1, _ = phase_congruency(img)
    pc2, _ = phase_congruency(rescaled)
    assert _corr(pc1, pc2) > 0.95


def test_pc_marks_structure_not_flat():
    img = _textured()
    pc, _ = phase_congruency(img)
    assert pc.max() > pc.mean() + pc.std()  # structure stands out
    flat = np.full((64, 64), 128.0)
    pc_flat, _ = phase_congruency(flat)
    assert pc_flat.max() < 1e-6  # no structure invented in flat regions


def test_pc_orientation_map_range():
    img = _textured()
    _, orient = phase_congruency(img)
    assert orient.min() >= 0.0 and orient.max() < np.pi


def test_pc_rejects_non_2d():
    with pytest.raises(ValueError):
        phase_congruency(np.zeros((8, 8, 3)))


def test_pc_arm_empty_on_blank():
    blank = np.full((128, 128), 128, np.uint8)
    pa, pb = pc_arm.match(blank, blank)
    assert pa.shape == (0, 2) and pb.shape == (0, 2)


def test_pc_arm_shifted_pair_finds_correspondences():
    from chandra_align.testing import make_pair_shift
    ref, mov, _ = make_pair_shift(shape=(256, 256), dx=5.0, dy=-2.0, seed=0)
    pa, pb = pc_arm.match(mov, ref)
    # textured synthetic pair: the arm must produce correspondences
    assert len(pa) > 10 and len(pa) == len(pb)
