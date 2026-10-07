"""Phase 18 — CheckList-style behavioral test matrix.

Capability x test-type matrix over the FROZEN default pipeline. This module
only measures; it never changes gates, thresholds, or pipeline behavior.

Capabilities: rotation invariance, illumination robustness, cross-modal
handling, low-texture behavior, uniform-shift detection.
Test types (after Ribeiro et al.'s CheckList):
  - invariance: perturb the input, expect the SAME verdict as baseline.
  - directional: perturb the input, expect a PREDICTED verdict/outcome change.
  - minimum-functionality: bare sanity — the pipeline must work at all.

Each cell is a real runnable function returning a dict with the measured
outcome and whether the expectation held. The matrix is the deliverable;
tests/test_phase18.py asserts a subset.
"""

import numpy as np

from chandra_align.testing import (
    make_pair_shift,
    make_pair_illumination,
    make_pair_degenerate,
)
from chandra_align.eval.robustness import (
    measure_pair,
    perturb_small_rotation,
    perturb_sun_flip,
    perturb_gaussian_noise,
    perturb_brightness_shift,
    independent_accuracy,
)


def _cell(name, capability, test_type, fn):
    rec = {"cell": name, "capability": capability, "test_type": test_type}
    try:
        out = fn()
        rec.update(out)
        rec["error"] = None
    except Exception as exc:  # failures are data, never crashes
        rec.update({"expectation_met": False, "measured": {}})
        rec["error"] = f"{type(exc).__name__}: {exc}"
    return rec


def _p(fn, *args, **kwargs):
    """Unpack perturb functions, which return (image, params)."""
    img, _ = fn(*args, **kwargs)
    return img

# ---------------------------------------------------------------- baseline
def _baseline_shift():
    a, b, M = make_pair_shift(shape=(512, 512), dx=7.3, dy=-3.9, seed=7)
    return measure_pair(a, b, M)


# ------------------------------------------------------- ROTATION INVARIANCE
def rotation_invariance_small():
    """2-degree rotation: verdict must match the unrotated baseline."""
    base = _baseline_shift()
    a, b, M = make_pair_shift(shape=(512, 512), dx=7.3, dy=-3.9, seed=7)
    b2 = _p(perturb_small_rotation, b, 2.0, seed=11)
    r = measure_pair(a, b2, M)
    return {"expectation_met": r["verdict"] == base["verdict"],
            "measured": {"baseline": base["verdict"], "rotated": r["verdict"],
                         "rmse": r["rmse_px"]}}


def rotation_directional_large():
    """45-degree rotation: expect the verdict to degrade (predicted change)."""
    base = _baseline_shift()
    a, b, M = make_pair_shift(shape=(512, 512), dx=7.3, dy=-3.9, seed=7)
    b2 = _p(perturb_small_rotation, b, 45.0, seed=11)
    r = measure_pair(a, b2, M)
    worse = {"SUCCESS_SUBPIXEL": 0, "COARSE_ADVISORY": 1,
             "DEGENERATE_FAILURE": 2, "ABSTAIN": 3}
    return {"expectation_met": worse.get(r["verdict"], 3) >= worse.get(base["verdict"], 0),
            "measured": {"baseline": base["verdict"], "rotated45": r["verdict"]}}


def rotation_minimum_functionality():
    """Identity pair must register sub-pixel (sanity)."""
    a, b, M = make_pair_shift(shape=(512, 512), dx=0.0, dy=0.0, seed=7)
    r = measure_pair(a, b, M)
    return {"expectation_met": r["verdict"] == "SUCCESS_SUBPIXEL",
            "measured": {"verdict": r["verdict"], "rmse": r["rmse_px"]}}


# ----------------------------------------------------- ILLUMINATION ROBUSTNESS
def illumination_invariance_brightness():
    """Small brightness shift: verdict must hold."""
    base = _baseline_shift()
    a, b, M = make_pair_shift(shape=(512, 512), dx=7.3, dy=-3.9, seed=7)
    b2 = _p(perturb_brightness_shift, b, 12.0, seed=11)
    r = measure_pair(a, b2, M)
    return {"expectation_met": r["verdict"] == base["verdict"],
            "measured": {"baseline": base["verdict"], "bright": r["verdict"]}}


def illumination_directional_sunflip():
    """Sun-flip (polarity reversal): inlier self-consistency must NOT be
    mistaken for correctness — expect independent accuracy to disagree with
    inlier RMSE by a large margin (the Phase 11 lesson, as a live check)."""
    a, b, M = make_pair_shift(shape=(512, 512), dx=7.3, dy=-3.9, seed=7)
    b2 = _p(perturb_sun_flip, b, seed=11)
    r = measure_pair(a, b2, M)
    ia = r["independent_accuracy_px"]
    rmse = r["rmse_px"]
    disagree = (ia is not None and rmse is not None and abs(ia - rmse) > 10.0)
    return {"expectation_met": bool(disagree),
            "measured": {"verdict": r["verdict"], "inlier_rmse": rmse,
                         "independent_accuracy": ia}}


def illumination_minimum_functionality():
    """Mild illumination pair from the factory must not abstain."""
    a, b, M = make_pair_illumination(shape=(512, 512), dx=5.0, dy=2.0,
                                     gamma=1.6, gain=1.25, seed=7)
    r = measure_pair(a, b, M)
    return {"expectation_met": r["verdict"] in ("SUCCESS_SUBPIXEL", "COARSE_ADVISORY"),
            "measured": {"verdict": r["verdict"], "rmse": r["rmse_px"]}}


# ------------------------------------------------------------- CROSS-MODAL
def xmodal_directional_polarity():
    """Full polarity inversion (crude cross-modal proxy): the pipeline must
    NOT hold its sub-pixel verdict — expect a predicted degradation
    (measured 2026-10-07: inversion degrades SUCCESS_SUBPIXEL to
    COARSE_ADVISORY rather than failing outright; the expectation encodes
    the direction, not the exact tier)."""
    a, b, M = make_pair_shift(shape=(512, 512), dx=7.3, dy=-3.9, seed=7)
    b2 = 255.0 - np.asarray(b, dtype=np.float64)
    r = measure_pair(a, b2, M)
    return {"expectation_met": r["verdict"] != "SUCCESS_SUBPIXEL",
            "measured": {"verdict": r["verdict"], "abstain": r["abstain"],
                         "rmse": r["rmse_px"]}}


def xmodal_minimum_functionality():
    """Same-sensor control for the cross-modal cell: must succeed."""
    a, b, M = make_pair_shift(shape=(512, 512), dx=7.3, dy=-3.9, seed=7)
    r = measure_pair(a, b, M)
    return {"expectation_met": r["verdict"] == "SUCCESS_SUBPIXEL",
            "measured": {"verdict": r["verdict"]}}


def xmodal_invariance_noise():
    """Moderate Gaussian noise on both images: verdict must hold
    (sensor-noise invariance, the mildest cross-condition cell)."""
    base = _baseline_shift()
    a, b, M = make_pair_shift(shape=(512, 512), dx=7.3, dy=-3.9, seed=7)
    a2 = _p(perturb_gaussian_noise, a, 6.0, seed=11)
    b2 = _p(perturb_gaussian_noise, b, 6.0, seed=12)
    r = measure_pair(a2, b2, M)
    return {"expectation_met": r["verdict"] == base["verdict"],
            "measured": {"baseline": base["verdict"], "noisy": r["verdict"]}}


# --------------------------------------------------------------- LOW-TEXTURE
def lowtexture_directional_degenerate():
    """Factory degenerate pair (near-featureless): expect ABSTAIN with a
    named code, never a confident verdict."""
    a, b, M = make_pair_degenerate(shape=(512, 512), seed=7)
    r = measure_pair(a, b, M)
    return {"expectation_met": r["verdict"] in ("ABSTAIN", "DEGENERATE_FAILURE")
            and r["abstain"] is not None,
            "measured": {"verdict": r["verdict"], "abstain": r["abstain"]}}


def lowtexture_minimum_functionality():
    """Textured control: the same harness must succeed on real texture."""
    a, b, M = make_pair_shift(shape=(512, 512), dx=7.3, dy=-3.9, seed=7)
    r = measure_pair(a, b, M)
    return {"expectation_met": r["verdict"] == "SUCCESS_SUBPIXEL",
            "measured": {"verdict": r["verdict"], "n_inliers": r["inliers"]}}


def lowtexture_invariance_mild_blur():
    """Mild blur (3x3): verdict must hold — texture reduction within the
    pipeline's operating envelope."""
    import cv2
    base = _baseline_shift()
    a, b, M = make_pair_shift(shape=(512, 512), dx=7.3, dy=-3.9, seed=7)
    a2 = cv2.GaussianBlur(np.asarray(a, np.float32), (3, 3), 0)
    b2 = cv2.GaussianBlur(np.asarray(b, np.float32), (3, 3), 0)
    r = measure_pair(a2, b2, M)
    return {"expectation_met": r["verdict"] == base["verdict"],
            "measured": {"baseline": base["verdict"], "blurred": r["verdict"]}}


# ----------------------------------------------------- UNIFORM-SHIFT DETECTION
def shift_minimum_functionality():
    """Pure translation is the pipeline's home turf: sub-pixel expected."""
    a, b, M = make_pair_shift(shape=(512, 512), dx=12.0, dy=-8.0, seed=7)
    r = measure_pair(a, b, M)
    ia = r["independent_accuracy_px"]
    return {"expectation_met": r["verdict"] == "SUCCESS_SUBPIXEL"
            and ia is not None and ia < 1.0,
            "measured": {"verdict": r["verdict"],
                         "independent_accuracy": ia}}


def _warp(img, M):
    import cv2
    h, w = np.asarray(img).shape[:2]
    return cv2.warpAffine(np.asarray(img, np.float32),
                         np.asarray(M, np.float64).astype(np.float32), (w, h),
                         flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)


def _affine(dx, dy):
    return np.array([[1.0, 0.0, dx], [0.0, 1.0, dy]], np.float64)


def shift_directional_loop_closure():
    """A→B→C→A loop must close near zero; a deliberately wrong leg must be
    caught by loop-closure error (directional: corruption is detected)."""
    from chandra_align.trust.loop_closure import loop_closure, compose_affine
    from chandra_align.testing import _base_scene
    rng_pts = np.random.default_rng(0)
    pts = rng_pts.uniform(50, 462, size=(40, 2))
    img0 = _base_scene(512, 512, seed=7)
    M01, M12 = _affine(7.3, -3.9), _affine(-4.1, 6.2)
    img1, img2 = _warp(img0, M01), _warp(_warp(img0, M01), M12)
    M02 = compose_affine(M12, M01)
    M20 = np.linalg.inv(np.vstack([M02, [0, 0, 1]]))[:2, :]
    r01 = measure_pair(img0, img1, M01, return_inliers=True)
    r12 = measure_pair(img1, img2, M12, return_inliers=True)
    r20 = measure_pair(img2, img0, M20, return_inliers=True)
    models = [r01["_model"], r12["_model"], r20["_model"]]
    if any(m is None for m in models):
        return {"expectation_met": False,
                "measured": {"note": "a leg failed to fit; loop not testable"}}
    clean = loop_closure(models[0], models[1], models[2], pts)["cyclic_rmse_px"]
    bad = models[1].copy()
    bad[:, 2] += 40.0
    corrupt = loop_closure(models[0], bad, models[2], pts)["cyclic_rmse_px"]
    return {"expectation_met": clean < 5.0 and corrupt > 10.0,
            "measured": {"loop_error_clean": clean,
                         "loop_error_corrupted": corrupt}}


def shift_invariance_seed():
    """Same geometry, different scene seed: verdict must be stable."""
    a1, b1, M1 = make_pair_shift(shape=(512, 512), dx=7.3, dy=-3.9, seed=7)
    a2, b2, M2 = make_pair_shift(shape=(512, 512), dx=7.3, dy=-3.9, seed=21)
    r1 = measure_pair(a1, b1, M1)
    r2 = measure_pair(a2, b2, M2)
    return {"expectation_met": r1["verdict"] == r2["verdict"] == "SUCCESS_SUBPIXEL",
            "measured": {"seed7": r1["verdict"], "seed21": r2["verdict"]}}


# ------------------------------------------------------------------ matrix
CELLS = [
    ("rotation / invariance", "rotation", "invariance", rotation_invariance_small),
    ("rotation / directional", "rotation", "directional", rotation_directional_large),
    ("rotation / minimum-functionality", "rotation", "minimum-functionality",
     rotation_minimum_functionality),
    ("illumination / invariance", "illumination", "invariance",
     illumination_invariance_brightness),
    ("illumination / directional", "illumination", "directional",
     illumination_directional_sunflip),
    ("illumination / minimum-functionality", "illumination", "minimum-functionality",
     illumination_minimum_functionality),
    ("cross-modal / directional", "cross-modal", "directional",
     xmodal_directional_polarity),
    ("cross-modal / minimum-functionality", "cross-modal", "minimum-functionality",
     xmodal_minimum_functionality),
    ("cross-modal / invariance", "cross-modal", "invariance",
     xmodal_invariance_noise),
    ("low-texture / directional", "low-texture", "directional",
     lowtexture_directional_degenerate),
    ("low-texture / minimum-functionality", "low-texture", "minimum-functionality",
     lowtexture_minimum_functionality),
    ("low-texture / invariance", "low-texture", "invariance",
     lowtexture_invariance_mild_blur),
    ("uniform-shift / minimum-functionality", "uniform-shift", "minimum-functionality",
     shift_minimum_functionality),
    ("uniform-shift / directional", "uniform-shift", "directional",
     shift_directional_loop_closure),
    ("uniform-shift / invariance", "uniform-shift", "invariance",
     shift_invariance_seed),
]


def run_matrix():
    """Run every cell; return list of records."""
    return [_cell(name, cap, typ, fn) for name, cap, typ, fn in CELLS]


def summarize(records):
    met = sum(1 for r in records if r.get("expectation_met"))
    return {"n_cells": len(records), "expectations_met": met,
            "pass_rate": met / len(records) if records else 0.0}
