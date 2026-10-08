"""A24: deformation-field x illumination — does the stage help cross-sun pairs?

Takes synthetic sun-angle sweep cases at the interesting boundary (reconstructed
from the documented recipe in docs/sunangle-full-sweep-2026-10-08.md) and runs
each through the TRUE app.py::_align_core flag OFF and flag ON
(CHANDRA_DEFORM_FIELD=1).

The sweep showed the pipeline degrades with azimuth difference (photometric
failure: shadows move). The deform-field stage fixes spatially-varying
geometric distortion. Question: does it help when the failure is photometric?

Pre-registered bars:
 (a) a "win" = verdict upgrade (DEGENERATE->COARSE or COARSE->SUCCESS) on >=2
     cases with zero downgrades, else clean negative;
 (b) overfit check: recovered transform vs KNOWN identity truth on an
     independent grid (Phase 11 lesson: inlier RMSE != correctness);
 (c) determinism spot-check.

Experiment only: no app.py / chandra_align changes. Deterministic (seed 7).
"""
import csv
import os
import sys

import cv2
import numpy as np
import tifffile

sys.path.insert(0, "/tmp/stubs")
sys.path.insert(0, "/home/hatch/workspace/chandra-align")

import app  # noqa: E402

DTM = "/home/hatch/workspace/lroc_data/NAC_DTM_VIKRAMSITE1.TIF"
# Sweep recipe: 500x500 crop, rows 7697-8197, cols 3584-4084
R0, R1, C0, C1 = 7697, 8197, 3584, 4084

# (name, elevation_deg, azimuth_deg); reference is always az=0, el=30
CASES = [
    ("el30_az060", 30.0, 60.0),    # COARSE in sweep (58 inl, 1.16px)
    ("el30_az090", 30.0, 90.0),    # COARSE in sweep (16 inl, 1.28px), near boundary
    ("el30_az105", 30.0, 105.0),   # DEGENERATE in sweep (0 inl), failure boundary
    ("el50_az090", 50.0, 90.0),    # COARSE in sweep (49 inl, 1.17px), survives
    ("el10_az030", 10.0, 30.0),    # DEGENERATE in sweep (0 inl)
]
REF_AZ, REF_EL = 0.0, 30.0


def load_dem():
    with tifffile.TiffFile(DTM) as tif:
        z = tif.asarray()[R0:R1, C0:C1].astype(np.float64)
    assert z.shape == (500, 500), z.shape
    # Inpaint nodata (-3.4e38) — small patch, must not corrupt gradients
    bad = z < -1e30
    if bad.any():
        zc = np.where(bad, np.nan, z)
        # fill with nearest valid via OpenCV inpaint on 8-bit proxy
        znorm = ((zc - np.nanmin(zc)) / (np.nanmax(zc) - np.nanmin(zc)) * 255)
        znorm = np.where(bad, 0, znorm).astype(np.uint8)
        mask = bad.astype(np.uint8) * 255
        inp = cv2.inpaint(znorm, mask, 3, cv2.INPAINT_TELEA).astype(np.float64)
        z = np.where(bad,
                     inp / 255.0 * (np.nanmax(zc) - np.nanmin(zc)) + np.nanmin(zc),
                     zc)
    return z


def render(z, albedo, az_deg, el_deg, px_size_m=3.0):
    """Lambertian render: max(0, n.s) * albedo. Azimuth clockwise from north."""
    gy, gx = np.gradient(z, px_size_m)  # DEM in meters, pixel spacing in meters
    # surface normal (x=east, y=north->image row down, z=up)
    n = np.dstack([-gx, gy, np.ones_like(z)])
    n /= np.linalg.norm(n, axis=2, keepdims=True)
    az = np.deg2rad(az_deg)
    el = np.deg2rad(el_deg)
    # sun vector: azimuth clockwise from north, elevation above horizon
    s = np.array([np.sin(az) * np.cos(el), np.cos(az) * np.cos(el), np.sin(el)])
    cos_i = np.clip(np.einsum("ijk,k->ij", n, s), 0.0, None)
    return (cos_i * albedo).astype(np.float64)


def build_renders(albedo_amp=0.30):
    z = load_dem()
    rng = np.random.default_rng(7)
    noise = rng.standard_normal(z.shape)
    # Note: sweep recipe says 0.12, but after σ=6 smoothing the effective
    # variation is ~±1%, too weak on this smooth terrain (mean slope 4.4°)
    # for cross-sun matching. Using 0.30 to create matchable texture while
    # keeping the illumination-dependent shading dominant.
    albedo = 1.0 + albedo_amp * cv2.GaussianBlur(noise.astype(np.float32), (0, 0), 6.0)
    raw = {}
    raw["ref"] = render(z, albedo, REF_AZ, REF_EL)
    for name, el, az in CASES:
        raw[name] = render(z, albedo, az, el)
    # common 1st-99th percentile stretch across all renders (sweep recipe)
    stack = np.concatenate([v.ravel() for v in raw.values()])
    lo, hi = np.percentile(stack, [1, 99])
    out = {}
    for k, v in raw.items():
        u8 = np.clip((v - lo) / max(hi - lo, 1e-9), 0, 1)
        out[k] = (u8 * 255).astype(np.uint8)
    return out


def truth_rmse(H):
    """Identity-truth RMSE on an independent 9-point grid (0.15/0.5/0.85)."""
    if H is None:
        return None
    H = np.asarray(H, dtype=np.float64)
    if H.shape != (3, 3):
        return None
    g = np.array([0.15, 0.5, 0.85]) * 500.0
    xx, yy = np.meshgrid(g, g)
    pts = np.stack([xx.ravel(), yy.ravel(), np.ones(9)], axis=1)
    mapped = (H @ pts.T).T
    w = mapped[:, 2:3]
    w = np.where(np.abs(w) < 1e-12, 1e-12, w)
    mapped = mapped[:, :2] / w
    err = np.linalg.norm(mapped - pts[:, :2], axis=1)
    return float(np.sqrt(np.mean(err ** 2)))


def run_pair(ref, sec, flag):
    old = os.environ.get("CHANDRA_DEFORM_FIELD")
    try:
        if flag:
            os.environ["CHANDRA_DEFORM_FIELD"] = "1"
        else:
            os.environ.pop("CHANDRA_DEFORM_FIELD", None)
        cv2.setRNGSeed(7)
        np.random.seed(7)
        out = app._align_core(ref, sec, pixel_scale_m=0.25,
                              reference_sensor_name="OHRC")
    except ValueError as exc:
        # Pipeline fail-closed: insufficient correspondences -> no fit possible.
        # This is the pipeline's DEGENERATE verdict for "can't match at all".
        return {
            "status_code": "DEGENERATE_FAILURE",
            "n_inliers": 0,
            "rmse_gate_px": None,
            "rmse_in_sample_px": None,
            "truth_rmse_px": None,
            "entropy": None,
            "quadrants": 0,
            "df_applied": False,
            "df_reason": "fail_closed_no_correspondences",
            "df_lambda": None,
            "df_heldout_px": None,
            "df_min_jac_det": None,
            "fail_closed_note": str(exc)[:100],
        }, None
    except Exception as exc:
        return None, "exception:%s:%s" % (type(exc).__name__, str(exc)[:100])
    finally:
        if old is None:
            os.environ.pop("CHANDRA_DEFORM_FIELD", None)
        else:
            os.environ["CHANDRA_DEFORM_FIELD"] = old
    jm = out.get("judge_metrics", {}) or {}
    df = jm.get("deform_field_stage", {}) or {}
    gm = jm.get("global_metrics", {}) or {}
    H = out.get("H")
    return {
        "status_code": out.get("status_code"),
        "n_inliers": out.get("inlier_cnt", gm.get("inlier_count")),
        "rmse_gate_px": jm.get("rmse_gate_px"),
        "rmse_in_sample_px": jm.get("rmse_in_sample_px"),
        "truth_rmse_px": truth_rmse(H),
        "entropy": out.get("quadrant_spatial_entropy", gm.get("spatial_entropy_score")),
        "quadrants": jm.get("active_quadrants_count"),
        "df_applied": df.get("applied"),
        "df_reason": df.get("reason"),
        "df_lambda": df.get("lambda_chosen"),
        "df_heldout_px": df.get("heldout_rmse_px"),
        "df_min_jac_det": df.get("min_jacobian_det"),
    }, None


def main():
    renders = build_renders()
    ref = renders["ref"]
    rows = []
    for name, el, az in CASES:
        sec = renders[name]
        for flag in (False, True):
            res, err = run_pair(ref, sec, flag)
            row = {
                "case": name, "elevation_deg": el, "azimuth_deg": az,
                "flag": "on" if flag else "off",
            }
            if err:
                row["error"] = err
            else:
                row.update({k: ("" if v is None else v) for k, v in res.items()})
            rows.append(row)
            print("%s flag=%s -> %s" % (
                name, row["flag"],
                err if err else "%s inl=%s gate=%.3s truth=%.3s df=%s" % (
                    res["status_code"], res["n_inliers"],
                    res["rmse_gate_px"], res["truth_rmse_px"],
                    res["df_applied"])))
    # determinism spot-check: re-run one case flag-on (use el10_az030 which produces a fit)
    res2, _ = run_pair(ref, renders["el10_az030"], True)
    # rows[8] = el10_az030 flag-off, rows[9] = el10_az030 flag-on
    r9 = rows[9]
    det = "IDENTICAL" if (
        res2 and r9["status_code"] == res2["status_code"]
        and str(r9["n_inliers"]) == str(res2["n_inliers"])
        and str(r9["rmse_gate_px"] or "") == str(res2["rmse_gate_px"] or "")) else "MISMATCH"
    print("determinism spot-check (el10_az030 flag-on re-run):", det)

    fields = ["case", "elevation_deg", "azimuth_deg", "flag", "status_code",
              "n_inliers", "rmse_in_sample_px", "rmse_gate_px", "truth_rmse_px",
              "entropy", "quadrants", "df_applied", "df_reason", "df_lambda",
              "df_heldout_px", "df_min_jac_det", "fail_closed_note", "error"]
    with open("/home/hatch/workspace/chandra-align/results/table_field_illum.csv",
              "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in fields})
    print("wrote results/table_field_illum.csv")
    return det


if __name__ == "__main__":
    main()
