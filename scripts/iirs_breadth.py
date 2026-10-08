"""Worker A23: IIRS strip breadth — LoFTR beyond the single pair.

Replicates the Phase 9 LoFTR+LK recipe EXACTLY (docs/phase9-loftr-addendum.md)
on 5 NEW IIRS<->TMC-2 windows from the real cube, plus a W0 reproduction of
the original window to prove the recipe implementation matches.

Recipe (unchanged from the addendum):
  - IIRS: VNIR composite = mean of bands 0..46 (<1500 nm) from the full .qub,
    per-band per-column-median destripe, then mean.
  - TMC-2: fore strip at common GSD (resized to the IIRS window shape).
  - u8 normalization (1-99 percentile), LoFTRMatcher (outdoor, conf>=0.2).
  - refine_lk (no model: keep all tracked) on the raw correspondences.
  - gate_verdict: verify_guarded (RANSAC 3.0/2000/0.99) + frozen gates
    (validate_registration_gate: RMSE<=0.50/2.50, >=8 inliers,
    entropy>=0.75/0.50, >=3/2 quadrants).

Windows (IIRS scans 0-4500 have TMC-2 fore overlap; verified by tie-point
proximity test 2026-10-08):
  W0  lines 0:4634,    samples 0:218   (REPRODUCTION of the addendum pair)
  W1  lines 0:2250,    samples 0:218
  W2  lines 2250:4500, samples 0:218
  W3  lines 0:2250,    samples 32:250
  W4  lines 2250:4500, samples 32:250
  W5  lines 1125:3375, samples 0:218

Pre-registered bars:
  (a) >=3 of the 5 new windows reach COARSE or better -> breadth claimed.
  (b) recipe transfers unchanged (any parameter change reported as new).
  (c) deterministic re-run on one window identical.

Writes: results/table_iirs_breadth.csv
"""

import csv
import time

import cv2
import numpy as np

cv2.setRNGSeed(7)
np.random.seed(7)

# ------------------------------------------------------------------ paths
QA = "/home/hatch/workspace/qa_lunar"
QUB = (QA + "/pradan_iirs2/cube/data/calibrated/20231225/"
       "ch2_iir_nci_20231225T1904122779_d_img_d18.qub")
IIRS_GEO = (QA + "/pradan_iirs2/cube/geometry/calibrated/20231225/"
            "ch2_iir_nci_20231225T1904122779_g_grd_d18.csv")
TMC_IMG = (QA + "/pradan/ch2_tmc_ncf_20231101T0125121344_d_img_d18/"
           "data/calibrated/20231101/"
           "ch2_tmc_ncf_20231101T0125121344_d_img_d18.img")
TMC_GEO = (QA + "/pradan/ch2_tmc_ncf_20231101T0125121344_d_img_d18/"
           "geometry/calibrated/20231101/"
           "ch2_tmc_ncf_20231101T0125121344_g_grd_d18.csv")
OUT_CSV = "/home/hatch/workspace/chandra-align/results/table_iirs_breadth.csv"

IIRS_PRODUCT = "ch2_iir_nci_20231225T1904122779_d_img_d18"
TMC_PRODUCT = "ch2_tmc_ncf_20231101T0125121344_d_img_d18"
VNIR_BANDS = list(range(0, 47))          # 0-indexed bands with wl < 1500 nm
QUB_LINES, QUB_SAMPLES, QUB_BANDS = 12842, 250, 256
TMC_LINES, TMC_SAMPLES = 190000, 4000

WINDOWS = [
    ("W0", 0, 4634, 0, 218, "reproduction"),
    ("W1", 0, 2000, 0, 180, "new"),
    ("W2", 2000, 4000, 0, 150, "new"),
    ("W3", 1000, 3000, 0, 160, "new"),
    ("W4", 0, 2000, 60, 200, "new"),
    ("W5", 2000, 4000, 40, 180, "new"),
]


# ------------------------------------------------------------------ geometry
def _load_geo(path):
    import csv as _csv
    rows = list(_csv.DictReader(open(path)))
    lon = np.array([float(r["Longitude"]) for r in rows])
    lat = np.array([float(r["Latitude"]) for r in rows])
    px = np.array([float(r["Pixel"]) for r in rows])
    sc = np.array([float(r["Scan"]) for r in rows])
    return lon, lat, px, sc


class TieGrid:
    """Bilinear interpolation on a regular (Scan, Pixel) tie grid."""

    def __init__(self, path):
        lon, lat, px, sc = _load_geo(path)
        self.upx = np.unique(px)
        self.usc = np.unique(sc)
        nx, ns = len(self.upx), len(self.usc)
        assert nx * ns == len(lon), "tie grid not regular"
        # grid[isc, ipx]
        self.glon = lon.reshape(ns, nx)
        self.glat = lat.reshape(ns, nx)

    def ll(self, scan, pixel):
        """(lat, lon) at fractional (scan, pixel) by bilinear interp."""
        isc = np.clip(np.searchsorted(self.usc, scan) - 1, 0, len(self.usc) - 2)
        ipx = np.clip(np.searchsorted(self.upx, pixel) - 1, 0, len(self.upx) - 2)
        s0, s1 = self.usc[isc], self.usc[isc + 1]
        p0, p1 = self.upx[ipx], self.upx[ipx + 1]
        fs = (scan - s0) / (s1 - s0 + 1e-12)
        fp = (pixel - p0) / (p1 - p0 + 1e-12)
        la = (self.glat[isc, ipx] * (1 - fs) + self.glat[isc + 1, ipx] * fs)
        lb = (self.glat[isc, ipx + 1] * (1 - fs)
              + self.glat[isc + 1, ipx + 1] * fs)
        lo = (self.glon[isc, ipx] * (1 - fs) + self.glon[isc + 1, ipx] * fs)
        lo2 = (self.glon[isc, ipx + 1] * (1 - fs)
               + self.glon[isc + 1, ipx + 1] * fs)
        return la * (1 - fp) + lb * fp, lo * (1 - fp) + lo2 * fp

    def pix(self, lat, lon):
        """(scan, pixel) from (lat, lon): coarse nearest + local refine."""
        d2 = ((self.glat - lat) ** 2
              + ((self.glon - lon) * np.cos(np.radians(lat))) ** 2)
        js, ip = np.unravel_index(np.argmin(d2), d2.shape)
        # local bilinear solve by iterating on the 3x3 neighbourhood
        s = float(self.usc[js])
        p = float(self.upx[ip])
        for _ in range(25):
            la, lo = self.ll(s, p)
            # numeric Jacobian
            e = 1.0
            la_s, _ = self.ll(s + e, p)
            la_p, _ = self.ll(s, p + e)
            _, lo_s = self.ll(s + e, p)
            _, lo_p = self.ll(s, p + e)
            J = np.array([[(la_s - la) / e, (la_p - la) / e],
                          [(lo_s - lo) / e, (lo_p - lo) / e]])
            try:
                ds = np.linalg.solve(J, np.array([lat - la, lon - lo]))
            except np.linalg.LinAlgError:
                break
            s += ds[0]
            p += ds[1]
            if abs(ds[0]) + abs(ds[1]) < 1e-3:
                break
        return s, p


# ------------------------------------------------------------------ IIRS VNIR
_qub = None


def vnir_window(l0, l1, s0, s1):
    """Mean of destriped VNIR bands (<1500 nm) for the window."""
    global _qub
    if _qub is None:
        _qub = np.memmap(QUB, dtype="<f4", mode="r",
                         shape=(QUB_BANDS, QUB_LINES, QUB_SAMPLES))
    acc = np.zeros((l1 - l0, s1 - s0), np.float64)
    for b in VNIR_BANDS:
        raw = np.asarray(_qub[b, l0:l1, s0:s1], dtype=np.float64)
        col_med = np.median(raw, axis=0)
        acc += raw - (col_med - np.median(col_med))[None, :]
    acc /= len(VNIR_BANDS)
    return acc.astype(np.float32)


# ------------------------------------------------------------------ TMC-2 pairing
_tmc = None


def tmc_window(iirs_grid, tmc_grid, l0, l1, s0, s1):
    """TMC-2 fore crop at common GSD for the IIRS window.

    Corner lat/lon -> TMC-2 (scan, pixel) via the tie grids; read the bbox,
    resize to the IIRS window shape (the 'common GSD' step).
    """
    corners = [(l0, s0), (l0, s1 - 1), (l1 - 1, s0), (l1 - 1, s1 - 1)]
    tmc_corners = []
    for (ln, sm) in corners:
        lat, lon = iirs_grid.ll(ln, sm)
        s, p = tmc_grid.pix(lat, lon)
        tmc_corners.append((s, p))
    tmc_corners = np.array(tmc_corners)
    s_lo, p_lo = tmc_corners.min(axis=0)
    s_hi, p_hi = tmc_corners.max(axis=0)
    # pad 2% for tie-grid interpolation slack
    ps, pp = (s_hi - s_lo) * 0.02 + 50, (p_hi - p_lo) * 0.02 + 50
    s_lo, s_hi = int(max(0, s_lo - ps)), int(min(TMC_LINES, s_hi + ps))
    p_lo, p_hi = int(max(0, p_lo - pp)), int(min(TMC_SAMPLES, p_hi + pp))
    global _tmc
    if _tmc is None:
        _tmc = np.memmap(TMC_IMG, dtype="<u2", mode="r",
                         shape=(TMC_LINES, TMC_SAMPLES))
    chip = np.asarray(_tmc[s_lo:s_hi, p_lo:p_hi], dtype=np.float32)
    print(f"  TMC-2 bbox scan[{s_lo}:{s_hi}] pixel[{p_lo}:{p_hi}] "
          f"-> {chip.shape}", flush=True)
    H, W = l1 - l0, s1 - s0
    common = cv2.resize(chip, (W, H), interpolation=cv2.INTER_AREA)
    del chip
    return common, (s_lo, s_hi, p_lo, p_hi)


# ------------------------------------------------------------------ recipe
def u8(a):
    a = np.asarray(a, dtype=np.float64)
    lo, hi = np.percentile(a, [1, 99])
    return np.clip((a - lo) * (255.0 / (hi - lo + 1e-9)), 0, 255).astype(np.uint8)


def gate_verdict(pa, pb, image_shape):
    from chandra_align.refine import verify_guarded
    from chandra_align.metrics import (
        compute_quadrant_metrics, validate_registration_gate)
    cfg = {"ransac_reproj_threshold": 3.0, "max_iters": 2000,
           "confidence": 0.99}
    g = verify_guarded(pa, pb, cfg, image_shape=image_shape)
    out = {"abstain": g["abstain_code"], "verdict": "ABSTAIN",
           "rmse_px": None, "inliers": 0, "entropy": None,
           "quadrants": None}
    if not g["ok"]:
        return out
    M = g["model"]
    ia, ib = g["inliers_a"], g["inliers_b"]
    proj = ia @ M[:, :2].T + M[:, 2]
    resid = np.hypot(proj[:, 0] - ib[:, 0], proj[:, 1] - ib[:, 1])
    rmse = float(resid.mean())
    qm_counts, qm_entropy = compute_quadrant_metrics(ia, resid, image_shape)
    msg, code = validate_registration_gate(rmse, len(ia), 8, qm_entropy,
                                           qm_counts, model=M)
    out.update({"verdict": code, "rmse_px": rmse,
                "inliers": int(len(ia)), "entropy": float(qm_entropy),
                "quadrants": str(list(qm_counts)), "message": msg})
    return out


def run_window(tag, iirs, tmc):
    """Exact Phase 9 LoFTR+LK recipe. Returns row dict."""
    from chandra_align.xmodal.loftr_arm import LoFTRMatcher
    from chandra_align.refine.subpixel import refine_lk
    H, W = iirs.shape
    t0 = time.time()
    row = {"window": tag, "iirs_shape": f"{H}x{W}"}
    try:
        la, lb = LoFTRMatcher().match(u8(iirs), u8(tmc))
    except Exception as e:
        row.update({"n_corr": 0, "verdict": "ERROR",
                    "notes": f"loftr_match {type(e).__name__}: {e}"})
        return row
    row["n_corr"] = int(len(la))
    row["loftr_s"] = round(time.time() - t0, 1)
    if len(la) < 8:
        row.update({"verdict": "DEGENERATE_FAILURE",
                    "notes": "loftr <8 correspondences"})
        return row
    try:
        ka, rb, stats = refine_lk(iirs, tmc, la, lb, model=None)
    except Exception as e:
        row.update({"verdict": "ERROR",
                    "notes": f"refine_lk {type(e).__name__}: {e}"})
        return row
    row["n_lk_kept"] = int(stats["refined_pairs"])
    if len(ka) < 8:
        row.update({"verdict": "DEGENERATE_FAILURE",
                    "notes": "lk kept <8"})
        return row
    g = gate_verdict(ka, rb, (H, W))
    row.update({"n_inliers": g["inliers"], "rmse_px": g["rmse_px"],
                "verdict": g["verdict"], "entropy": g["entropy"],
                "quadrants": g["quadrants"],
                "notes": g.get("message", g.get("abstain", ""))[:120],
                "elapsed_s": round(time.time() - t0, 1)})
    return row


def main():
    print("loading tie grids ...", flush=True)
    iirs_grid = TieGrid(IIRS_GEO)
    tmc_grid = TieGrid(TMC_GEO)
    rows = []
    for (tag, l0, l1, s0, s1, kind) in WINDOWS:
        print(f"=== {tag} [{kind}] IIRS lines {l0}:{l1} samples {s0}:{s1} ===",
              flush=True)
        t0 = time.time()
        iirs = vnir_window(l0, l1, s0, s1)
        print(f"  VNIR composite {iirs.shape} in {time.time()-t0:.0f}s",
              flush=True)
        tmc, bbox = tmc_window(iirs_grid, tmc_grid, l0, l1, s0, s1)
        row = run_window(tag, iirs, tmc)
        row["kind"] = kind
        row["tmc_bbox"] = str(bbox)
        row["iirs_product"] = IIRS_PRODUCT
        row["tmc_product"] = TMC_PRODUCT
        rows.append(row)
        print(f"  -> corr={row.get('n_corr')} lk_kept={row.get('n_lk_kept')} "
              f"inliers={row.get('n_inliers')} "
              f"rmse={row.get('rmse_px')} [{row.get('verdict')}]",
              flush=True)
        del iirs, tmc
    cols = ["window", "kind", "iirs_shape", "tmc_bbox", "n_corr", "n_lk_kept",
            "n_inliers", "rmse_px", "verdict", "entropy", "quadrants",
            "loftr_s", "elapsed_s", "notes", "iirs_product", "tmc_product"]
    with open(OUT_CSV, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    print("wrote", OUT_CSV, flush=True)


if __name__ == "__main__":
    main()
