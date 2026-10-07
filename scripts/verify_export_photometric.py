"""Independent photometric verification of the dense-remap export.

For ohrc_01/02/03 (flag on): compares the exported field-remapped raster
against the affine-only warp of the IDENTICAL run (same matching, same
matrices -- only the final warp differs, via sabotaging
build_field_remap_maps). Metric: patch NCC on an independent check grid
(never seen by the stage; the stage's own stride split is not reused).

This is independent of the TPS fit: the fit never saw intensities, and the
check grid is unrelated to the stage's internal held-out split.
"""

import csv
import os
import sys

sys.path.insert(0, "/tmp/stubs")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import cv2

import app
import chandra_align.deform_field as df_mod

PAIRS = ["ohrc_01", "ohrc_02", "ohrc_03"]
CROP = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                    "data", "benchmark_crops")


def run_pair(pair, sabotage):
    ref = cv2.imread(os.path.join(CROP, "%s_reference.png" % pair),
                     cv2.IMREAD_GRAYSCALE)
    sec = cv2.imread(os.path.join(CROP, "%s_source.png" % pair),
                     cv2.IMREAD_GRAYSCALE)
    assert ref is not None and sec is not None
    os.environ["CHANDRA_DEFORM_FIELD"] = "1"
    orig = df_mod.build_field_remap_maps
    if sabotage:
        def _raiser(*a, **k):
            raise RuntimeError("sabotaged for affine-only comparison")
        df_mod.build_field_remap_maps = _raiser
    try:
        out = app._align_core(ref, sec, reference_sensor_name="OHRC")
    finally:
        df_mod.build_field_remap_maps = orig
        os.environ.pop("CHANDRA_DEFORM_FIELD", None)
    return out


def patch_ncc(a, b):
    a = a.astype(np.float64)
    b = b.astype(np.float64)
    a -= a.mean()
    b -= b.mean()
    denom = np.sqrt((a ** 2).sum() * (b ** 2).sum())
    if denom < 1e-9:
        return None
    return float((a * b).sum() / denom)


def main():
    rows = []
    for pair in PAIRS:
        out_f = run_pair(pair, sabotage=False)
        out_a = run_pair(pair, sabotage=True)
        kind_f = out_f["judge_metrics"]["warp_export_kind"]
        kind_a = out_a["judge_metrics"]["warp_export_kind"]
        assert kind_f == "field_remap", (pair, kind_f)
        assert kind_a.startswith("affine (field remap failed"), (pair, kind_a)
        assert out_f["status_code"] == out_a["status_code"] == "SUCCESS_SUBPIXEL"
        wf = out_f["warped_sec"]
        wa = out_a["warped_sec"]
        ref = out_f["ref_original"]
        assert wf.shape == wa.shape == ref.shape
        h, w = ref.shape[:2]
        ncc_f, ncc_a, mad_f, mad_a = [], [], [], []
        wins = 0
        n = 0
        for gy in np.linspace(120, h - 120, 20):
            for gx in np.linspace(120, w - 120, 20):
                y, x = int(gy), int(gx)
                pr = ref[y - 7:y + 8, x - 7:x + 8]
                pf = wf[y - 7:y + 8, x - 7:x + 8]
                pa = wa[y - 7:y + 8, x - 7:x + 8]
                if pr.shape != (15, 15) or pr.std() < 4.0:
                    continue
                nf = patch_ncc(pr, pf)
                na = patch_ncc(pr, pa)
                if nf is None or na is None:
                    continue
                ncc_f.append(nf)
                ncc_a.append(na)
                mad_f.append(float(np.abs(pr.astype(float) - pf.astype(float)).mean()))
                mad_a.append(float(np.abs(pr.astype(float) - pa.astype(float)).mean()))
                wins += (nf > na)
                n += 1
        rows.append({
            "pair": pair,
            "gate_rmse_px": round(float(out_f["judge_metrics"]["rmse_gate_px"]), 4),
            "n_patches": n,
            "mean_ncc_field": round(float(np.mean(ncc_f)), 4),
            "mean_ncc_affine": round(float(np.mean(ncc_a)), 4),
            "frac_field_wins": round(wins / n, 3),
            "mean_mad_field": round(float(np.mean(mad_f)), 3),
            "mean_mad_affine": round(float(np.mean(mad_a)), 3),
        })
        print(pair, rows[-1], flush=True)
    out_csv = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                           "results", "table_export_photometric.csv")
    with open(out_csv, "w", newline="") as fh:
        fh.write(",".join(rows[0].keys()) + "\n")
        for r in rows:
            fh.write(",".join(str(v) for v in r.values()) + "\n")
    print("wrote", out_csv)


if __name__ == "__main__":
    main()
