#!/usr/bin/env python3
"""Build ChandraBench v0.2 landmark CSVs from the human-verified clicker
exports (2026-10-24). Maps the clicker schema onto the v0.1 schema exactly;
VERIFIED -> ACCEPTED; ncc stays blank (the clicker exports carry no NCC
values); coordinate_space states the 2048x2048 crop space explicitly so
v0.2 pairs are never mixed with v0.1 full-resolution coordinates.
"""
import csv

SRC = {
    "ohrc_01": "/home/hatch/workspace/your_files/gt_ohrc_01_human_clicked.csv",
    "ohrc_04": "/home/hatch/workspace/your_files/gt_ohrc_04_human_clicked.csv",
}
REF_PRODUCT = "ch2_ohr_nrp_20240425T1209509264_d_img_d18"
SRC_PRODUCT = "ch2_ohr_nrp_20240425T1406019344_d_img_d18"
OUT = {
    "ohrc_01": "chandrabench/landmarks_ohrc_01_v0.2.csv",
    "ohrc_04": "chandrabench/landmarks_ohrc_04_v0.2.csv",
}
CAVEAT = {
    "ohrc_01": (
        "Pair coverage LT=2 below the >=3/quadrant protocol floor "
        "(eval reports floor_pass=false); labeled-good for strip-guard "
        "calibration (landmark RMSE 1.8869 px)"
    ),
    "ohrc_04": (
        "Pair coverage LB=2 below the >=3/quadrant protocol floor "
        "(eval reports floor_pass=false); thin coverage, 18 points below "
        "the ~20 target; UNLABELED for strip-guard calibration (landmark "
        "RMSE 2.2591 px fails the <2.0 px bar)"
    ),
}
HEADER = [
    "id", "pair", "x_ref", "y_ref", "x_src", "y_src", "ncc",
    "review_status", "generated_by", "reviewed_by", "review_date",
    "reference_product", "source_product", "coordinate_space", "caveat",
]

for pair, src_path in SRC.items():
    rows = list(csv.DictReader(open(src_path, newline="", encoding="utf-8")))
    assert all(r["review_status"] == "VERIFIED" for r in rows), pair
    assert len({r["id"] for r in rows}) == len(rows), pair
    with open(OUT[pair], "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=HEADER)
        w.writeheader()
        for r in rows:
            for k in ("x_ref", "y_ref", "x_src", "y_src"):
                v = float(r[k])
                assert 0.0 <= v < 2048.0, (pair, r["id"], k, v)
            w.writerow({
                "id": r["id"],
                "pair": pair,
                "x_ref": r["x_ref"],
                "y_ref": r["y_ref"],
                "x_src": r["x_src"],
                "y_src": r["y_src"],
                "ncc": "",
                "review_status": "ACCEPTED",
                "generated_by": (
                    "SIFT+RANSAC candidate proposal (NCC>=0.96 gate, "
                    "spread-enforced; independent of pipeline fit), "
                    "human-verified in landmark clicker"
                ),
                "reviewed_by": "Revan, human verification (landmark clicker)",
                "review_date": "2026-10-24",
                "reference_product": REF_PRODUCT,
                "source_product": SRC_PRODUCT,
                "coordinate_space": (
                    "2048x2048 benchmark-crop pixel coordinates "
                    "(x=column, y=row); crops "
                    f"data/benchmark_crops/{pair}_reference.png and "
                    f"data/benchmark_crops/{pair}_source.png - NOT "
                    "full-resolution coordinates; do not mix with v0.1 pairs"
                ),
                "caveat": CAVEAT[pair],
            })
    print(f"{pair}: {len(rows)} rows -> {OUT[pair]}")
