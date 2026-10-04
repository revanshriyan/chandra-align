"""Materialize repeatable, approximate co-located PRADAN strip crop pairs.

The strip raster inputs are local PRADAN files and are intentionally not copied
into Git. Browse-image SIFT/RANSAC mappings were used to choose overlapping
windows; these are crop-selection aids, not independent ground truth.
"""

from __future__ import annotations

import argparse
import csv
import random
import sys
from pathlib import Path

import numpy as np
import rasterio
from rasterio.windows import Window

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

SIZE = 2048
MANIFEST = ROOT / "data" / "benchmark_pairs.csv"
CROPS = ROOT / "data" / "benchmark_crops"
FIELDS = [
    "pair_id", "sensor", "strip", "window_x", "terrain_note", "window_y",
    "crop_size", "reference_product", "source_product", "reference_crop",
    "source_crop", "source_window_x", "source_window_y", "mapping_method", "mapping_seed",
]

# Reference-to-source affine mappings measured on the local full-strip browse
# PNGs, whose width and height are each one tenth of the calibrated/raw strip.
# Coordinates here are in the full-resolution image pixel system. OHRC maps the
# 12:09 product to the 14:06 product; TMC maps fore to nadir.
MAPPINGS = {
    "OHRC": np.array([[1.00553448, -0.01578604, 1060.4775],
                       [0.01578604, 1.00553448, 1302.2119]], dtype=np.float64),
    "TMC-2": np.array([[1.00421806, 0.00238950, -315.0303],
                       [-0.00238950, 1.00421806, 7780.0726]], dtype=np.float64),
}

PRODUCTS = {
    "OHRC": (
        "ch2_ohr_nrp_20240425T1209509264_d_img_d18",
        "ch2_ohr_nrp_20240425T1406019344_d_img_d18",
    ),
    "TMC-2": (
        "ch2_tmc_ncf_20231101T0125121344_d_img_d18",
        "ch2_tmc_ncn_20231101T0125121377_d_img_d18",
    ),
}

TERRAIN_NOTES = {
    "OHRC": (
        "south-polar highlands; cratered surface with prominent shadowed crater rims (visual only; no DEM label)",
        "south-polar highlands; rough cratered relief and shadowed massif/ridge-like slopes (visual only)",
        "south-polar highlands; densely cratered, hummocky relief (visual only)",
        "south-polar highlands; inter-crater terrain with clustered small craters (visual only)",
        "south-polar highlands; mixed crater sizes and low-sun shadows (visual only)",
        "south-polar highlands; prominent shadowed crater and rugged rim relief (visual only)",
    ),
    "TMC-2": (
        "south-polar highlands; large shadowed crater walls and rugged relief (visual only; no DEM label)",
        "south-polar highlands; cratered surface with broad shadowed relief (visual only)",
        "south-polar highlands; large shadowed crater and cratered surroundings (visual only)",
        "south-polar highlands; crater rim and rugged inter-crater relief (visual only)",
        "south-polar highlands; densely cratered inter-crater terrain (visual only)",
        "south-polar highlands; isolated crater with shadowed rim relief (visual only)",
    ),
}


def mapped_crop_origin(matrix: np.ndarray, x: int, y: int) -> tuple[int, int]:
    center = np.array([x + SIZE / 2, y + SIZE / 2, 1.0])
    target = matrix @ center
    return round(target[0] - SIZE / 2), round(target[1] - SIZE / 2)


def read_crop(path: Path, x: int, y: int) -> np.ndarray:
    label = path.with_suffix(".xml")
    with rasterio.open(label if label.exists() else path) as dataset:
        if x < 0 or y < 0 or x + SIZE > dataset.width or y + SIZE > dataset.height:
            raise ValueError(f"Crop ({x},{y},{SIZE}) is outside {path.name} ({dataset.width}x{dataset.height})")
        raw = dataset.read(1, window=Window(x, y, SIZE, SIZE), masked=True)
        arr = np.asarray(raw.astype(np.float32).filled(np.nan))
    valid_fraction = float(np.count_nonzero(np.isfinite(arr) & (arr != 0)) / arr.size)
    if valid_fraction < 0.70:
        raise ValueError(f"Crop {path.name} at ({x},{y}) is mostly fill/black ({valid_fraction:.1%} valid)")
    import app

    normalized = app.load_lunar_raster(arr, max_dimension=SIZE)
    if normalized is None or normalized.shape != (SIZE, SIZE):
        raise RuntimeError(f"Could not normalize exact crop from {path}")
    return normalized


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-root", type=Path, default=ROOT / "data" / "qa" / "fullres",
                        help="Directory containing the local PRADAN .img and .xml files")
    parser.add_argument("--browse-root", type=Path,
                        default=ROOT / "data" / "qa" / "lite" / "chandra-align-images" / "pradan",
                        help="Directory containing the full-strip browse PNGs (for provenance only)")
    parser.add_argument("--seed", type=int, default=20261005)
    args = parser.parse_args()
    rng = random.Random(args.seed)
    CROPS.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, str | int]] = []

    # Six samples per pair. Rows span the available acquisition strips, while
    # different cross-track positions avoid repeatedly sampling one column.
    layouts = {
        "OHRC": [(y, x) for y, x in zip((1000, 17000, 33000, 49000, 65000, 81000),
                                         (600, 2100, 3700, 5300, 6900, 8500))],
        "TMC-2": [(y, x) for y, x in zip((53000, 76000, 99000, 122000, 145000, 168000),
                                          (300, 750, 1200, 1800, 900, 1200))],
    }
    for sensor, layout in layouts.items():
        ref_product, src_product = PRODUCTS[sensor]
        matrix = MAPPINGS[sensor]
        strip = "OHRC 12:09:50Z → 14:06:01Z" if sensor == "OHRC" else "TMC-2 fore → nadir"
        for index, (base_y, base_x) in enumerate(layout, start=1):
            # Seeded, small offsets make the generated windows reproducible and
            # avoid aligning every crop to the same artificial grid boundary.
            x = base_x + rng.randint(-24, 24)
            y = base_y + rng.randint(-24, 24)
            sx, sy = mapped_crop_origin(matrix, x, y)
            ref_path = args.input_root / f"{ref_product}.img"
            src_path = args.input_root / f"{src_product}.img"
            ref = read_crop(ref_path, x, y)
            src = read_crop(src_path, sx, sy)
            terrain = TERRAIN_NOTES[sensor][index - 1]
            pair_id = f"{sensor.lower().replace('-', '')}_{index:02d}"
            ref_file = CROPS / f"{pair_id}_reference.png"
            src_file = CROPS / f"{pair_id}_source.png"
            import cv2
            if not cv2.imwrite(str(ref_file), ref) or not cv2.imwrite(str(src_file), src):
                raise OSError(f"Failed to write crop images for {pair_id}")
            rows.append({
                "pair_id": pair_id, "sensor": sensor, "strip": strip,
                "window_x": x, "terrain_note": terrain, "window_y": y,
                "crop_size": SIZE, "reference_product": ref_product,
                "source_product": src_product,
                "reference_crop": ref_file.relative_to(ROOT).as_posix(),
                "source_crop": src_file.relative_to(ROOT).as_posix(),
                "source_window_x": sx, "source_window_y": sy,
                "mapping_method": "browse-image SIFT ratio matches + affine RANSAC; 10x browse-to-strip scale",
                "mapping_seed": args.seed,
            })

    MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    with MANIFEST.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} pairs, {2 * len(rows)} normalized PNG crops to {MANIFEST}")
    print("Window mapping is approximate browse-image registration; independent GT remains pending.")


if __name__ == "__main__":
    main()
