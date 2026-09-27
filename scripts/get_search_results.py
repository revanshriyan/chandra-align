#!/usr/bin/env python3
# Provenance: adapted workflow from reyyishreyas/lunar-image-registration:
# https://github.com/reyyishreyas/lunar-image-registration/blob/main/dataset_download_pipeline/get_search_results.py
"""Find LRO NAC products overlapping a Chandrayaan-2 PDS4 label footprint.

The script reads label geometry, queries NASA's ODE REST API, ranks candidate
LROC CDR products by polygon overlap then incidence angle, and optionally streams
the top product images to disk.

Workflow provenance: reyyishreyas/lunar-image-registration, specifically
dataset_download_pipeline/get_search_results.py. This repository implementation
retains the cited search/ranking workflow and adds input, geometry, and transfer
validation for this project.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

try:
    import requests
    from requests.adapters import HTTPAdapter
    from urllib3.util.retry import Retry
except ImportError as exc:  # pragma: no cover - installation guidance
    raise SystemExit("Install requirements with: python -m pip install -r requirements.txt") from exc

try:
    from shapely import wkt as shapely_wkt
    from shapely.geometry import Polygon
    from shapely.ops import transform as transform_geometry
except ImportError as exc:  # pragma: no cover - installation guidance
    raise SystemExit("Install requirements with: python -m pip install -r requirements.txt") from exc

try:
    from tqdm import tqdm
except ImportError as exc:  # pragma: no cover - installation guidance
    raise SystemExit("Install requirements with: python -m pip install -r requirements.txt") from exc


SOURCE_REFERENCE = (
    "https://github.com/reyyishreyas/lunar-image-registration/blob/main/"
    "dataset_download_pipeline/get_search_results.py"
)
ODE_REST = "https://oderest.rsl.wustl.edu/live2/"
ISDA_NAMESPACE = "https://isda.issdc.gov.in/pds4/isda/v1"
KM_PER_DEGREE_LATITUDE = 30.3
CHUNK_SIZE = 1024 * 1024
USER_AGENT = "chandra-align-ode-search/1.0 (SIH 2026; research use)"


def build_session() -> requests.Session:
    """Create a reusable HTTP session with retries for transient archive errors."""
    session = requests.Session()
    retry = Retry(
        total=4,
        connect=4,
        read=4,
        status=4,
        backoff_factor=0.8,
        status_forcelist=(408, 425, 429, 500, 502, 503, 504),
        allowed_methods=frozenset({"GET", "HEAD"}),
        respect_retry_after_header=True,
    )
    adapter = HTTPAdapter(max_retries=retry)
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    session.headers.update({"User-Agent": USER_AGENT})
    return session


def _qualified(namespace: str, tag: str) -> str:
    return f"{{{namespace}}}{tag}"


def _find_geometry(root: ET.Element) -> ET.Element:
    ns = ISDA_NAMESPACE
    for name in ("Refined_Corner_Coordinates", "System_Level_Coordinates"):
        found = root.find(f".//{_qualified(ns, 'Geometry_Parameters')}/{_qualified(ns, name)}")
        if found is not None:
            return found
    # Some label revisions use an alternate namespace URI. Match by local name
    # while keeping the search scoped to the instrument geometry section.
    for element in root.iter():
        if element.tag.rsplit("}", 1)[-1] != "Geometry_Parameters":
            continue
        for child in element:
            if child.tag.rsplit("}", 1)[-1] in {
                "Refined_Corner_Coordinates", "System_Level_Coordinates"
            }:
                return child
    raise ValueError(
        "No Refined_Corner_Coordinates or System_Level_Coordinates found in the PDS4 label."
    )


def _read_coordinate(container: ET.Element, name: str) -> float:
    matches = [node for node in container.iter() if node.tag.rsplit("}", 1)[-1] == name]
    if not matches or matches[0].text is None:
        raise ValueError(f"Missing coordinate element: {name}")
    try:
        value = float(matches[0].text.strip())
    except (ValueError, AttributeError) as exc:
        raise ValueError(f"Invalid coordinate value for {name}: {matches[0].text!r}") from exc
    if not math.isfinite(value):
        raise ValueError(f"Non-finite coordinate value for {name}")
    return value


def _unwrap_ring(corners: dict[str, tuple[float, float]]) -> list[tuple[float, float]]:
    """Unwrap corner longitudes so a dateline-crossing footprint stays narrow."""
    ordered = [corners[key] for key in ("UL", "UR", "LR", "LL")]
    ring: list[tuple[float, float]] = []
    previous_lon: float | None = None
    for lat, lon in ordered:
        lon = ((lon + 180.0) % 360.0) - 180.0
        if previous_lon is not None:
            while lon - previous_lon > 180.0:
                lon -= 360.0
            while lon - previous_lon < -180.0:
                lon += 360.0
        ring.append((lon, lat))
        previous_lon = lon
    return ring


def get_corners(xml_path: str | os.PathLike[str]) -> tuple[dict[str, tuple[float, float]], float, float, float, float]:
    """Read and validate the four label footprint corners (latitude, longitude)."""
    try:
        root = ET.parse(xml_path).getroot()
    except (ET.ParseError, OSError) as exc:
        raise ValueError(f"Could not read PDS4 label {xml_path}: {exc}") from exc
    geometry = _find_geometry(root)
    fields = {
        "UL": ("upper_left_latitude", "upper_left_longitude"),
        "UR": ("upper_right_latitude", "upper_right_longitude"),
        "LR": ("lower_right_latitude", "lower_right_longitude"),
        "LL": ("lower_left_latitude", "lower_left_longitude"),
    }
    corners = {
        key: (_read_coordinate(geometry, names[0]), _read_coordinate(geometry, names[1]))
        for key, names in fields.items()
    }
    for lat, lon in corners.values():
        if not -90.0 <= lat <= 90.0:
            raise ValueError(f"Latitude outside [-90, 90]: {lat}")
        if not -360.0 <= lon <= 360.0:
            raise ValueError(f"Longitude outside a recognized lunar convention: {lon}")

    ring = _unwrap_ring(corners)
    min_lon = min(lon for lon, _ in ring)
    max_lon = max(lon for lon, _ in ring)
    latitudes = [lat for lat, _ in corners.values()]
    if max_lon - min_lon >= 180.0:
        raise ValueError("Footprint spans 180 degrees or more; refusing an ambiguous ODE search box")
    return corners, min(latitudes), max(latitudes), min_lon, max_lon


def corners_to_polygon(corners: dict[str, tuple[float, float]]) -> Polygon:
    ring = _unwrap_ring(corners)
    polygon = Polygon(ring)
    if not polygon.is_valid:
        polygon = polygon.buffer(0)
    if polygon.is_empty or polygon.area <= 0:
        raise ValueError("Label corner coordinates do not form a non-empty polygon")
    return polygon


def _ode_longitude(lon: float) -> float:
    """Convert an unwrapped longitude to the 0..360 convention used by ODE."""
    return lon % 360.0


def query_ode(
    session: requests.Session,
    minlat: float,
    maxlat: float,
    westlon: float,
    eastlon: float,
    page_size: int = 500,
) -> list[dict[str, Any]]:
    if not 1 <= page_size <= 500:
        raise ValueError("ODE page size must be between 1 and 500")
    params = {
        "query": "product", "target": "moon", "results": "fmpc",
        "ihid": "LRO", "iid": "LROC", "pt": "CDRNAC4", "output": "JSON",
        "minlat": minlat, "maxlat": maxlat,
        "westlon": _ode_longitude(westlon), "eastlon": _ode_longitude(eastlon),
        "limit": page_size,
    }
    print("  Querying NASA ODE REST API...")
    west = params["westlon"]
    east = params["eastlon"]
    boxes = [(west, east)] if west <= east else [(west, 360.0), (0.0, east)]
    products: list[dict[str, Any]] = []
    for box_west, box_east in boxes:
        box_params = {**params, "westlon": box_west, "eastlon": box_east}
        response = session.get(ODE_REST, params=box_params, timeout=(15, 60))
        try:
            response.raise_for_status()
            try:
                payload = response.json()
            except requests.exceptions.JSONDecodeError as exc:
                raise ValueError("ODE returned a non-JSON response") from exc
        finally:
            response.close()
        batch = payload.get("ODEResults", {}).get("Products", {}).get("Product", [])
        if isinstance(batch, dict):
            batch = [batch]
        if not isinstance(batch, list):
            raise ValueError("Unexpected ODE response: Product field is not a list or object")
        products.extend(product for product in batch if isinstance(product, dict))
    # Wrapped search boxes can return the same product twice.
    unique: dict[str, dict[str, Any]] = {}
    for product in products:
        key = str(product.get("pdsid") or product.get("Product_name") or id(product))
        unique[key] = product
    products = list(unique.values())
    print(f"  ODE returned {len(products)} candidate(s).")
    return [product for product in products if isinstance(product, dict)]


def _polygon_near_target(candidate: Polygon, target_center_lon: float) -> Polygon:
    """Unwrap ODE coordinates onto the longitude branch used by the label."""
    def normalize_x(x: Any, y: Any, z: Any = None):
        def adjust(value: float) -> float:
            return target_center_lon + ((float(value) - target_center_lon + 180.0) % 360.0) - 180.0
        try:
            xs = [adjust(value) for value in x]
            return (xs, y) if z is None else (xs, y, z)
        except TypeError:
            return (adjust(x), y) if z is None else (adjust(x), y, z)
    return transform_geometry(normalize_x, candidate)


def _product_image_url(product: dict[str, Any]) -> str | None:
    files = product.get("Product_files", {}).get("Product_file", [])
    if isinstance(files, dict):
        files = [files]
    if not isinstance(files, list):
        return None
    for record in files:
        if not isinstance(record, dict):
            continue
        filename = str(record.get("FileName", "")).upper()
        url = record.get("URL") or record.get("Url") or record.get("url")
        if filename.endswith(".IMG") and isinstance(url, str):
            parsed = urlparse(url)
            if parsed.scheme == "https" and parsed.netloc:
                return url
    return None


def rank_products(products: list[dict[str, Any]], target_poly: Polygon) -> list[dict[str, Any]]:
    target_area = target_poly.area
    if target_area <= 0:
        raise ValueError("Target footprint has no measurable area")
    target_center_lon = target_poly.centroid.x
    lat_mid = target_poly.centroid.y
    km_lon = KM_PER_DEGREE_LATITUDE * max(0.0, math.cos(math.radians(abs(lat_mid))))
    ranked: list[dict[str, Any]] = []

    for product in products:
        footprint = product.get("Footprint_geometry", "")
        overlap_km2 = overlap_pct = 0.0
        try:
            candidate_poly = shapely_wkt.loads(footprint)
            if candidate_poly.geom_type not in {"Polygon", "MultiPolygon"}:
                raise ValueError("Footprint is not polygonal")
            if not candidate_poly.is_valid:
                candidate_poly = candidate_poly.buffer(0)
            candidate_poly = _polygon_near_target(candidate_poly, target_center_lon)
            intersection = target_poly.intersection(candidate_poly)
            overlap_km2 = max(0.0, intersection.area * KM_PER_DEGREE_LATITUDE * km_lon)
            overlap_pct = max(0.0, min(100.0, 100.0 * intersection.area / target_area))
        except Exception:
            # Missing/malformed footprints remain visible as zero-overlap rows.
            pass

        try:
            incidence = float(product.get("Incidence_angle", 999.0))
            if not math.isfinite(incidence):
                incidence = 999.0
        except (TypeError, ValueError):
            incidence = 999.0

        ranked.append({
            "product_id": product.get("pdsid") or product.get("Product_name") or "unknown",
            "overlap_km2": round(overlap_km2, 2),
            "overlap_pct": round(overlap_pct, 1),
            "incidence": incidence,
            "img_url": _product_image_url(product),
            "obs_time": product.get("Observation_time", ""),
        })
    ranked.sort(key=lambda row: (-row["overlap_pct"], row["incidence"], str(row["product_id"])))
    return ranked


def _validate_image_response(prefix: bytes, content_type: str) -> None:
    media_type = content_type.lower().split(";", 1)[0]
    probe = prefix.lstrip().lower()
    if media_type in {"text/html", "application/json", "application/problem+json"}:
        raise ValueError(f"Expected an IMG product but server returned {media_type}")
    if probe.startswith((b"<!doctype html", b"<html", b"{\"error", b"access denied", b"not found")):
        raise ValueError("Server returned an error page, not the requested IMG data")
    if not prefix:
        raise ValueError("Server returned an empty IMG response")


def download(session: requests.Session, url: str, dest_path: str | os.PathLike[str]) -> Path:
    destination = Path(dest_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    response = session.get(url, stream=True, timeout=(20, 180))
    try:
        response.raise_for_status()
    except requests.RequestException:
        response.close()
        raise
    content_length = response.headers.get("Content-Length")
    total = int(content_length) if content_length and content_length.isdigit() else None
    fd, temp_name = tempfile.mkstemp(prefix=destination.name + ".", suffix=".part",
                                     dir=destination.parent)
    actual = 0
    try:
        iterator = response.iter_content(chunk_size=CHUNK_SIZE)
        first = next(iterator, b"")
        _validate_image_response(first[:512], response.headers.get("Content-Type", ""))
        with os.fdopen(fd, "wb") as output, tqdm(
            total=total, unit="B", unit_scale=True, unit_divisor=1024,
            desc=destination.name, dynamic_ncols=True,
        ) as progress:
            if first:
                output.write(first)
                actual += len(first)
                progress.update(len(first))
            for chunk in iterator:
                if chunk:
                    output.write(chunk)
                    actual += len(chunk)
                    progress.update(len(chunk))
            output.flush()
            os.fsync(output.fileno())
        if total is not None and actual != total:
            raise IOError(f"Transfer size mismatch: received {actual} bytes; expected {total}")
        if actual == 0:
            raise IOError("Downloaded image is empty")
        os.replace(temp_name, destination)
    except Exception:
        try:
            os.close(fd)
        except OSError:
            pass
        if os.path.exists(temp_name):
            os.unlink(temp_name)
        raise
    finally:
        response.close()
    print(f"  Saved {actual:,} bytes to {destination}")
    return destination


def _print_ranked(ranked: list[dict[str, Any]], count: int = 20) -> None:
    print(f"\n{'#':>3}  {'Product ID':25}  {'Overlap':>12}  {'% cover':>8}  {'Incidence':>10}  Observation")
    print("-" * 100)
    for index, result in enumerate(ranked[:count], 1):
        print(
            f"{index:3d}  {str(result['product_id'])[:25]:25}  "
            f"{result['overlap_km2']:>9.1f} km²  {result['overlap_pct']:>7.1f}%  "
            f"{result['incidence']:>9.2f}°  {str(result['obs_time'])[:19]}"
        )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("xml", help="Chandrayaan-2 PDS4 label XML (OHRC, TMC-2, or IIRS)")
    parser.add_argument("--out-dir", default="downloads", help="Directory for selected LRO NAC .IMG products")
    parser.add_argument("--top", type=int, default=1, help="Number of highest-ranked products to download")
    parser.add_argument("--no-download", action="store_true", help="Only search and rank; do not download")
    parser.add_argument("--page-size", type=int, default=500, help="ODE result limit (1..500)")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.top < 1:
        print("--top must be at least 1", file=sys.stderr)
        return 2
    xml_path = Path(args.xml)
    if not xml_path.is_file():
        print(f"Label file not found: {xml_path}", file=sys.stderr)
        return 2

    try:
        print(f"\nReading label: {xml_path}")
        corners, minlat, maxlat, westlon, eastlon = get_corners(xml_path)
        target_poly = corners_to_polygon(corners)
        print(f"  Latitude: [{minlat:.5f}, {maxlat:.5f}]  longitude: [{westlon:.5f}, {eastlon:.5f}] (unwrapped)")
        session = build_session()
        products = query_ode(session, minlat, maxlat, westlon, eastlon, args.page_size)
        if not products:
            print("ODE returned no candidates for this footprint.", file=sys.stderr)
            return 1
        ranked = rank_products(products, target_poly)
        _print_ranked(ranked)
        if args.no_download:
            return 0

        selected = [row for row in ranked if row["img_url"]][:args.top]
        if not selected:
            print("No ODE candidates contain a direct HTTPS NAC IMG file URL.", file=sys.stderr)
            return 1
        output_dir = Path(args.out_dir) / xml_path.stem
        print(f"\nDownloading {len(selected)} candidate(s) into {output_dir}")
        failed = 0
        for row in selected:
            filename = Path(urlparse(row["img_url"]).path).name
            # ODE filenames are archive-provided; reject path components before use.
            if not filename or filename in {".", ".."} or re.search(r"[\\/\x00]", filename):
                print(f"  Skipping unsafe archive filename for {row['product_id']}", file=sys.stderr)
                failed += 1
                continue
            destination = output_dir / filename
            print(f"\n{row['product_id']} — overlap {row['overlap_pct']:.1f}%, incidence {row['incidence']:.2f}°")
            try:
                download(session, row["img_url"], destination)
            except (requests.RequestException, OSError, ValueError) as exc:
                print(f"  Download failed: {exc}", file=sys.stderr)
                failed += 1
        return 1 if failed else 0
    except (requests.RequestException, ET.ParseError, ValueError, OSError) as exc:
        print(f"Dataset search failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
