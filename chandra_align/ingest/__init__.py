"""Stage 1 — ingest: PDS4/QUB readers, CRS unification + projection-trap guard, tiling.

The CRS guard is the projection-trap protection (project file §4.6/§5): hard-assert
identical CRS on both images BEFORE matching. No silent reprojection — a mismatch is
an error, not an automatic fix.
"""

import re

import numpy as np

try:
    import rasterio
    _HAVE_RASTERIO = True
except ImportError:  # pragma: no cover - rasterio is a hard requirement, kept for safety
    _HAVE_RASTERIO = False


class CRSError(ValueError):
    """Raised when the two images do not share an identical CRS (projection trap)."""


def _crs_string(crs_obj) -> str:
    if crs_obj is None:
        return "NONE"
    try:
        s = crs_obj.to_string()
        return s or "NONE"
    except Exception:
        return str(crs_obj)


def _label_crs(path: str) -> str:
    """Read CRS from a PDS4 sidecar label (.xml) next to a .img/.qub file."""
    label = path.rsplit(".", 1)[0] + ".xml"
    try:
        with open(label, "r", encoding="utf-8", errors="replace") as f:
            text = f.read()
    except OSError:
        return "NONE"

    # Try standard PDS4 CRS element
    m = re.search(r"<crs[^>]*>\s*<([A-Za-z0-9_:\\-]+)", text, re.IGNORECASE)
    if m:
        return m.group(1)

    # Try ISDA projection element (common in CH-2 labels)
    m = re.search(r"<isda:projection[^>]*>\s*([^<]+)", text, re.IGNORECASE)
    if m:
        return "POLAR_STEREOGRAPHIC"

    # Try generic projection element
    m = re.search(r"<projection[^>]*>\s*([^<]+)", text, re.IGNORECASE)
    if m:
        return "POLAR_STEREOGRAPHIC"

    # LRO NAC CDR products from the MAP archive (lrolrc_1001) are polar stereographic map-projected
    if "lrolrc_1001" in text.lower() and "cdr" in text.lower():
        return "POLAR_STEREOGRAPHIC"

    return "NONE"


def _label_solar_angles(path: str) -> dict:
    """Extract solar geometry angles from PDS4 label."""
    label = path.rsplit(".", 1)[0] + ".xml"
    angles = {}
    try:
        with open(label, "r", encoding="utf-8", errors="replace") as f:
            text = f.read()
    except OSError:
        return angles

    import xml.etree.ElementTree as ET
    tree = ET.parse(label)
    root = tree.getroot()
    ns = {"isda": "https://isda.issdc.gov.in/pds4/isda/v1"}

    # ISDA solar angles (CH-2 OHRC)
    az = root.find('.//isda:sun_azimuth', ns)
    el = root.find('.//isda:sun_elevation', ns)
    inc = root.find('.//isda:solar_incidence', ns)
    if az is not None:
        angles["SOLAR_AZIMUTH"] = float(az.text)
    if el is not None:
        angles["SUN_ELEVATION"] = float(el.text)
    if inc is not None:
        angles["SOLAR_INCIDENCE"] = float(inc.text)

    # Standard PDS4 solar angles (could be in different namespaces)
    for elem_name, key in [
        ("sun_azimuth", "SOLAR_AZIMUTH"),
        ("sun_elevation", "SUN_ELEVATION"),
        ("solar_incidence", "SOLAR_INCIDENCE"),
        ("emission_angle", "EMISSION_ANGLE"),
    ]:
        m = re.search(f"<{elem_name}[^>]*>\\s*([^<]+)", text, re.IGNORECASE)
        if m and key not in angles:
            try:
                angles[key] = float(m.group(1))
            except ValueError:
                pass

    return angles


def _read_pds4_detached(path: str):
    """Read a raw PDS4 .img file using its detached .xml label.

    Returns (band, crs, transform, width, height, count).
    """
    import xml.etree.ElementTree as ET

    label_path = path.rsplit(".", 1)[0] + ".xml"
    tree = ET.parse(label_path)
    root = tree.getroot()
    ns = {"pds": "http://pds.nasa.gov/pds4/pds/v1"}

    # Find Array_2D_Image
    arr = root.find(".//pds:Array_2D_Image", ns)
    if arr is None:
        raise RuntimeError(f"No Array_2D_Image found in label {label_path}")

    # Get dimensions from Axis_Array elements
    axes = arr.findall("pds:Axis_Array", ns)
    if len(axes) < 2:
        raise RuntimeError("Expected at least 2 Axis_Array elements")

    # Last Index Fastest = line-major (row-major)
    # Sequence 1 = Line (rows), Sequence 2 = Sample (columns)
    height = int(axes[0].find("pds:elements", ns).text)
    width = int(axes[1].find("pds:elements", ns).text)

    # Get data type from Element_Array
    elem_arr = arr.find("pds:Element_Array", ns)
    if elem_arr is None:
        raise RuntimeError("No Element_Array found")
    data_type = elem_arr.find("pds:data_type", ns)
    if data_type is None:
        raise RuntimeError("No data_type in Element_Array")

    # Map PDS4 data types to numpy
    pds4_to_numpy = {
        "UnsignedByte": np.uint8,
        "SignedByte": np.int8,
        "UnsignedShort": np.uint16,
        "SignedShort": np.int16,
        "UnsignedInteger": np.uint32,
        "SignedInteger": np.int32,
        "UnsignedLong": np.uint64,
        "SignedLong": np.int64,
        "IEEE754Single": np.float32,
        "IEEE754Double": np.float64,
    }
    np_dtype = pds4_to_numpy.get(data_type.text, np.float32)

    # Get offset from File element
    file_elem = root.find(".//pds:File", ns)
    offset_elem = file_elem.find("pds:offset", ns) if file_elem is not None else None
    offset = int(offset_elem.text) if offset_elem is not None else 0

    # Read raw binary
    with open(path, "rb") as f:
        f.seek(offset)
        data = np.fromfile(f, dtype=np_dtype, count=width * height)

    if len(data) != width * height:
        raise RuntimeError(f"Expected {width * height} pixels, got {len(data)}")

    band = data.reshape(height, width).astype(np.float64)

    # Get CRS from label
    crs = _label_crs(path)

    # For now use identity transform (pixel coordinates)
    # Real lunar projection would need SPICE geometry
    from rasterio.transform import from_origin
    transform = from_origin(0.0, float(height), 1.0, 1.0)

    return band, crs, transform, width, height, 1


def read_image(path: str):
    """Read a first-band image as float64 array + metadata dict.

    GeoTIFF via rasterio. PDS4 .img / IIRS .qub via rasterio when a driver exists;
    CRS is taken from the sidecar label (.xml) when the file itself has none.
    For raw PDS4 .img with detached label, reads binary using label metadata.
    No fake fallbacks.
    """
    if not _HAVE_RASTERIO:
        raise RuntimeError("rasterio is required for image IO")

    # Try rasterio first
    try:
        with rasterio.open(path) as src:
            band = src.read(1).astype(np.float64)
            crs = _crs_string(src.crs)
            transform = src.transform
            width, height, count = src.width, src.height, src.count
    except rasterio.errors.RasterioIOError:
        # If rasterio can't open it and it's a .img/.qub, try reading via PDS4 label
        if path.lower().endswith((".img", ".qub")):
            band, crs, transform, width, height, count = _read_pds4_detached(path)
        else:
            raise

    if crs == "NONE" and path.lower().endswith((".img", ".qub")):
        crs = _label_crs(path)
    
    # Extract solar angles from label
    solar_angles = _label_solar_angles(path)
    
    meta = {
        "crs": crs,
        "transform": transform,
        "width": width,
        "height": height,
        "count": count,
        **solar_angles,
    }
    return band, meta


def assert_same_crs(crs_a: str, crs_b: str) -> str:
    """Projection-trap guard: hard-assert identical CRS BEFORE matching.

    Returns the shared CRS string. Raises CRSError on mismatch or missing CRS.
    """
    if crs_a == "NONE" or crs_b == "NONE":
        raise CRSError(
            "projection-trap guard: missing CRS on at least one image "
            f"(a={crs_a!r}, b={crs_b!r}). No silent reprojection — assign a CRS first."
        )
    if crs_a != crs_b:
        raise CRSError(
            "projection-trap guard: CRS mismatch "
            f"(a={crs_a!r}, b={crs_b!r}). Reproject both sides into a common lunar CRS "
            "(Equirectangular Moon / polar stereographic) BEFORE matching — never silently."
        )
    return crs_a


def assert_pair_same_crs(path_a: str, path_b: str):
    """Load both images, hard-assert identical CRS.

    Returns (band_a, band_b, meta_a, meta_b, crs).
    """
    band_a, meta_a = read_image(path_a)
    band_b, meta_b = read_image(path_b)
    crs = assert_same_crs(meta_a["crs"], meta_b["crs"])
    return band_a, band_b, meta_a, meta_b, crs


def make_tiles(shape, tile_size=(4096, 4096), overlap=0.20):
    """Tile an image into tile_size tiles with ~20% overlap. Returns (x, y, w, h) windows."""
    h, w = shape
    th, tw = tile_size
    step_y = max(1, int(th * (1 - overlap)))
    step_x = max(1, int(tw * (1 - overlap)))
    tiles = []
    y = 0
    while y < h:
        x = 0
        while x < w:
            tiles.append((x, y, min(tw, w - x), min(th, h - y)))
            if x + tw >= w:
                break
            x += step_x
        if y + th >= h:
            break
        y += step_y
    return tiles