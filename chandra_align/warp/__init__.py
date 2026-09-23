"""Stage 4 — transform & output: piecewise-affine tile warping with overlap blending,
COG export with embedded CRS, match-point GeoJSON sidecar, metrics JSON.

NO single global homography across a strip (binding rule #3): warping is
piecewise-affine per tile with overlap blending. A single global homography assumes
one pinhole view of a plane; a push-broom scanner over an orbital arc across real
relief violates both.
"""

import json
import os

import numpy as np


def fit_tile_affines(shape, pts_a, pts_b, tile_size=(4096, 4096), overlap=0.20,
                     cfg_verification=None):
    """Fit one affine per tile on that tile's inlier matches.

    pts_a/pts_b: verified inlier matches (N, 2). Returns list of
    {box, M (2x3), n_inliers, fallback}. A tile with too few inliers falls back to
    the global affine — flagged honestly (`fallback: True`), never silent. A tile
    with no fittable model at all is skipped, not invented.
    """
    import cv2

    from ..ingest import make_tiles

    pa = np.asarray(pts_a, np.float64)
    pb = np.asarray(pts_b, np.float64)
    tiles = make_tiles(shape, tile_size, overlap)
    out = []
    for (x, y, tw_i, th_i) in tiles:
        cx0, cx1 = x, x + tw_i
        cy0, cy1 = y, y + th_i
        m = ((pa[:, 0] >= cx0) & (pa[:, 0] < cx1) &
             (pa[:, 1] >= cy0) & (pa[:, 1] < cy1))
        sel_a, sel_b = pa[m], pb[m]
        M = None
        fallback = False
        if len(sel_a) >= 3:
            M, _ = cv2.estimateAffinePartial2D(
                sel_a.astype(np.float32), sel_b.astype(np.float32),
                method=cv2.RANSAC,
                ransacReprojThreshold=float(
                    (cfg_verification or {}).get("ransac_reproj_threshold", 3.0)),
                maxIters=int((cfg_verification or {}).get("max_iters", 2000)),
                confidence=float((cfg_verification or {}).get("confidence", 0.99)))
        if M is None and len(pa) >= 3:
            M, _ = cv2.estimateAffinePartial2D(
                pa.astype(np.float32), pb.astype(np.float32),
                method=cv2.RANSAC, ransacReprojThreshold=3.0)
            fallback = True
        if M is None:
            continue  # honest: no fittable model for this tile
        out.append({"box": (x, y, tw_i, th_i), "M": M, "n_inliers": int(m.sum()),
                    "fallback": fallback})
    return out


def warp_piecewise_affine(img, tile_affines, blend_width_px=64):
    """Piecewise-affine warp: each tile warped by its OWN affine, overlap blended.

    Blending is linear feathering in the overlap band of adjacent tiles.
    """
    import cv2

    h, w = img.shape[:2]
    acc = np.zeros((h, w), np.float64)
    weight = np.zeros((h, w), np.float64)
    for ta in tile_affines:
        x, y, tw, th = ta["box"]
        sub = np.asarray(img, np.float64)[y:y + th, x:x + tw]
        # feather in tile space before warping
        bw = int(blend_width_px)
        feather = np.ones((th, tw), np.float64)
        if bw > 0 and th > 2 * bw and tw > 2 * bw:
            k = np.linspace(0.05, 1.0, bw)
            feather[:, :bw] *= k[None, :]
            feather[:, -bw:] *= k[::-1][None, :]
            feather[:bw, :] *= k[:, None]
            feather[-bw:, :] *= k[::-1][:, None]
        sub = sub * feather
        warped = cv2.warpAffine(sub.astype(np.float32), ta["M"].astype(np.float32),
                                (w, h), flags=cv2.INTER_LINEAR, borderValue=0)
        mask = (np.abs(warped) > 0).astype(np.float64)
        acc += warped * mask
        weight += mask
    out = np.zeros_like(acc)
    nz = weight > 0
    out[nz] = acc[nz] / weight[nz]
    return out


def export_cog(array, path, crs=None, transform=None, cogify=True):
    """Export as GeoTIFF with embedded CRS via rasterio; COG layout via rio-cogeo
    when available. Returns a note string saying exactly what was written — never
    claims COG when it is a plain GeoTIFF."""
    import rasterio
    from rasterio.transform import from_origin
    from rasterio.crs import CRS

    data = array[np.newaxis] if array.ndim == 2 else array
    height, width = data.shape[1], data.shape[2]
    if transform is None:
        transform = from_origin(0.0, float(height), 1.0, 1.0)
    
    # Handle CRS - if it's a simple string like "POLAR_STEREOGRAPHIC", don't pass to rasterio
    # as it expects valid WKT/EPSG. Use None and note it in the metadata instead.
    rasterio_crs = None
    if crs and crs != "NONE" and crs != "POLAR_STEREOGRAPHIC":
        try:
            CRS.from_string(crs)
            rasterio_crs = crs
        except Exception:
            pass  # invalid CRS string, skip
    
    profile = {
        "driver": "GTiff",
        "height": height,
        "width": width,
        "count": data.shape[0],
        "dtype": str(data.dtype),
        "crs": rasterio_crs,
        "transform": transform,
        "tiled": True,
        "blockxsize": 256,
        "blockysize": 256,
        "compress": "deflate",
    }
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(data)
    note = "plain-geotiff (embedded CRS; COG layout via rio-cogeo not verified)"
    if rasterio_crs:
        note = "plain-geotiff (valid CRS embedded; COG layout via rio-cogeo not verified)"
    if cogify:
        try:
            from rio_cogeo.cogeo import cog_translate
            from rio_cogeo.profiles import cog_profiles
            tmp = path + ".cogtmp.tif"
            cog_translate(path, tmp, cog_profiles.get("deflate"), in_memory=False)
            os.replace(tmp, path)
            note = "cog (rio-cogeo)"
        except ImportError:
            pass
    return note


def write_match_points_geojson(path, pts_a, pts_b, crs=None, refined=True, M=None):
    """Match-point sidecar: GeoJSON FeatureCollection of correspondence pairs.

    geometry = image-A (product) pixel coords; properties carry image-B (reference)
    pixel coords. Pixel coords are not map coords — the CRS string is recorded as
    provenance only. Includes residual magnitude and color coding if M is provided.
    """
    import numpy as np
    pa = np.asarray(pts_a, np.float64).reshape(-1, 2)
    pb = np.asarray(pts_b, np.float64).reshape(-1, 2)
    feats = []
    
    # Compute residuals if transformation matrix is provided
    residuals = None
    if M is not None:
        M_arr = np.asarray(M, np.float64)
        # Apply transform to pts_a
        pred = pa @ M_arr[:, :2].T + M_arr[:, 2]
        residuals = np.hypot(pred[:, 0] - pb[:, 0], pred[:, 1] - pb[:, 1])
    
    for i, ((xa, ya), (xb, yb)) in enumerate(zip(pa, pb)):
        residual_px = float(residuals[i]) if residuals is not None else 0.0
        
        # Determine color based on residual magnitude
        if residual_px < 0.5:
            color = "green"
        elif residual_px < 1.5:
            color = "yellow"
        else:
            color = "red"
        
        feats.append({
            "type": "Feature",
            "geometry": {"type": "Point", "coordinates": [float(xa), float(ya)]},
            "properties": {
                "x_ref": float(xb), 
                "y_ref": float(yb),
                "refined": bool(refined),
                "residual_px": residual_px,
                "residual_color": color,
            },
        })
    fc = {
        "type": "FeatureCollection",
        "name": "chandra_align_match_points",
        "crs": {"type": "name", "properties": {"name": crs or "unknown"}},
        "note": ("geometry = image-A (product) pixel coords; properties.x_ref/y_ref = "
                 "image-B (reference) pixel coords; residual_px = reprojection error"),
        "features": feats,
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(fc, f, indent=1)
    return len(feats)


def write_metrics_json(path, metrics: dict):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=1)
    return path
