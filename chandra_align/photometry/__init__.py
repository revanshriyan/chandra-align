"""
Stage 1 — photometric normalisation.

Lommel-Seeliger / lunar-Lambert normalisation from incidence/emission/phase angles
(SPICE-derived when kernels are available). If SPICE kernels are absent, runs in
"geometry-from-metadata" mode and emits approximation_flag=true on every output
(metadata angles are ellipsoid-referenced: correct on flat mare, approximate on
crater walls — declared, not hidden).

Also includes CLAHE preprocessing and unified normalization pipeline.
"""

import numpy as np

from .normalization import (
    apply_clahe,
    apply_lommel_seeliger,
    apply_lunar_lambert,
    normalize_photometry,
)


def shadow_saturation_mask(img, shadow_threshold_deg=80.0, saturation_threshold=0.98,
                           incidence=None):
    """Mask pure-shadow (incidence > threshold) and blown-out pixels.

    Returns a boolean mask, True where the pixel is usable.
    """
    img = np.asarray(img, dtype=np.float64)
    finite = np.isfinite(img)
    lo, hi = np.nanmin(img[finite]), np.nanmax(img[finite])
    rng = hi - lo if hi > lo else 1.0
    norm = (img - lo) / rng
    usable = finite & (norm < saturation_threshold)
    if incidence is not None:
        usable &= ~(np.asarray(incidence, dtype=np.float64) > shadow_threshold_deg)
    return usable


def lommel_seeliger_model(incidence_deg, emission_deg):
    """Lommel-Seeliger disc function: 2*cos(i) / (cos(i) + cos(e)). Dimensionless."""
    i = np.radians(np.asarray(incidence_deg, dtype=np.float64))
    e = np.radians(np.asarray(emission_deg, dtype=np.float64))
    ci, ce = np.cos(i), np.cos(e)
    denom = ci + ce
    out = np.zeros_like(ci)
    nz = denom > 1e-9
    out[nz] = 2.0 * ci[nz] / denom[nz]
    return out


def lunar_lambert_model(incidence_deg, emission_deg):
    """Lunar-Lambert disc function: (1-L)*cos(i) + 2*L*cos(i)/(cos(i)+cos(e)).

    L = 2 / (1 + cos(e)) per McEwen. Dimensionless.
    """
    i = np.radians(np.asarray(incidence_deg, dtype=np.float64))
    e = np.radians(np.asarray(emission_deg, dtype=np.float64))
    ci, ce = np.cos(i), np.cos(e)
    L = 2.0 / (1.0 + ce)
    denom = ci + ce
    ls = np.zeros_like(ci)
    nz = denom > 1e-9
    ls[nz] = 2.0 * ci[nz] / denom[nz]
    return (1.0 - L) * ci + L * ls


def normalise(img, incidence_deg, emission_deg, model="lommel_seeliger",
              ref_incidence_deg=60.0, ref_emission_deg=0.0):
    """Divide by the photometric model at observed geometry, multiply at reference geometry.

    Same principle as USGS ISIS. Angles are scalar or per-pixel arrays.
    Returns (normalised_image, approximation_flag). approximation_flag is True
    whenever angles come from metadata rather than SPICE — set by the caller.
    """
    img = np.asarray(img, dtype=np.float64)
    if model == "lommel_seeliger":
        disc = lommel_seeliger_model(incidence_deg, emission_deg)
        ref = float(lommel_seeliger_model(ref_incidence_deg, ref_emission_deg))
    elif model == "lunar_lambert":
        disc = lunar_lambert_model(incidence_deg, emission_deg)
        ref = float(lunar_lambert_model(ref_incidence_deg, ref_emission_deg))
    else:
        raise ValueError(f"unknown photometric model: {model!r}")
    disc = np.asarray(disc, dtype=np.float64)
    safe = disc > 1e-6
    out = np.zeros_like(img)
    out[safe] = img[safe] * (ref / disc[safe])
    return out, True  # approximation flag: caller sets geometry provenance


def normalise_from_metadata(img, meta_angles: dict, cfg_photometry: dict):
    """Convenience wrapper: normalise using angles stored in image metadata.

    Always runs in geometry-from-metadata mode -> approximation_flag=True on the output.
    cfg_photometry: the `photometry` block of a modality profile.
    """
    model = cfg_photometry.get("model", "lommel_seeliger")
    inc = meta_angles.get(cfg_photometry.get("incidence_angle_key", "SOLAR_INCIDENCE"), 60.0)
    eme = meta_angles.get(cfg_photometry.get("emission_angle_key", "EMISSION_ANGLE"), 0.0)
    return normalise(img, inc, eme, model=model)


def apply_approximation_flag(metrics: dict) -> dict:
    """Stamp approximation_flag=true onto a metrics dict (geometry-from-metadata mode)."""
    metrics["approximation_flag"] = True
    metrics["geometry_mode"] = "from_metadata"
    return metrics


__all__ = [
    "shadow_saturation_mask",
    "lommel_seeliger_model",
    "lunar_lambert_model",
    "normalise",
    "normalise_from_metadata",
    "apply_approximation_flag",
    "apply_clahe",
    "apply_lommel_seeliger",
    "apply_lunar_lambert",
    "normalize_photometry",
]