from dataclasses import dataclass
from enum import Enum
from typing import Optional


class MatcherStrategy(str, Enum):
    LIGHTGLUE_ALIKED = "LIGHTGLUE_ALIKED"
    SIFT_RANSAC = "SIFT_RANSAC"
    RIFT2_PHASE_CONGRUENCY = "RIFT2_PHASE_CONGRUENCY"
    CHAINED_TRANSFORM = "CHAINED_TRANSFORM"
    MUTUAL_INFORMATION = "MUTUAL_INFORMATION"
    SAR_OPTICAL_CROSSMODAL = "SAR_OPTICAL_CROSSMODAL"


@dataclass
class SensorMeta:
    sensor_name: str
    sensor_type: str          # e.g., "PANCHROMATIC", "SWIR"
    gsd_meters: float
    sun_azimuth_deg: float
    sun_incidence_deg: float

    # Aliases for backward compatibility
    @property
    def solar_azimuth_deg(self) -> float:
        return self.sun_azimuth_deg

    @property
    def solar_elevation_deg(self) -> float:
        return self.sun_incidence_deg


@dataclass
class RoutingDecision:
    primary_strategy: MatcherStrategy
    fallback_strategy: MatcherStrategy
    delta_azimuth_deg: float
    delta_incidence_deg: float
    scale_ratio: float
    needs_pre_resampling: bool
    target_resample_gsd: Optional[float]
    routing_reason: str


def compute_azimuth_delta(az1: float, az2: float) -> float:
    """Computes the shortest angular distance between two azimuth angles in degrees."""
    diff = abs(az1 - az2) % 360.0
    return 360.0 - diff if diff > 180.0 else diff


def evaluate_route(source_meta: SensorMeta, ref_meta: SensorMeta) -> RoutingDecision:
    delta_az = compute_azimuth_delta(source_meta.sun_azimuth_deg, ref_meta.sun_azimuth_deg)
    delta_inc = abs(source_meta.sun_incidence_deg - ref_meta.sun_incidence_deg)

    # Protect against division by zero
    scale_ratio = source_meta.gsd_meters / ref_meta.gsd_meters if ref_meta.gsd_meters > 0 else 1.0

    needs_resample = scale_ratio > 4.0 or scale_ratio < 0.25
    target_gsd = ref_meta.gsd_meters if needs_resample else None

    # Rule 1: SAR / Radar Cross-Modal (DF-SAR, L-band, S-band, etc.)
    sar_keywords = ["SAR", "RADAR", "DF-SAR", "L-BAND", "S-BAND"]
    is_sar_source = any(kw in source_meta.sensor_name.upper() for kw in sar_keywords)
    is_sar_ref = any(kw in ref_meta.sensor_name.upper() for kw in sar_keywords)
    
    if is_sar_source or is_sar_ref:
        return RoutingDecision(
            primary_strategy=MatcherStrategy.SAR_OPTICAL_CROSSMODAL,
            fallback_strategy=MatcherStrategy.MUTUAL_INFORMATION,
            delta_azimuth_deg=delta_az,
            delta_incidence_deg=delta_inc,
            scale_ratio=scale_ratio,
            needs_pre_resampling=needs_resample,
            target_resample_gsd=target_gsd,
            routing_reason="SAR/Radar cross-modal payload detected."
        )

    # Rule 1: SWIR / Hyperspectral Cross-Modal; RIFT2 is opt-in only.
    if source_meta.sensor_type == "SWIR" or ref_meta.sensor_type == "SWIR":
        return RoutingDecision(
            primary_strategy=MatcherStrategy.LIGHTGLUE_ALIKED,
            fallback_strategy=MatcherStrategy.MUTUAL_INFORMATION,
            delta_azimuth_deg=delta_az,
            delta_incidence_deg=delta_inc,
            scale_ratio=scale_ratio,
            needs_pre_resampling=needs_resample,
            target_resample_gsd=target_gsd,
            routing_reason="Cross-modal SWIR/Panchromatic payload detected."
        )

    # Rule 2: Illumination Shift (Azimuth >= 60 deg). The validated default
    # remains LightGlue, with SIFT as the classical fallback.
    if delta_az >= 60.0:
        return RoutingDecision(
            primary_strategy=MatcherStrategy.LIGHTGLUE_ALIKED,
            fallback_strategy=MatcherStrategy.SIFT_RANSAC,
            delta_azimuth_deg=delta_az,
            delta_incidence_deg=delta_inc,
            scale_ratio=scale_ratio,
            needs_pre_resampling=needs_resample,
            target_resample_gsd=target_gsd,
            routing_reason=f"High solar azimuth shift ({delta_az:.1f}° >= 60.0°)."
        )

    # Rule 3 & 4: Standard / Scale-adapted Regime
    return RoutingDecision(
        primary_strategy=MatcherStrategy.LIGHTGLUE_ALIKED,
        fallback_strategy=MatcherStrategy.SIFT_RANSAC,
        delta_azimuth_deg=delta_az,
        delta_incidence_deg=delta_inc,
        scale_ratio=scale_ratio,
        needs_pre_resampling=needs_resample,
        target_resample_gsd=target_gsd,
        routing_reason="Standard optical matching regime."
    )
