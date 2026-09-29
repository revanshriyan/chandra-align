"""
Chandra-Align Pydantic v2 Models

Strict type-safe models for all pipeline data structures, API contracts,
and export schemas. All models use Pydantic v2 with strict validation.
"""

from __future__ import annotations
from datetime import datetime
from enum import Enum
from typing import Any, Literal, Optional, Union
from pydantic import BaseModel, Field, ConfigDict, field_validator
import numpy as np


# ============================================================================
# Sentinel value for UNMEASURED states - use a string literal
# ============================================================================

UNMEASURED = "UNMEASURED"

# Type alias for fields that can be a value or UNMEASURED
UnmeasuredFloat = Union[float, Literal["UNMEASURED"]]
UnmeasuredInt = Union[int, Literal["UNMEASURED"]]
UnmeasuredStr = Union[str, Literal["UNMEASURED"]]


# ============================================================================
# Enums
# ============================================================================

class TrustFlag(str, Enum):
    TRUSTED = "Trusted"
    NOT_TRUSTED = "Not Trusted"


class CalibrationStatus(str, Enum):
    CALIBRATED = "CALIBRATED"
    UNCALIBRATED = "UNCALIBRATED"


class NotTrustedReason(str, Enum):
    LOW_TRUST = "LOW_TRUST"
    LOW_INLIER_RATIO = "LOW_INLIER_RATIO"
    LOW_UNIFORMITY = "LOW_UNIFORMITY"
    INSUFFICIENT_COVERAGE = "INSUFFICIENT_COVERAGE"
    HIGH_RMSE = "HIGH_RMSE"
    ILL_CONDITIONED_MATRIX = "ILL_CONDITIONED_MATRIX"
    TRANSFORM_FALLBACK_USED = "TRANSFORM_FALLBACK_USED"
    INSUFFICIENT_FOOTPRINT_OVERLAP = "INSUFFICIENT_FOOTPRINT_OVERLAP"
    NO_MATCHES = "NO_MATCHES"


class MatcherTier(str, Enum):
    TIER1_RIFT2 = "rift2"
    TIER2_LIGHTGLUE_ALIKED = "lightglue_aliked"
    TIER2_LIGHTGLUE_DISK = "lightglue_disk"
    TIER2_SIFT = "sift"  # synthetic fallback


class TransformationMode(str, Enum):
    PIECEWISE_AFFINE = "piecewise_affine"
    GLOBAL_AFFINE = "global_affine"
    SIMILARITY = "similarity"
    IDENTITY = "identity"


class PipelineStatus(str, Enum):
    COMPLETED = "completed"
    LOGGED_NOT_RUN = "logged-not-run"
    FAILED = "failed"


# ============================================================================
# Core Metrics Models
# ============================================================================

class RMSEMetrics(BaseModel):
    """Sub-pixel RMSE metrics with held-out validation."""
    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    held_out: bool = Field(..., description="Whether RMSE computed on held-out check points")
    n_check_points: int = Field(ge=0, description="Number of held-out check points")
    rmse_x_px: UnmeasuredFloat = Field(default=UNMEASURED, description="RMSE in X direction (pixels)")
    rmse_y_px: UnmeasuredFloat = Field(default=UNMEASURED, description="RMSE in Y direction (pixels)")
    rmse_px: UnmeasuredFloat = Field(default=UNMEASURED, description="Total RMSE (pixels)")
    rmse_m: UnmeasuredFloat = Field(default=UNMEASURED, description="RMSE in ground meters (RMSE_px * GSD)")
    confidence_interval_95: Optional[tuple[float, float]] = Field(
        default=None, description="95% CI from bootstrap if available"
    )

    @field_validator("rmse_x_px", "rmse_y_px", "rmse_px", "rmse_m", mode="before")
    @classmethod
    def normalize_unmeasured(cls, v: Any) -> UnmeasuredFloat:
        if v is None or v == "UNMEASURED":
            return UNMEASURED
        return float(v)


class InlierStats(BaseModel):
    """Inlier and match statistics."""
    model_config = ConfigDict(extra="allow", validate_assignment=True)

    raw_matches: int = Field(default=0, ge=0, description="Total raw feature matches found")
    verified_inliers: int = Field(default=0, ge=0, description="MAGSAC++ verified inliers")
    inliers: int = Field(default=0, ge=0, description="Alias for verified_inliers")
    inlier_ratio: float = Field(default=0.0, ge=0.0, le=1.0, description="Verified inliers / raw matches")
    uniformity_score: float = Field(default=0.0, ge=0.0, le=1.0, description="Quadtree spatial uniformity (0-1)")
    quadtree_entropy: Optional[float] = Field(
        default=None, ge=0.0, le=1.0, description="Normalized quadtree entropy"
    )
    spatial_coverage_pct: float = Field(default=0.0, ge=0.0, le=100.0, description="Spatial coverage percentage")


class MetricsBundle(BaseModel):
    """Complete pipeline metrics bundle - primary API contract."""
    model_config = ConfigDict(extra="allow", validate_assignment=True)

    # Identity
    run_id: str = ""
    timestamp: datetime = Field(default_factory=datetime.utcnow)

    # Trust & calibration
    trust_flag: TrustFlag = TrustFlag.NOT_TRUSTED
    calibration_status: CalibrationStatus = CalibrationStatus.UNCALIBRATED
    status: PipelineStatus = PipelineStatus.COMPLETED

    # Core metrics
    inliers: InlierStats = Field(default_factory=InlierStats)
    rmse: RMSEMetrics = Field(default_factory=RMSEMetrics)

    # Quality flags
    condition_number_kappa: UnmeasuredFloat = Field(default=UNMEASURED, description="SVD condition number of transform matrix")
    footprint_iou: UnmeasuredFloat = Field(default=UNMEASURED, description="Bounding box IoU of input pair")
    quality_flags: list[NotTrustedReason] = Field(default_factory=list)

    # Pipeline execution
    matcher_tier_used: MatcherTier = MatcherTier.TIER1_RIFT2
    escalated_to_tier2: bool = False
    transformation_mode: TransformationMode = TransformationMode.PIECEWISE_AFFINE
    approximation_flag: bool = Field(default=True, description="Geometry-from-metadata mode")
    runtime_s: float = Field(default=0.0, ge=0.0)

    # Outputs
    registered_cog_path: Optional[str] = None
    match_points_geojson_path: Optional[str] = None

    # Legacy fields from existing pipeline output
    matcher: Optional[str] = None
    success_rate: Optional[str] = None
    correct_match_ratio: Optional[str] = None
    raw_matches: Optional[int] = None
    export: Optional[dict] = None
    residual_map: Optional[dict] = None
    refinement: Optional[dict] = None

    @field_validator("condition_number_kappa", "footprint_iou", mode="before")
    @classmethod
    def normalize_unmeasured(cls, v: Any) -> UnmeasuredFloat:
        if v is None or v == "UNMEASURED":
            return UNMEASURED
        return float(v)

    @field_validator("inliers", mode="before")
    @classmethod
    def normalize_inliers(cls, v: Any) -> dict:
        if isinstance(v, int):
            # Legacy format: inliers was an int count
            return {"raw_matches": 0, "verified_inliers": v, "inliers": v, "inlier_ratio": 0.0}
        if isinstance(v, dict):
            # Map old field names to new
            if "inliers" in v and "verified_inliers" not in v:
                v["verified_inliers"] = v["inliers"]
            if "inliers" not in v and "verified_inliers" in v:
                v["inliers"] = v["verified_inliers"]
        return v

    @field_validator("rmse", mode="before")
    @classmethod
    def normalize_rmse(cls, v: Any) -> dict:
        if isinstance(v, dict):
            # Ensure all fields have defaults
            v.setdefault("held_out", False)
            v.setdefault("n_check_points", 0)
        return v


# ============================================================================
# Geometry / GeoJSON Models
# ============================================================================

class MatchPoint(BaseModel):
    """Single match point with refined/unrefined coordinates."""
    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    # Reference image coordinates (target)
    x_ref: float
    y_ref: float

    # Moving image coordinates (source)
    x_mov: float
    y_mov: float

    # Refinement status
    refined: bool = False
    residual_px: float = 0.0


class MatchPointFeature(BaseModel):
    """GeoJSON Feature for a match point."""
    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    type: Literal["Feature"] = "Feature"
    id: int
    geometry: dict = Field(default_factory=lambda: {"type": "Point", "coordinates": [0.0, 0.0]})
    properties: dict = Field(default_factory=dict)

    @classmethod
    def from_match_point(cls, mp: MatchPoint, feature_id: int) -> "MatchPointFeature":
        # Determine color based on residual magnitude
        residual = mp.residual_px
        if residual < 0.5:
            color = "green"
        elif residual < 1.5:
            color = "yellow"
        else:
            color = "red"
        
        return cls(
            id=feature_id,
            geometry={"type": "Point", "coordinates": [mp.x_ref, mp.y_ref]},
            properties={
                "x_ref": mp.x_ref,
                "y_ref": mp.y_ref,
                "x_mov": mp.x_mov,
                "y_mov": mp.y_mov,
                "refined": mp.refined,
                "residual_px": mp.residual_px,
                "residual_color": color,
            }
        )


class MatchPointsGeoJSON(BaseModel):
    """GeoJSON FeatureCollection of match points."""
    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    type: Literal["FeatureCollection"] = "FeatureCollection"
    name: str = "chandra_align_match_points"
    crs: dict = Field(default_factory=lambda: {"type": "name", "properties": {"name": "EPSG:4326"}})
    features: list[MatchPointFeature] = Field(default_factory=list)


# ============================================================================
# Configuration Models
# ============================================================================

class PhotometryConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    shadow_threshold_deg: float = Field(default=5.0, ge=0.0, le=90.0)
    saturation_threshold: float = Field(default=0.99, ge=0.0, le=1.0)
    normalization_method: Literal["lommel_seeliger", "lunar_lambert", "none"] = "lommel_seeliger"


class MatcherConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    tier1_method: MatcherTier = MatcherTier.TIER1_RIFT2
    tier2_method: MatcherTier = MatcherTier.TIER2_LIGHTGLUE_ALIKED
    inlier_ratio_floor: float = Field(default=0.15, ge=0.0, le=1.0)
    uniformity_floor: float = Field(default=0.125, ge=0.0, le=1.0)
    grid: tuple[int, int] = (8, 8)
    max_keypoints: int = 4096


class VerificationConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    method: Literal["RANSAC", "MAGSAC++", "USAC_MAGSAC"] = "RANSAC"
    confidence: float = Field(default=0.999, ge=0.9, le=0.9999)
    max_iters: int = Field(default=10000, ge=1000)
    threshold_px: float = Field(default=3.0, ge=0.5, le=20.0)
    min_inliers: int = Field(default=8, ge=4)


class RefinementConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    ncc_window: int = Field(default=15, ge=5, le=31)
    subpixel_method: Literal["parabolic", "gaussian"] = "parabolic"
    max_shift_px: float = Field(default=5.0, ge=0.5)


class WarpConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    tile_size: tuple[int, int] = (4096, 4096)
    overlap: float = Field(default=0.2, ge=0.0, le=0.5)
    blend_width_px: int = Field(default=64, ge=16, le=512)


class TrustConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    inlier_ratio_floor: float = Field(default=0.15, ge=0.0, le=1.0)
    uniformity_floor: float = Field(default=0.125, ge=0.0, le=1.0)
    rmse_px_ceiling: float = Field(default=0.3, ge=0.05)
    min_inliers: int = Field(default=8, ge=4)
    min_held_out: int = Field(default=3, ge=2)


class PipelineConfig(BaseModel):
    """Complete pipeline configuration from YAML."""
    model_config = ConfigDict(extra="forbid")

    photometry: PhotometryConfig
    matcher: MatcherConfig
    verification: VerificationConfig
    refinement: RefinementConfig
    warp: WarpConfig
    trust: TrustConfig
    tile_size: tuple[int, int] = (4096, 4096)
    tile_overlap: float = 0.2

    @classmethod
    def from_yaml(cls, yaml_dict: dict) -> "PipelineConfig":
        """Create PipelineConfig from loaded YAML dict."""
        return cls(**yaml_dict)


# ============================================================================
# Export Models
# ============================================================================

class RegistrationDossier(BaseModel):
    """Complete registration dossier for export."""
    model_config = ConfigDict(extra="forbid")

    # Metadata
    run_id: str
    timestamp: datetime
    chandra_align_version: str = "0.1.0"

    # Input images
    product_image: dict  # Flexible input metadata
    reference_image: dict

    # Registration results
    gsd_ratio: UnmeasuredFloat = UNMEASURED
    matcher_used: MatcherTier
    transformation_mode: TransformationMode
    transformation_matrix: list[list[float]]  # 3x3 or 2x3 affine
    match_points: list[MatchPoint]
    match_count: int

    # Metrics (embedded for completeness)
    inlier_count: int
    inlier_ratio: float
    uniformity_score: float
    spatial_coverage_pct: float
    quadtree_entropy: float
    matrix_condition_number: UnmeasuredFloat = UNMEASURED
    matrix_stable: bool = False

    # RMSE
    rmse_pixels: UnmeasuredFloat = UNMEASURED
    rmse_meters: UnmeasuredFloat = UNMEASURED
    rmse_x_pixels: UnmeasuredFloat = UNMEASURED
    rmse_y_pixels: UnmeasuredFloat = UNMEASURED

    # Trust
    trust_flag: TrustFlag
    calibration_status: CalibrationStatus
    approximation_flag: bool
    escalated_to_tier2: bool

    # Details
    fallback_reason: Optional[str] = None
    mesh_info: dict = Field(default_factory=dict)
    validation: dict = Field(default_factory=dict)
    quality_flags: list[NotTrustedReason] = Field(default_factory=list)

    @field_validator("matrix_condition_number", "rmse_pixels", "rmse_meters", "rmse_x_pixels", "rmse_y_pixels", mode="before")
    @classmethod
    def normalize_unmeasured(cls, v: Any) -> UnmeasuredFloat:
        if v is None or v == "UNMEASURED":
            return UNMEASURED
        return float(v)


class BenchmarkRow(BaseModel):
    """Single row for benchmark CSV export."""
    model_config = ConfigDict(extra="forbid")

    pair_id: str
    product_id: str
    reference_id: str
    acquisition_date: str
    solar_azimuth_delta_deg: UnmeasuredFloat = UNMEASURED
    solar_elevation_delta_deg: UnmeasuredFloat = UNMEASURED
    gsd_ratio: UnmeasuredFloat = UNMEASURED
    inlier_count: int = 0
    raw_matches: int = 0
    inlier_ratio: float = 0.0
    spatial_coverage_pct: float = 0.0
    uniformity_score: float = 0.0
    rmse_px: UnmeasuredFloat = UNMEASURED
    rmse_m: UnmeasuredFloat = UNMEASURED
    trust_status: TrustFlag = TrustFlag.NOT_TRUSTED
    matcher_used: MatcherTier = MatcherTier.TIER1_RIFT2
    transformation_mode: TransformationMode = TransformationMode.PIECEWISE_AFFINE
    condition_number: UnmeasuredFloat = UNMEASURED
    runtime_seconds: float = 0.0
    quality_flags: str = ""


# ============================================================================
# Utility functions
# ============================================================================

def unmeasured_to_none(obj: dict) -> dict:
    """Convert UNMEASURED sentinel to None for JSON serialization."""
    def convert(v: Any) -> Any:
        if v is UNMEASURED:
            return None
        if isinstance(v, dict):
            return {k: convert(v2) for k, v2 in v.items()}
        if isinstance(v, list):
            return [convert(v2) for v2 in v]
        return v
    return convert(obj)


def none_to_unmeasured(obj: dict) -> dict:
    """Convert None back to UNMEASURED for model validation."""
    def convert(v: Any) -> Any:
        if v is None:
            return UNMEASURED
        if isinstance(v, dict):
            return {k: convert(v2) for k, v2 in v.items()}
        if isinstance(v, list):
            return [convert(v2) for v2 in v]
        return v
    return convert(obj)
