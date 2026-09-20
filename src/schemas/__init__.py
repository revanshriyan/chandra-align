"""
Chandra-Align Type-Safe Schema Layer

Pydantic v2 models for all pipeline data structures, API contracts,
and export schemas. Ensures strict type validation across boundaries.
"""

from src.schemas.models import (
    # Core metrics
    RMSEMetrics,
    InlierStats,
    MetricsBundle,
    TrustFlag,
    CalibrationStatus,
    # Geometry
    MatchPoint,
    MatchPointsGeoJSON,
    MatchPointFeature,
    # Config
    PipelineConfig,
    MatcherConfig,
    VerificationConfig,
    RefinementConfig,
    WarpConfig,
    TrustConfig,
    PhotometryConfig,
    # Exports
    RegistrationDossier,
    BenchmarkRow,
    # Enums/Types
    UNMEASURED,
    NotTrustedReason,
)

__all__ = [
    "RMSEMetrics",
    "InlierStats",
    "MetricsBundle",
    "TrustFlag",
    "CalibrationStatus",
    "MatchPoint",
    "MatchPointsGeoJSON",
    "MatchPointFeature",
    "PipelineConfig",
    "MatcherConfig",
    "VerificationConfig",
    "RefinementConfig",
    "WarpConfig",
    "TrustConfig",
    "PhotometryConfig",
    "RegistrationDossier",
    "BenchmarkRow",
    "UNMEASURED",
    "NotTrustedReason",
]