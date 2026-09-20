/**
 * Chandra-Align TypeScript Type Definitions
 * 
 * Synchronized with src/schemas/models.py Pydantic models.
 * These types must match the Python backend API contracts exactly.
 */

// ============================================================================
// Sentinel types for UNMEASURED states
// ============================================================================

export type UnmeasuredFloat = number | 'UNMEASURED';
export type UnmeasuredInt = number | 'UNMEASURED';
export type UnmeasuredString = string | 'UNMEASURED';

// ============================================================================
// Enums
// ============================================================================

export type TrustFlag = 'Trusted' | 'Not Trusted';
export type CalibrationStatus = 'CALIBRATED' | 'UNCALIBRATED';
export type PipelineStatus = 'completed' | 'logged-not-run' | 'failed';

export type NotTrustedReason =
  | 'LOW_INLIER_RATIO'
  | 'LOW_UNIFORMITY'
  | 'INSUFFICIENT_COVERAGE'
  | 'HIGH_RMSE'
  | 'ILL_CONDITIONED_MATRIX'
  | 'TRANSFORM_FALLBACK_USED'
  | 'INSUFFICIENT_FOOTPRINT_OVERLAP'
  | 'NO_MATCHES'
  | 'LOW_TRUST';

export type MatcherTier =
  | 'rift2'
  | 'lightglue_aliked'
  | 'lightglue_disk'
  | 'sift';

export type TransformationMode =
  | 'piecewise_affine'
  | 'global_affine'
  | 'similarity'
  | 'identity';

// ============================================================================
// Core Metrics Types
// ============================================================================

export interface RMSEMetrics {
  held_out: boolean;
  n_check_points: number;
  rmse_x_px: UnmeasuredFloat;
  rmse_y_px: UnmeasuredFloat;
  rmse_px: UnmeasuredFloat;
  rmse_m: UnmeasuredFloat;
  confidence_interval_95?: [number, number];
}

export interface InlierStats {
  raw_matches: number;
  verified_inliers: number;
  inlier_ratio: number;
  uniformity_score: number;
  quadtree_entropy?: number;
  spatial_coverage_pct: number;
}

export interface MetricsBundle {
  // Identity
  run_id: string;
  timestamp: string; // ISO 8601

  // Trust & calibration
  trust_flag: TrustFlag;
  calibration_status: CalibrationStatus;
  status: PipelineStatus;

  // Core metrics
  inliers: InlierStats;
  rmse: RMSEMetrics;

  // Quality flags
  condition_number_kappa: UnmeasuredFloat;
  footprint_iou: UnmeasuredFloat;
  quality_flags: NotTrustedReason[];

  // Pipeline execution
  matcher_tier_used: MatcherTier;
  escalated_to_tier2: boolean;
  transformation_mode: TransformationMode;
  approximation_flag: boolean;
  runtime_s: number;

  // Outputs
  registered_cog_path?: string;
  match_points_geojson_path?: string;
}

// ============================================================================
// Geometry / GeoJSON Types
// ============================================================================

export interface MatchPoint {
  // Reference image coordinates (target)
  x_ref: number;
  y_ref: number;

  // Moving image coordinates (source)
  x_mov: number;
  y_mov: number;

  // Refinement status
  refined: boolean;
  residual_px: number;
}

export interface MatchPointFeature {
  type: 'Feature';
  id: number;
  geometry: {
    type: 'Point';
    coordinates: [number, number]; // [x, y] or [lon, lat] depending on CRS
  };
  properties: {
    x_ref: number;
    y_ref: number;
    x_mov: number;
    y_mov: number;
    refined: boolean;
    residual_px: number;
  };
}

export interface MatchPointsGeoJSON {
  type: 'FeatureCollection';
  name: string;
  crs: {
    type: 'name';
    properties: {
      name: string; // e.g., 'EPSG:4326'
    };
  };
  features: MatchPointFeature[];
}

// ============================================================================
// Configuration Types
// ============================================================================

export interface PhotometryConfig {
  shadow_threshold_deg: number;
  saturation_threshold: number;
  normalization_method: 'lommel_seeliger' | 'lunar_lambert' | 'none';
}

export interface MatcherConfig {
  tier1_method: MatcherTier;
  tier2_method: MatcherTier;
  inlier_ratio_floor: number;
  uniformity_floor: number;
  grid: [number, number];
  max_keypoints: number;
}

export interface VerificationConfig {
  method: 'RANSAC' | 'MAGSAC++' | 'USAC_MAGSAC';
  confidence: number;
  max_iters: number;
  threshold_px: number;
  min_inliers: number;
}

export interface RefinementConfig {
  ncc_window: number;
  subpixel_method: 'parabolic' | 'gaussian';
  max_shift_px: number;
}

export interface WarpConfig {
  tile_size: [number, number];
  overlap: number;
  blend_width_px: number;
}

export interface TrustConfig {
  inlier_ratio_floor: number;
  uniformity_floor: number;
  rmse_px_ceiling: number;
  min_inliers: number;
  min_held_out: number;
}

export interface PipelineConfig {
  photometry: PhotometryConfig;
  matcher: MatcherConfig;
  verification: VerificationConfig;
  refinement: RefinementConfig;
  warp: WarpConfig;
  trust: TrustConfig;
  tile_size: [number, number];
  tile_overlap: number;
}

// ============================================================================
// Export Types
// ============================================================================

export interface RegistrationDossier {
  // Metadata
  run_id: string;
  timestamp: string; // ISO 8601
  chandra_align_version: string;

  // Input images
  product_image: {
    id: string;
    path: string;
    crs: string;
    gsd_m: number;
    size: { width: number; height: number };
    solar_geometry: {
      sun_azimuth_deg: UnmeasuredFloat;
      sun_elevation_deg: UnmeasuredFloat;
      incidence_angle_deg: UnmeasuredFloat;
    };
  };
  reference_image: {
    id: string;
    path: string;
    crs: string;
    gsd_m: number;
    size: { width: number; height: number };
    solar_geometry: {
      sun_azimuth_deg: UnmeasuredFloat;
      sun_elevation_deg: UnmeasuredFloat;
      incidence_angle_deg: UnmeasuredFloat;
    };
  };

  // Registration results
  gsd_ratio: UnmeasuredFloat;
  matcher_used: MatcherTier;
  transformation_mode: TransformationMode;
  transformation_matrix: number[][]; // 3x3 or 2x3 affine
  match_points: MatchPoint[];
  match_count: number;

  // Metrics (embedded for completeness)
  inlier_count: number;
  inlier_ratio: number;
  uniformity_score: number;
  spatial_coverage_pct: number;
  quadtree_entropy: number;
  matrix_condition_number: UnmeasuredFloat;
  matrix_stable: boolean;

  // RMSE
  rmse_pixels: UnmeasuredFloat;
  rmse_meters: UnmeasuredFloat;
  rmse_x_pixels: UnmeasuredFloat;
  rmse_y_pixels: UnmeasuredFloat;

  // Trust
  trust_flag: TrustFlag;
  calibration_status: CalibrationStatus;
  approximation_flag: boolean;
  escalated_to_tier2: boolean;

  // Details
  fallback_reason?: string;
  mesh_info: Record<string, any>;
  validation: Record<string, any>;
  quality_flags: NotTrustedReason[];
}

export interface BenchmarkRow {
  pair_id: string;
  product_id: string;
  reference_id: string;
  acquisition_date: string;
  solar_azimuth_delta_deg: UnmeasuredFloat;
  solar_elevation_delta_deg: UnmeasuredFloat;
  gsd_ratio: UnmeasuredFloat;
  inlier_count: number;
  raw_matches: number;
  inlier_ratio: number;
  spatial_coverage_pct: number;
  uniformity_score: number;
  rmse_px: UnmeasuredFloat;
  rmse_m: UnmeasuredFloat;
  trust_status: TrustFlag;
  matcher_used: MatcherTier;
  transformation_mode: TransformationMode;
  condition_number: UnmeasuredFloat;
  runtime_seconds: number;
  quality_flags: string;
}

// ============================================================================
// API Response Types
// ============================================================================

export interface RegisterResponse {
  run_id: string;
  trust_flag: TrustFlag;
  metrics: MetricsBundle;
  cog_note: string;
  paths: {
    registered: string;
    match_points: string;
    metrics: string;
  };
}

export interface RunListItem {
  run_id: string;
  trust_flag: TrustFlag;
  matcher: string;
  runtime_s: number | 'UNMEASURED';
  has_metrics: boolean;
  has_cog: boolean;
  product_id?: string;
  status?: PipelineStatus;
}

// ============================================================================
// Utility Functions
// ============================================================================

export function isUnmeasured(value: UnmeasuredFloat | UnmeasuredInt | UnmeasuredString): boolean {
  return value === 'UNMEASURED';
}

export function formatUnmeasured<T>(value: UnmeasuredFloat | UnmeasuredInt | UnmeasuredString, fallback: string = 'UNMEASURED'): string {
  return isUnmeasured(value) ? fallback : String(value);
}

export function getTrustBadgeClass(trustFlag: TrustFlag, calibrationStatus: CalibrationStatus): string {
  if (trustFlag === 'Trusted') return 'trusted';
  if (trustFlag === 'Not Trusted') return 'not-trusted';
  if (calibrationStatus === 'UNCALIBRATED') return 'uncalibrated';
  return 'unknown';
}