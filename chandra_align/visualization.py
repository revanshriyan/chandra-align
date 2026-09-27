"""
CHANDRA-ALIGN: Lunar Cross-Sensor Registration Engine
Spatial Error Vector Field Visualization Module

Publication-ready 2D Spatial Residual Map generation:
- Quiver Plot / Vector Map
- Sub-pixel Error Distribution Plot
"""

import numpy as np
import cv2
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from typing import List, Dict, Optional, Tuple, Union
from dataclasses import dataclass
from io import BytesIO
import base64


@dataclass
class ResidualVector:
    """2D residual vector for a single inlier point."""
    ref_x: float
    ref_y: float
    sec_x: float
    sec_y: float
    dx_px: float
    dy_px: float
    magnitude_px: float
    magnitude_m: float
    inlier_weight: float = 1.0


def _display_bgr_uint8(image: np.ndarray) -> np.ndarray:
    """Convert grayscale/color rasters, including uint16, to displayable BGR8."""
    raster = np.asarray(image)
    if raster.size == 0 or raster.ndim not in (2, 3):
        raise ValueError("Reference raster must be a non-empty grayscale or color image")
    if raster.ndim == 3 and raster.shape[2] not in (1, 3, 4):
        raise ValueError("Reference raster must have 1, 3, or 4 channels")

    if raster.dtype == np.uint8:
        display = np.ascontiguousarray(raster)
    else:
        values = np.nan_to_num(
            raster.astype(np.float32, copy=False), nan=0.0,
            posinf=65535.0, neginf=0.0,
        )
        low, high = np.percentile(values, (1.0, 99.0))
        if high <= low:
            display = np.zeros(values.shape, dtype=np.uint8)
        else:
            display = np.clip((values - low) * (255.0 / (high - low)), 0, 255).astype(np.uint8)

    if display.ndim == 2:
        return cv2.cvtColor(display, cv2.COLOR_GRAY2BGR)
    if display.shape[2] == 1:
        return cv2.cvtColor(display[..., 0], cv2.COLOR_GRAY2BGR)
    if display.shape[2] == 4:
        return cv2.cvtColor(display, cv2.COLOR_BGRA2BGR)
    return display.copy()


def draw_error_vector_overlay(
    ref_img: np.ndarray,
    inliers_src: np.ndarray,
    inliers_dst_warped: np.ndarray,
    scale_factor: float = 10,
) -> np.ndarray:
    """Draw inlier points and 10x-scaled reprojection residual arrows.

    Coordinates are ``(x, y)`` pairs in the reference raster pixel frame.
    Green circles mark source/reference inliers; red arrows point toward each
    warped destination coordinate. The returned image is BGR uint8.
    """
    image = _display_bgr_uint8(ref_img)
    try:
        factor = float(scale_factor)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("scale_factor must be a finite positive number") from exc
    if not np.isfinite(factor) or factor <= 0:
        raise ValueError("scale_factor must be a finite positive number")

    source = np.asarray(inliers_src if inliers_src is not None else [], dtype=np.float64).reshape(-1, 2)
    warped = np.asarray(
        inliers_dst_warped if inliers_dst_warped is not None else [], dtype=np.float64
    ).reshape(-1, 2)
    if len(source) != len(warped):
        raise ValueError("Source and warped inlier coordinate counts must match")

    height, width = image.shape[:2]
    for src, dst in zip(source, warped):
        if not (np.isfinite(src).all() and np.isfinite(dst).all()):
            continue
        x, y = (int(round(float(value))) for value in src)
        if x < 0 or x >= width or y < 0 or y >= height:
            continue
        dx, dy = dst - src
        endpoint = (
            int(round(float(src[0] + dx * factor))),
            int(round(float(src[1] + dy * factor))),
        )
        cv2.circle(image, (x, y), 3, (0, 255, 0), thickness=-1, lineType=cv2.LINE_AA)
        if endpoint != (x, y):
            cv2.arrowedLine(
                image, (x, y), endpoint, (0, 0, 255),
                thickness=1, line_type=cv2.LINE_AA, tipLength=0.3,
            )
    return image


def create_quiver_plot(
    ref_image: np.ndarray,
    deformation_vectors: List[ResidualVector],
    grid_shape: Tuple[int, int] = (8, 8),
    max_vectors_per_cell: int = 10,
    scale: float = 1.0,
    colormap: str = 'RdYlGn_r',
    title: str = "2D Spatial Error Vector Field"
) -> plt.Figure:
    """
    Create a quiver plot overlaying displacement vectors onto the reference image.
    
    Args:
        ref_image: Reference image (H, W) or (H, W, 3)
        deformation_vectors: List of ResidualVector objects
        grid_shape: Grid resolution for spatial binning (rows, cols)
        max_vectors_per_cell: Max vectors to display per grid cell (for clarity)
        scale: Arrow scaling factor
        colormap: Matplotlib colormap for magnitude encoding
        title: Plot title
        
    Returns:
        matplotlib Figure object
    """
    if not deformation_vectors:
        fig, ax = plt.subplots(figsize=(10, 8))
        ax.imshow(ref_image, cmap='gray')
        ax.set_title(f"{title} (No Data)")
        ax.axis('off')
        return fig
    
    # Determine image bounds
    if ref_image.ndim == 3:
        h, w = ref_image.shape[:2]
    else:
        h, w = ref_image.shape
    
    rows, cols = grid_shape
    cell_h, cell_w = h / rows, w / cols
    
    # Bin vectors into grid cells
    grid_cells = {}
    for v in deformation_vectors:
        gx = int(np.clip(v.ref_x // cell_w, 0, cols - 1))
        gy = int(np.clip(v.ref_y // cell_h, 0, rows - 1))
        key = (gy, gx)
        if key not in grid_cells:
            grid_cells[key] = []
        grid_cells[key].append(v)
    
    # Subsample vectors per cell for visual clarity
    display_vectors = []
    for gy in range(rows):
        for gx in range(cols):
            key = (gy, gx)
            if key in grid_cells:
                vecs = grid_cells[key]
                # Sort by magnitude, take top N
                vecs_sorted = sorted(vecs, key=lambda v: v.magnitude_px, reverse=True)
                display_vectors.extend(vecs_sorted[:max_vectors_per_cell])
    
    if not display_vectors:
        fig, ax = plt.subplots(figsize=(10, 8))
        ax.imshow(ref_image, cmap='gray')
        ax.set_title(f"{title} (No Vectors After Subsampling)")
        ax.axis('off')
        return fig
    
    # Extract arrays for quiver
    X = np.array([v.ref_x for v in display_vectors])
    Y = np.array([v.ref_y for v in display_vectors])
    U = np.array([v.dx_px for v in display_vectors])
    V = np.array([v.dy_px for v in display_vectors])
    M = np.array([v.magnitude_px for v in display_vectors])
    
    # Normalize magnitudes for color
    M_norm = (M - M.min()) / (M.max() - M.min() + 1e-8)
    
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # Display reference image
    if ref_image.ndim == 3:
        ax.imshow(ref_image)
    else:
        ax.imshow(ref_image, cmap='gray')
    
    # Create quiver plot
    q = ax.quiver(
        X, Y, U * scale, V * scale,
        M, cmap=colormap, alpha=0.8,
        scale_units='xy', angles='xy', scale=1.0,
        width=0.003, headwidth=3, headlength=4
    )
    
    # Add colorbar
    cbar = plt.colorbar(q, ax=ax, fraction=0.03, pad=0.02)
    cbar.set_label('Residual Magnitude (px)', rotation=270, labelpad=15)
    
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xlim(0, w)
    ax.set_ylim(h, 0)  # Image coordinates
    ax.set_xlabel('X (pixels)')
    ax.set_ylabel('Y (pixels)')
    
    # Add grid overlay
    for i in range(1, rows):
        ax.axhline(i * cell_h, color='white', alpha=0.2, linewidth=0.5)
    for j in range(1, cols):
        ax.axvline(j * cell_w, color='white', alpha=0.2, linewidth=0.5)
    
    plt.tight_layout()
    return fig


def create_error_distribution_plot(
    deformation_vectors: List[ResidualVector],
    rings: List[float] = [0.5, 1.0],
    title: str = "Sub-pixel Error Distribution"
) -> plt.Figure:
    """
    Generate a color-coded scatter plot of residuals (Δx, Δy) 
    centered at (0,0) with error concentric rings.
    
    Args:
        deformation_vectors: List of ResidualVector objects
        rings: Concentric ring radii in pixels (default: [0.5, 1.0])
        title: Plot title
        
    Returns:
        matplotlib Figure object
    """
    if not deformation_vectors:
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.set_title(f"{title} (No Data)")
        ax.set_xlim(-2, 2)
        ax.set_ylim(-2, 2)
        return fig
    
    DX = np.array([v.dx_px for v in deformation_vectors])
    DY = np.array([v.dy_px for v in deformation_vectors])
    M = np.array([v.magnitude_px for v in deformation_vectors])
    
    fig, ax = plt.subplots(figsize=(8, 8))
    
    # Scatter plot colored by magnitude
    scatter = ax.scatter(
        DX, DY, c=M, cmap='hot', s=20, alpha=0.7, edgecolors='none'
    )
    
    # Add concentric rings
    theta = np.linspace(0, 2 * np.pi, 200)
    max_ring = max(rings + [M.max() * 1.2])
    for r in rings:
        ax.plot(r * np.cos(theta), r * np.sin(theta), 'w--', alpha=0.6, linewidth=1.5)
        # Add ring label
        ax.text(r * 0.7, r * 0.7, f'{r} px', color='white', fontsize=9, alpha=0.8)
    
    # Add axes
    ax.axhline(0, color='white', alpha=0.3, linewidth=0.5)
    ax.axvline(0, color='white', alpha=0.3, linewidth=0.5)
    
    # Set limits
    lim = max_ring * 1.3
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_aspect('equal')
    
    # Add colorbar
    cbar = plt.colorbar(scatter, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('Residual Magnitude (px)', rotation=270, labelpad=15)
    
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xlabel('ΔX (pixels)')
    ax.set_ylabel('ΔY (pixels)')
    ax.grid(True, alpha=0.2)
    
    # Add statistics text
    stats_text = (
        f"N = {len(deformation_vectors)}\n"
        f"Mean = {M.mean():.4f} px\n"
        f"RMSE = {np.sqrt(np.mean(M**2)):.4f} px\n"
        f"Max = {M.max():.4f} px"
    )
    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
            fontsize=10, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='black', alpha=0.7))
    
    plt.tight_layout()
    return fig


def create_combined_visualization(
    ref_image: np.ndarray,
    warped_secondary: np.ndarray,
    diff_map: np.ndarray,
    deformation_vectors: List[ResidualVector],
    grid_shape: Tuple[int, int] = (8, 8),
    pixel_scale_m: float = 0.25,
    rmse_px: float = 0.0,
    mae_px: float = 0.0,
    engine_name: str = "Unknown",
    inlier_count: int = 0,
    total_matches: int = 0
) -> plt.Figure:
    """
    Create a 4-panel scientific verification dossier:
    Panel 1: Side-by-side (Reference | Aligned Secondary | Delta)
    Panel 2: Quiver Plot / Vector Field
    Panel 3: Radiometric Delta Heatmap
    Panel 4: Sub-pixel Error Distribution
    """
    fig = plt.figure(figsize=(22, 17), facecolor='#090c10')
    gs = GridSpec(2, 2, figure=fig, hspace=0.20, wspace=0.12)
    
    # Overall title
    inlier_pct = (inlier_count / max(total_matches, 1)) * 100
    rmse_m = rmse_px * pixel_scale_m
    mae_m = mae_px * pixel_scale_m
    
    fig.suptitle(
        f"CHANDRA-ALIGN | Photogrammetric Verification Dossier\n"
        f"Engine: {engine_name} | Inliers: {inlier_count} / {total_matches} ({inlier_pct:.1f}%) | "
        f"RMSE: {rmse_px:.4f} px ({rmse_m:.4f} m) | MAE: {mae_px:.4f} px ({mae_m:.4f} m)",
        fontsize=14, fontweight='bold', color='#ffffff', y=0.96
    )
    
    # Panel 1: Side-by-side (Reference | Aligned Secondary | Delta)
    ax1 = fig.add_subplot(gs[0, 0])
    side_by_side = np.hstack((ref_image, warped_secondary, diff_map))
    if side_by_side.ndim == 2:
        ax1.imshow(side_by_side, cmap='gray')
    else:
        ax1.imshow(side_by_side)
    ax1.set_title("Panel 1: Reference | Aligned Secondary | Radiometric Delta", 
                  fontsize=11, color='#e6edf3', fontweight='bold')
    ax1.axis('off')
    
    # Panel 2: Quiver Plot / Vector Field
    ax2 = fig.add_subplot(gs[0, 1])
    if ref_image.ndim == 3:
        ax2.imshow(ref_image)
    else:
        ax2.imshow(ref_image, cmap='gray')
    
    if deformation_vectors:
        X = np.array([v.ref_x for v in deformation_vectors])
        Y = np.array([v.ref_y for v in deformation_vectors])
        U = np.array([v.dx_px for v in deformation_vectors])
        V = np.array([v.dy_px for v in deformation_vectors])
        M = np.array([v.magnitude_px for v in deformation_vectors])
        
        q = ax2.quiver(
            X, Y, U * 3, V * 3, M, cmap='RdYlGn_r', alpha=0.8,
            scale_units='xy', angles='xy', scale=1.0,
            width=0.003, headwidth=3, headlength=4
        )
        cbar = plt.colorbar(q, ax=ax2, fraction=0.03, pad=0.02)
        cbar.set_label('Residual (px)', rotation=270, labelpad=10, color='#8b949e')
        cbar.ax.tick_params(colors='#8b949e')
    
    ax2.set_title("Panel 2: 2D Spatial Error Vector Field (Quiver)", 
                  fontsize=11, color='#e6edf3', fontweight='bold')
    ax2.set_xlim(0, ref_image.shape[1])
    ax2.set_ylim(ref_image.shape[0], 0)
    ax2.tick_params(colors='#8b949e')
    
    # Grid overlay
    h, w = ref_image.shape[:2]
    rows, cols = grid_shape
    cell_h, cell_w = h / rows, w / cols
    for i in range(1, rows):
        ax2.axhline(i * cell_h, color='white', alpha=0.15, linewidth=0.5)
    for j in range(1, cols):
        ax2.axvline(j * cell_w, color='white', alpha=0.15, linewidth=0.5)
    
    # Panel 3: Radiometric Delta Heatmap
    ax3 = fig.add_subplot(gs[1, 0])
    im3 = ax3.imshow(diff_map, cmap='inferno')
    ax3.set_title("Panel 3: Radiometric Delta Heatmap (|I_ref - I_warped|)", 
                  fontsize=11, color='#e6edf3', fontweight='bold')
    ax3.axis('off')
    cbar3 = fig.colorbar(im3, ax=ax3, fraction=0.046, pad=0.03)
    cbar3.set_label('Radiometric Difference', color='#8b949e', fontsize=9)
    cbar3.ax.tick_params(colors='#8b949e')
    
    # Panel 4: Sub-pixel Error Distribution
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.set_facecolor('#0d1117')
    
    if deformation_vectors:
        DX = np.array([v.dx_px for v in deformation_vectors])
        DY = np.array([v.dy_px for v in deformation_vectors])
        M = np.array([v.magnitude_px for v in deformation_vectors])
        
        scatter = ax4.scatter(DX, DY, c=M, cmap='hot', s=20, alpha=0.7, edgecolors='none')
        
        # Concentric rings
        theta = np.linspace(0, 2 * np.pi, 200)
        for r in [0.5, 1.0]:
            ax4.plot(r * np.cos(theta), r * np.sin(theta), 'w--', alpha=0.6, linewidth=1.5)
            ax4.text(r * 0.7, r * 0.7, f'{r} px', color='white', fontsize=9, alpha=0.8)
        
        ax4.axhline(0, color='white', alpha=0.3, linewidth=0.5)
        ax4.axvline(0, color='white', alpha=0.3, linewidth=0.5)
        
        lim = max(1.5, M.max() * 1.3)
        ax4.set_xlim(-lim, lim)
        ax4.set_ylim(-lim, lim)
        ax4.set_aspect('equal')
        
        cbar4 = plt.colorbar(scatter, ax=ax4, fraction=0.046, pad=0.04)
        cbar4.set_label('Residual (px)', rotation=270, labelpad=10, color='#8b949e')
        cbar4.ax.tick_params(colors='#8b949e')
    
    ax4.set_title("Panel 4: Sub-pixel Error Distribution (ΔX, ΔY)", 
                  fontsize=11, color='#e6edf3', fontweight='bold')
    ax4.set_xlabel('ΔX (pixels)', fontsize=9, color='#8b949e')
    ax4.set_ylabel('ΔY (pixels)', fontsize=9, color='#8b949e')
    ax4.tick_params(colors='#8b949e')
    ax4.grid(True, linestyle=':', alpha=0.2, color='#8b949e')
    
    # Add statistics
    if deformation_vectors:
        M = np.array([v.magnitude_px for v in deformation_vectors])
        stats_text = (
            f"N = {len(deformation_vectors)}\n"
            f"Mean = {M.mean():.4f} px\n"
            f"RMSE = {np.sqrt(np.mean(M**2)):.4f} px\n"
            f"MAE = {M.mean():.4f} px\n"
            f"Std = {M.std():.4f} px\n"
            f"Max = {M.max():.4f} px"
        )
    else:
        stats_text = "No deformation data"
    
    ax4.text(0.02, 0.98, stats_text, transform=ax4.transAxes,
             fontsize=10, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='#161b22', edgecolor='#30363d', alpha=0.9))
    
    plt.tight_layout()
    return fig


def fig_to_base64(fig: plt.Figure, dpi: int = 150, format: str = 'png') -> str:
    """Convert matplotlib figure to base64 encoded string for Gradio."""
    buf = BytesIO()
    fig.savefig(buf, format=format, dpi=dpi, bbox_inches='tight', 
                facecolor=fig.get_facecolor(), edgecolor='none')
    buf.seek(0)
    img_base64 = base64.b64encode(buf.read()).decode('utf-8')
    buf.close()
    plt.close(fig)
    return img_base64


def fig_to_file(fig: plt.Figure, path: str, dpi: int = 300, format: str = 'png') -> None:
    """Save matplotlib figure to file."""
    fig.savefig(path, format=format, dpi=dpi, bbox_inches='tight',
                facecolor=fig.get_facecolor(), edgecolor='none')
    plt.close(fig)


__all__ = [
    "ResidualVector",
    "draw_error_vector_overlay",
    "create_quiver_plot",
    "create_error_distribution_plot",
    "create_combined_visualization",
    "fig_to_base64",
    "fig_to_file"
]
