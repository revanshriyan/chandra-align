"""
Numerical Stability Guards - SVD Condition Number Checks
"""

import numpy as np
import warnings


def compute_condition_number(matrix: np.ndarray) -> float:
    """
    Compute the condition number kappa(H) = sigma_max / sigma_min using SVD.
    
    Args:
        matrix: 2D transformation matrix (2x3 for affine, 3x3 for homography)
        
    Returns:
        Condition number kappa (float)
    """
    if matrix is None:
        return float('inf')
    
    # Ensure we're working with a 2D matrix
    M = np.asarray(matrix, dtype=np.float64)
    
    # For affine matrices (2x3), use the 2x2 linear part
    if M.shape == (2, 3):
        M = M[:, :2]
    elif M.shape == (3, 3):
        M = M[:2, :2]
    
    # Ensure square matrix
    if M.shape[0] != M.shape[1]:
        return float('inf')
    
    # SVD
    try:
        U, s, Vt = np.linalg.svd(M)
        if s[-1] == 0:
            return float('inf')
        kappa = s[0] / s[-1]
        return float(kappa)
    except np.linalg.LinAlgError:
        return float('inf')


def check_condition_number(matrix: np.ndarray, threshold: float = 2000.0) -> dict:
    """
    Check if matrix condition number exceeds threshold.
    
    Args:
        matrix: Transformation matrix
        threshold: Maximum allowed condition number (default 2000)
        
    Returns:
        Dict with: condition_number, is_stable, warning
    """
    kappa = compute_condition_number(matrix)
    is_stable = kappa <= threshold
    
    result = {
        'condition_number': kappa,
        'is_stable': is_stable,
        'threshold': threshold,
        'warning': None
    }
    
    if not is_stable:
        result['warning'] = (
            f"Matrix condition number {kappa:.2f} exceeds threshold {threshold}. "
            f"Transformation is numerically unstable. Triggering fallback."
        )
        warnings.warn(result['warning'], UserWarning)
    
    return result


def validate_transformation_matrix(matrix: np.ndarray, min_determinant: float = 1e-6) -> dict:
    """
    Validate a transformation matrix for numerical stability.
    
    Args:
        matrix: 2x3 affine or 3x3 homography matrix
        min_determinant: Minimum acceptable absolute determinant
        
    Returns:
        Dict with validation results
    """
    M = np.asarray(matrix, dtype=np.float64)
    
    result = {
        'valid': True,
        'issues': []
    }
    
    # Check for NaN/Inf
    if not np.isfinite(M).all():
        result['valid'] = False
        result['issues'].append('Matrix contains NaN or Inf values')
        return result
    
    # Check determinant of linear part
    if M.shape == (2, 3):
        linear = M[:, :2]
    elif M.shape == (3, 3):
        linear = M[:2, :2]
    else:
        result['valid'] = False
        result['issues'].append(f'Invalid matrix shape: {M.shape}')
        return result
    
    det = np.linalg.det(linear)
    if abs(det) < min_determinant:
        result['valid'] = False
        result['issues'].append(
            f'Matrix determinant {det:.2e} below minimum {min_determinant}'
        )
    
    # Condition number check
    cond_result = check_condition_number(M)
    if not cond_result['is_stable']:
        result['issues'].append(
            f"Condition number {cond_result['condition_number']:.2f} exceeds threshold"
        )
    
    result['valid'] = len(result['issues']) == 0
    return result