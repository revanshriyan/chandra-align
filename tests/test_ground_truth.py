import numpy as np
import pytest

from scripts.eval_ground_truth import evaluate_ground_truth


def test_ground_truth_rmse_and_circularity_guard():
    matrix = np.array([[1.0, 0.0, 2.0], [0.0, 1.0, -1.0]])
    fit_ref = np.array([[10.0, 10.0]])
    fit_src = np.array([[8.0, 11.0]])
    gt_ref = np.array([[20.0, 20.0], [30.0, 30.0]])
    gt_src = np.array([[18.0, 21.0], [28.0, 31.0]])
    assert evaluate_ground_truth(matrix, fit_ref, fit_src, gt_ref, gt_src) == pytest.approx(0.0)

    with pytest.raises(ValueError, match="CIRCULARITY GUARD"):
        evaluate_ground_truth(matrix, fit_ref, fit_src, np.vstack([gt_ref, fit_ref]), np.vstack([gt_src, fit_src]))
