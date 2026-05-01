import numpy as np

from route_reconstruction.metrics import normalized_rmse, normalized_xte


def test_metrics_non_negative():
    y_true = np.array([[0.0, 0.0], [10.0, 0.0], [20.0, 0.0]])
    y_pred = np.array([[0.0, 0.0], [9.0, 1.0], [20.0, 0.0]])

    assert normalized_rmse(y_true, y_pred) >= 0.0
    assert normalized_xte(y_true, y_pred) >= 0.0
