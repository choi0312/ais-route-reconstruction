from __future__ import annotations

import numpy as np


def normalized_rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    mse = np.mean(np.sum((y_true - y_pred) ** 2, axis=1))
    rmse = np.sqrt(mse)

    x_range = np.nanmax(y_true[:, 0]) - np.nanmin(y_true[:, 0])
    y_range = np.nanmax(y_true[:, 1]) - np.nanmin(y_true[:, 1])
    denom = np.sqrt(x_range ** 2 + y_range ** 2)

    return float(rmse / max(denom, 1e-8))


def cross_track_error(points: np.ndarray, start: np.ndarray, end: np.ndarray) -> np.ndarray:
    points = np.asarray(points, dtype=float)
    start = np.asarray(start, dtype=float)
    end = np.asarray(end, dtype=float)

    line = end - start
    denom = np.linalg.norm(line)

    if denom < 1e-8:
        return np.linalg.norm(points - start, axis=1)

    return np.abs(np.cross(line, points - start) / denom)


def normalized_xte(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    if len(y_true) < 2:
        return 0.0

    start = y_true[0]
    end = y_true[-1]

    xte = cross_track_error(y_pred, start, end).mean()

    x_range = np.nanmax(y_true[:, 0]) - np.nanmin(y_true[:, 0])
    y_range = np.nanmax(y_true[:, 1]) - np.nanmin(y_true[:, 1])
    denom = np.sqrt(x_range ** 2 + y_range ** 2)

    return float(xte / max(denom, 1e-8))
