from __future__ import annotations

import numpy as np
import pandas as pd


def _noise_by_size(size_class: str, base: float) -> float:
    if size_class == "SMALL":
        return base * 1.35
    if size_class == "LARGE":
        return base * 0.80
    return base


def kalman_rts_for_voyage(
    group: pd.DataFrame,
    schema: dict,
    cfg: dict,
) -> pd.DataFrame:
    group = group.copy().reset_index(drop=False)
    lat_col = schema["lat_col"]
    lon_col = schema["lon_col"]
    target_col = schema["target_col"]

    n = len(group)
    if n == 0:
        return group

    kcfg = cfg["kalman"]
    q_base = float(kcfg.get("process_noise_base", 0.35))
    r_pos = float(kcfg.get("measurement_position_noise", 12.0))
    r_vel = float(kcfg.get("measurement_velocity_noise", 2.0))
    speed_noise_scale = float(kcfg.get("speed_noise_scale", 0.08))

    xs = np.zeros((n, 4), dtype=float)
    Ps = np.zeros((n, 4, 4), dtype=float)
    Fs = np.zeros((n, 4, 4), dtype=float)
    x_preds = np.zeros((n, 4), dtype=float)
    P_preds = np.zeros((n, 4, 4), dtype=float)

    first = group.iloc[0]
    xs[0] = [
        float(first[lon_col]),
        float(first[lat_col]),
        float(first.get("vx_obs", 0.0)),
        float(first.get("vy_obs", 0.0)),
    ]
    Ps[0] = np.diag([100.0, 100.0, 10.0, 10.0])

    H_pos = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], dtype=float)
    H_full = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]], dtype=float)

    for i in range(1, n):
        row = group.iloc[i]
        dt = max(float(row.get("dt", 60.0)), 1.0)
        size_class = str(row.get("size_class", "MEDIUM"))
        speed = max(float(row.get("SOG_ms", 0.0)), 0.0)

        F = np.array(
            [
                [1, 0, dt, 0],
                [0, 1, 0, dt],
                [0, 0, 1, 0],
                [0, 0, 0, 1],
            ],
            dtype=float,
        )
        Fs[i] = F

        sigma_acc = _noise_by_size(size_class, q_base) * (1.0 + speed_noise_scale * speed)
        q = sigma_acc ** 2

        Q = q * np.array(
            [
                [dt ** 4 / 4, 0, dt ** 3 / 2, 0],
                [0, dt ** 4 / 4, 0, dt ** 3 / 2],
                [dt ** 3 / 2, 0, dt ** 2, 0],
                [0, dt ** 3 / 2, 0, dt ** 2],
            ],
            dtype=float,
        )

        x_pred = F @ xs[i - 1]
        P_pred = F @ Ps[i - 1] @ F.T + Q

        observed_position = int(row.get(target_col, 0)) == 0
        lon = float(row[lon_col])
        lat = float(row[lat_col])
        has_valid_pos = np.isfinite(lon) and np.isfinite(lat)

        if observed_position and has_valid_pos:
            z = np.array([lon, lat, float(row.get("vx_obs", 0.0)), float(row.get("vy_obs", 0.0))])
            R = np.diag([r_pos ** 2, r_pos ** 2, r_vel ** 2, r_vel ** 2])
            H = H_full
        else:
            z = np.array([float(row.get("vx_obs", 0.0)), float(row.get("vy_obs", 0.0))])
            H = np.array([[0, 0, 1, 0], [0, 0, 0, 1]], dtype=float)
            R = np.diag([(r_vel * 2) ** 2, (r_vel * 2) ** 2])

        S = H @ P_pred @ H.T + R
        K = P_pred @ H.T @ np.linalg.pinv(S)
        x_new = x_pred + K @ (z - H @ x_pred)
        P_new = (np.eye(4) - K @ H) @ P_pred

        x_preds[i] = x_pred
        P_preds[i] = P_pred
        xs[i] = x_new
        Ps[i] = P_new

    x_smooth = xs.copy()
    P_smooth = Ps.copy()

    for i in range(n - 2, -1, -1):
        F = Fs[i + 1]
        P_pred = P_preds[i + 1]
        if not np.isfinite(P_pred).all() or np.linalg.cond(P_pred + np.eye(4) * 1e-9) > 1e12:
            continue

        C = Ps[i] @ F.T @ np.linalg.pinv(P_pred)
        x_smooth[i] = xs[i] + C @ (x_smooth[i + 1] - x_preds[i + 1])
        P_smooth[i] = Ps[i] + C @ (P_smooth[i + 1] - P_pred) @ C.T

    group["kf_lon"] = x_smooth[:, 0]
    group["kf_lat"] = x_smooth[:, 1]
    group["kf_vx"] = x_smooth[:, 2]
    group["kf_vy"] = x_smooth[:, 3]

    group = group.set_index("index").sort_index()
    return group


def apply_kalman_rts(df: pd.DataFrame, schema: dict, cfg: dict) -> pd.DataFrame:
    parts = []
    voyage_col = schema["voyage_col"]

    for _, group in df.groupby(voyage_col, sort=False):
        parts.append(kalman_rts_for_voyage(group, schema, cfg))

    return pd.concat(parts, axis=0).sort_index()
