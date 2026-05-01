from __future__ import annotations

import numpy as np
import pandas as pd


def find_target_blocks(mask: np.ndarray) -> list[tuple[int, int]]:
    blocks = []
    start = None

    for i, value in enumerate(mask):
        if value and start is None:
            start = i
        elif not value and start is not None:
            blocks.append((start, i - 1))
            start = None

    if start is not None:
        blocks.append((start, len(mask) - 1))

    return blocks


def _safe_anchor(group: pd.DataFrame, idx: int, lat_col: str, lon_col: str, fallback_lat: float, fallback_lon: float):
    if 0 <= idx < len(group):
        row = group.iloc[idx]
        lat = float(row[lat_col])
        lon = float(row[lon_col])
        if np.isfinite(lat) and np.isfinite(lon):
            return lat, lon
    return fallback_lat, fallback_lon


def _block_alpha(block: pd.DataFrame, cfg: dict) -> float:
    rcfg = cfg["reconstruction"]
    base = float(rcfg.get("alpha_base", 0.55))
    gap_gain = float(rcfg.get("alpha_gap_gain", 0.20))
    curve_gain = float(rcfg.get("alpha_curve_gain", 0.20))

    L = max(len(block), 1)
    gap_term = min(1.0, L / 30.0)
    curve = float(pd.to_numeric(block.get("curvature", 0), errors="coerce").fillna(0).mean())
    curve_term = min(1.0, curve * 1000.0)

    alpha = base + gap_gain * gap_term + curve_gain * curve_term
    return float(np.clip(alpha, 0.25, 0.90))


def _gaussian_weights(length: int) -> np.ndarray:
    if length <= 1:
        return np.ones(length)

    x = np.linspace(-1.0, 1.0, length)
    w = np.exp(-0.5 * (x / 0.55) ** 2)
    w = (w - w.min()) / max(w.max() - w.min(), 1e-8)
    return 0.35 + 0.65 * w


def reconstruct_group(group: pd.DataFrame, schema: dict, cfg: dict) -> pd.DataFrame:
    group = group.copy().reset_index(drop=True)
    lat_col = schema["lat_col"]
    lon_col = schema["lon_col"]
    target_col = schema["target_col"]

    if "kf_lat" not in group.columns:
        group["kf_lat"] = group[lat_col]
    if "kf_lon" not in group.columns:
        group["kf_lon"] = group[lon_col]

    group["pred_lat"] = group[lat_col].astype(float)
    group["pred_lon"] = group[lon_col].astype(float)

    target_mask = group[target_col].astype(int).values == 1
    blocks = find_target_blocks(target_mask)

    for start, end in blocks:
        block = group.iloc[start : end + 1].copy()
        L = len(block)

        pre_idx = start - 1
        post_idx = end + 1

        fallback_pre_lat = float(group.loc[start, "kf_lat"])
        fallback_pre_lon = float(group.loc[start, "kf_lon"])
        fallback_post_lat = float(group.loc[end, "kf_lat"])
        fallback_post_lon = float(group.loc[end, "kf_lon"])

        pre_lat, pre_lon = _safe_anchor(group, pre_idx, lat_col, lon_col, fallback_pre_lat, fallback_pre_lon)
        post_lat, post_lon = _safe_anchor(group, post_idx, lat_col, lon_col, fallback_post_lat, fallback_post_lon)

        linear_lat = np.linspace(pre_lat, post_lat, L + 2)[1:-1]
        linear_lon = np.linspace(pre_lon, post_lon, L + 2)[1:-1]

        kf_lat = block["kf_lat"].to_numpy(dtype=float)
        kf_lon = block["kf_lon"].to_numpy(dtype=float)

        if L <= int(cfg["reconstruction"].get("short_gap_threshold", 2)):
            path_lat = 0.65 * linear_lat + 0.35 * kf_lat
            path_lon = 0.65 * linear_lon + 0.35 * kf_lon
        else:
            dlat = np.diff(np.r_[pre_lat, kf_lat])
            dlon = np.diff(np.r_[pre_lon, kf_lon])

            total_lat = np.sum(dlat)
            total_lon = np.sum(dlon)
            anchor_lat = post_lat - pre_lat
            anchor_lon = post_lon - pre_lon

            scale_lat = anchor_lat / total_lat if abs(total_lat) > 1e-8 else 1.0
            scale_lon = anchor_lon / total_lon if abs(total_lon) > 1e-8 else 1.0

            clip_factor = float(cfg["reconstruction"].get("delta_clip_factor", 1.8))
            scale_lat = float(np.clip(scale_lat, -clip_factor, clip_factor))
            scale_lon = float(np.clip(scale_lon, -clip_factor, clip_factor))

            scaled_lat = pre_lat + np.cumsum(dlat * scale_lat)
            scaled_lon = pre_lon + np.cumsum(dlon * scale_lon)

            alpha = _block_alpha(block, cfg)
            center_w = _gaussian_weights(L) if cfg["reconstruction"].get("use_gaussian_center_weight", True) else np.ones(L)

            dyn_alpha = alpha * center_w
            path_lat = dyn_alpha * scaled_lat + (1.0 - dyn_alpha) * kf_lat
            path_lon = dyn_alpha * scaled_lon + (1.0 - dyn_alpha) * kf_lon

            anchor_w = np.linspace(0.0, 1.0, L + 2)[1:-1]
            anchor_lat_path = (1.0 - anchor_w) * pre_lat + anchor_w * post_lat
            anchor_lon_path = (1.0 - anchor_w) * pre_lon + anchor_w * post_lon

            path_lat = 0.85 * path_lat + 0.15 * anchor_lat_path
            path_lon = 0.85 * path_lon + 0.15 * anchor_lon_path

        group.loc[start : end, "pred_lat"] = path_lat
        group.loc[start : end, "pred_lon"] = path_lon

    return group


def reconstruct_all(df: pd.DataFrame, schema: dict, cfg: dict) -> pd.DataFrame:
    voyage_col = schema["voyage_col"]
    parts = []

    for _, group in df.groupby(voyage_col, sort=False):
        parts.append(reconstruct_group(group, schema, cfg))

    return pd.concat(parts, axis=0).reset_index(drop=True)
