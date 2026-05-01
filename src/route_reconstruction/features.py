from __future__ import annotations

import numpy as np
import pandas as pd


def _as_numeric(series: pd.Series, default: float = 0.0) -> pd.Series:
    return pd.to_numeric(series, errors="coerce").fillna(default)


def _angle_diff_deg(current: pd.Series, previous: pd.Series) -> pd.Series:
    diff = (current - previous + 180.0) % 360.0 - 180.0
    return diff


def _compute_dt(group: pd.DataFrame, time_col: str, clip_seconds: float) -> pd.Series:
    parsed = pd.to_datetime(group[time_col], errors="coerce")

    if parsed.notna().any():
        dt = parsed.diff().dt.total_seconds()
    else:
        values = pd.to_numeric(group[time_col], errors="coerce")
        dt = values.diff()

    dt = dt.replace([np.inf, -np.inf], np.nan)
    positive = dt[dt > 0]

    if len(positive) > 0:
        fill_value = float(positive.median())
    else:
        fill_value = 60.0

    dt = dt.fillna(fill_value)
    dt = dt.mask(dt <= 0, fill_value)
    dt = dt.clip(lower=1.0, upper=clip_seconds)

    return pd.Series(dt.values, index=group.index, dtype=float)


def _assign_groupwise_dt(df: pd.DataFrame, voyage_col: str, time_col: str, clip_seconds: float) -> pd.Series:
    dt_series = pd.Series(index=df.index, dtype=float)

    for _, group in df.groupby(voyage_col, sort=False):
        dt_series.loc[group.index] = _compute_dt(group, time_col, clip_seconds)

    return dt_series.sort_index()


def add_features(df: pd.DataFrame, schema: dict, cfg: dict) -> pd.DataFrame:
    df = df.copy()
    feature_cfg = cfg["features"]

    voyage_col = schema["voyage_col"]
    time_col = schema["time_col"]
    lat_col = schema["lat_col"]
    lon_col = schema["lon_col"]

    dt_clip = float(feature_cfg.get("dt_clip_seconds", 28800))
    sog_to_ms = float(feature_cfg.get("sog_to_ms", 0.514444))

    df["dt"] = _assign_groupwise_dt(
        df=df,
        voyage_col=voyage_col,
        time_col=time_col,
        clip_seconds=dt_clip,
    )

    for col in [lat_col, lon_col, "SOG", "COG", "ROT", "DRFT", "HD", "VSSL_LEN", "VSSL_WIDTH"]:
        if col not in df.columns:
            df[col] = np.nan

    df[lat_col] = _as_numeric(df[lat_col], 0.0)
    df[lon_col] = _as_numeric(df[lon_col], 0.0)
    df["SOG"] = _as_numeric(df["SOG"], 0.0)
    df["COG"] = _as_numeric(df["COG"], 0.0)
    df["ROT"] = _as_numeric(df["ROT"], 0.0)
    df["DRFT"] = _as_numeric(df["DRFT"], 0.0)
    df["HD"] = _as_numeric(df["HD"], 0.0)
    df["VSSL_LEN"] = _as_numeric(df["VSSL_LEN"], 0.0)
    df["VSSL_WIDTH"] = _as_numeric(df["VSSL_WIDTH"], 0.0)

    theta = np.deg2rad(df["COG"].values.astype(float))
    df["COG_sin"] = np.sin(theta)
    df["COG_cos"] = np.cos(theta)

    df["SOG_ms"] = df["SOG"] * sog_to_ms
    df["vx_obs"] = df["SOG_ms"] * df["COG_sin"]
    df["vy_obs"] = df["SOG_ms"] * df["COG_cos"]

    df["prev_vx"] = df.groupby(voyage_col)["vx_obs"].shift(1)
    df["prev_vy"] = df.groupby(voyage_col)["vy_obs"].shift(1)

    df["accel"] = (
        np.sqrt(
            (df["vx_obs"] - df["prev_vx"]).pow(2)
            + (df["vy_obs"] - df["prev_vy"]).pow(2)
        )
        / df["dt"].clip(lower=1.0)
    ).fillna(0.0)

    prev_cog = df.groupby(voyage_col)["COG"].shift(1)
    df["yaw_rate"] = (
        _angle_diff_deg(df["COG"], prev_cog)
        / df["dt"].clip(lower=1.0)
    ).fillna(0.0)

    speed = df["SOG_ms"].clip(lower=0.1)
    df["curvature"] = (
        np.abs(np.deg2rad(df["yaw_rate"]))
        / speed
    ).replace([np.inf, -np.inf], 0.0)
    df["curvature"] = df["curvature"].fillna(0.0)

    df["turn_radius"] = 1.0 / df["curvature"].replace(0, np.nan)
    df["turn_radius"] = df["turn_radius"].replace([np.inf, -np.inf], np.nan).fillna(1e6)

    volume_proxy = (
        df["VSSL_LEN"].clip(lower=1.0)
        * df["VSSL_WIDTH"].clip(lower=1.0)
    ).clip(lower=1.0)
    df["volume_log"] = np.log(volume_proxy)

    small_th = float(feature_cfg.get("small_volume_log_threshold", 5.694))
    large_th = float(feature_cfg.get("large_volume_log_threshold", 8.365))

    df["size_class"] = "MEDIUM"
    df.loc[df["volume_log"] < small_th, "size_class"] = "SMALL"
    df.loc[df["volume_log"] >= large_th, "size_class"] = "LARGE"

    df["size_SMALL"] = (df["size_class"] == "SMALL").astype(int)
    df["size_MEDIUM"] = (df["size_class"] == "MEDIUM").astype(int)
    df["size_LARGE"] = (df["size_class"] == "LARGE").astype(int)

    df = df.replace([np.inf, -np.inf], np.nan)
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df[numeric_cols] = df[numeric_cols].fillna(0.0)

    return df


def feature_columns() -> list[str]:
    return [
        "dt",
        "SOG_ms",
        "COG_sin",
        "COG_cos",
        "vx_obs",
        "vy_obs",
        "ROT",
        "DRFT",
        "HD",
        "accel",
        "yaw_rate",
        "curvature",
        "turn_radius",
        "VSSL_LEN",
        "VSSL_WIDTH",
        "volume_log",
        "size_SMALL",
        "size_MEDIUM",
        "size_LARGE",
    ]
