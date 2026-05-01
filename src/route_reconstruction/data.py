from __future__ import annotations

from pathlib import Path
from typing import Iterable

import pandas as pd


def read_csv_smart(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"CSV file not found: {path}")

    for enc in ["utf-8", "utf-8-sig", "cp949"]:
        try:
            return pd.read_csv(path, encoding=enc)
        except Exception:
            continue

    return pd.read_csv(path)


def choose_column(df: pd.DataFrame, candidates: Iterable[str], required: bool = True) -> str | None:
    for col in candidates:
        if col in df.columns:
            return col

    if required:
        raise ValueError(f"None of candidate columns found: {list(candidates)}")

    return None


def load_data(cfg: dict) -> dict:
    data_cfg = cfg["data"]

    out = {
        "test": read_csv_smart(data_cfg["test_path"]),
    }

    train_path = Path(data_cfg["train_path"])
    sample_path = Path(data_cfg["sample_submission_path"])

    if train_path.exists():
        out["train"] = read_csv_smart(train_path)
    else:
        out["train"] = None

    if sample_path.exists():
        out["sample_submission"] = read_csv_smart(sample_path)
    else:
        out["sample_submission"] = None

    return out


def infer_schema(df: pd.DataFrame, cfg: dict) -> dict:
    data_cfg = cfg["data"]

    voyage_col = choose_column(
        df,
        data_cfg.get("voyage_col_candidates", ["voyage_id"]),
        required=True,
    )
    time_col = choose_column(
        df,
        data_cfg.get("time_col_candidates", ["UPDT_TM"]),
        required=True,
    )
    id_col = choose_column(
        df,
        data_cfg.get("id_col_candidates", ["row_id", "ID", "id"]),
        required=False,
    )

    target_col = data_cfg.get("target_col", "IS_TARGET")
    lat_col = data_cfg.get("lat_col", "LAT_REL")
    lon_col = data_cfg.get("lon_col", "LON_REL")

    for col in [target_col, lat_col, lon_col]:
        if col not in df.columns:
            raise ValueError(f"Required column missing: {col}")

    return {
        "id_col": id_col,
        "voyage_col": voyage_col,
        "time_col": time_col,
        "target_col": target_col,
        "lat_col": lat_col,
        "lon_col": lon_col,
    }


def sort_by_voyage_time(df: pd.DataFrame, schema: dict) -> pd.DataFrame:
    df = df.copy()
    time_col = schema["time_col"]
    voyage_col = schema["voyage_col"]

    parsed = pd.to_datetime(df[time_col], errors="coerce")
    if parsed.notna().any():
        df["_time_sort"] = parsed
    else:
        df["_time_sort"] = pd.to_numeric(df[time_col], errors="coerce").fillna(0)

    if schema["id_col"] is not None and schema["id_col"] in df.columns:
        sort_cols = [voyage_col, "_time_sort", schema["id_col"]]
    else:
        sort_cols = [voyage_col, "_time_sort"]

    df = df.sort_values(sort_cols).reset_index(drop=True)
    return df
