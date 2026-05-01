from __future__ import annotations

from pathlib import Path

import pandas as pd


def make_submission(
    reconstructed: pd.DataFrame,
    sample_submission: pd.DataFrame | None,
    schema: dict,
    output_path: str | Path,
) -> pd.DataFrame:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    target_col = schema["target_col"]
    lat_col = schema["lat_col"]
    lon_col = schema["lon_col"]
    id_col = schema["id_col"]

    target_rows = reconstructed[reconstructed[target_col].astype(int) == 1].copy()

    if sample_submission is None:
        submission = target_rows[[id_col, "pred_lat", "pred_lon"]].copy() if id_col else target_rows[["pred_lat", "pred_lon"]].copy()
        submission = submission.rename(columns={"pred_lat": lat_col, "pred_lon": lon_col})
        submission.to_csv(output_path, index=False)
        return submission

    submission = sample_submission.copy()

    lat_candidates = [lat_col, "LAT", "lat", "y"]
    lon_candidates = [lon_col, "LON", "lon", "x"]

    sub_lat_col = next((c for c in lat_candidates if c in submission.columns), None)
    sub_lon_col = next((c for c in lon_candidates if c in submission.columns), None)

    if sub_lat_col is None or sub_lon_col is None:
        if len(submission.columns) >= 3:
            sub_lat_col = submission.columns[-2]
            sub_lon_col = submission.columns[-1]
        else:
            raise ValueError("Cannot infer submission coordinate columns.")

    if id_col is not None and id_col in submission.columns and id_col in target_rows.columns:
        pred = target_rows[[id_col, "pred_lat", "pred_lon"]].copy()
        submission = submission.drop(columns=[sub_lat_col, sub_lon_col], errors="ignore")
        submission = submission.merge(pred, on=id_col, how="left")
        submission[sub_lat_col] = submission["pred_lat"]
        submission[sub_lon_col] = submission["pred_lon"]
        submission = submission.drop(columns=["pred_lat", "pred_lon"], errors="ignore")
    else:
        if len(submission) != len(target_rows):
            n = min(len(submission), len(target_rows))
            submission.loc[: n - 1, sub_lat_col] = target_rows["pred_lat"].values[:n]
            submission.loc[: n - 1, sub_lon_col] = target_rows["pred_lon"].values[:n]
        else:
            submission[sub_lat_col] = target_rows["pred_lat"].values
            submission[sub_lon_col] = target_rows["pred_lon"].values

    submission.to_csv(output_path, index=False)
    return submission
