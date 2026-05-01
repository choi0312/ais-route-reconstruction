import pandas as pd
from pathlib import Path

from route_reconstruction.pipeline import run_pipeline


def test_pipeline_synthetic(tmp_path):
    raw = tmp_path / "raw"
    raw.mkdir(parents=True)

    test = pd.DataFrame(
        {
            "row_id": [1, 2, 3, 4, 5],
            "voyage_id": ["v1"] * 5,
            "UPDT_TM": pd.date_range("2025-01-01", periods=5, freq="min").astype(str),
            "IS_TARGET": [0, 1, 1, 1, 0],
            "LAT_REL": [0.0, 0.0, 0.0, 0.0, 40.0],
            "LON_REL": [0.0, 0.0, 0.0, 0.0, 20.0],
            "SOG": [10.0] * 5,
            "COG": [45.0] * 5,
        }
    )

    sample = pd.DataFrame(
        {
            "row_id": [2, 3, 4],
            "LAT_REL": [0.0, 0.0, 0.0],
            "LON_REL": [0.0, 0.0, 0.0],
        }
    )

    test.to_csv(raw / "test.csv", index=False)
    sample.to_csv(raw / "sample_submission.csv", index=False)

    config = tmp_path / "config.yaml"
    config.write_text(
        f"""
project:
  seed: 42
  output_dir: {tmp_path / "outputs"}
data:
  train_path: {raw / "train.csv"}
  test_path: {raw / "test.csv"}
  sample_submission_path: {raw / "sample_submission.csv"}
  id_col_candidates: [row_id]
  voyage_col_candidates: [voyage_id]
  time_col_candidates: [UPDT_TM]
  target_col: IS_TARGET
  lat_col: LAT_REL
  lon_col: LON_REL
features:
  dt_clip_seconds: 28800
  sog_to_ms: 0.514444
  small_volume_log_threshold: 5.694
  large_volume_log_threshold: 8.365
kalman:
  process_noise_base: 0.35
  measurement_position_noise: 12.0
  measurement_velocity_noise: 2.0
  speed_noise_scale: 0.08
reconstruction:
  use_kalman: true
  use_gaussian_center_weight: true
  short_gap_threshold: 2
  alpha_base: 0.55
  alpha_gap_gain: 0.20
  alpha_curve_gain: 0.20
  delta_clip_factor: 1.8
submission:
  filename: submission.csv
        """,
        encoding="utf-8",
    )

    result = run_pipeline(config)
    assert Path(result["submission_path"]).exists()
