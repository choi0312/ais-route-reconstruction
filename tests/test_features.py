import pandas as pd
import numpy as np

from route_reconstruction.features import add_features


def test_add_features_basic():
    df = pd.DataFrame(
        {
            "voyage_id": ["v1", "v1", "v1"],
            "UPDT_TM": ["2025-01-01 00:00:00", "2025-01-01 00:01:00", "2025-01-01 00:02:00"],
            "IS_TARGET": [0, 1, 0],
            "LAT_REL": [0.0, 10.0, 20.0],
            "LON_REL": [0.0, 5.0, 10.0],
            "SOG": [10.0, 10.0, 10.0],
            "COG": [90.0, 90.0, 90.0],
        }
    )

    schema = {
        "voyage_col": "voyage_id",
        "time_col": "UPDT_TM",
        "target_col": "IS_TARGET",
        "lat_col": "LAT_REL",
        "lon_col": "LON_REL",
    }

    cfg = {
        "features": {
            "dt_clip_seconds": 28800,
            "sog_to_ms": 0.514444,
            "small_volume_log_threshold": 5.694,
            "large_volume_log_threshold": 8.365,
        }
    }

    out = add_features(df, schema, cfg)

    assert "dt" in out.columns
    assert "COG_sin" in out.columns
    assert "vx_obs" in out.columns
    assert np.isfinite(out["dt"]).all()
