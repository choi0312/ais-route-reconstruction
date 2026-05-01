from __future__ import annotations

from pathlib import Path

from route_reconstruction.config import load_config
from route_reconstruction.data import infer_schema, load_data, sort_by_voyage_time
from route_reconstruction.features import add_features
from route_reconstruction.kalman import apply_kalman_rts
from route_reconstruction.reconstruction import reconstruct_all
from route_reconstruction.submission import make_submission
from route_reconstruction.utils import ensure_dir, save_json, seed_everything


def run_pipeline(config_path: str | Path) -> dict:
    cfg = load_config(config_path)
    seed_everything(int(cfg["project"].get("seed", 42)))

    output_dir = ensure_dir(cfg["project"].get("output_dir", "outputs"))

    data = load_data(cfg)
    test_df = data["test"]
    sample_submission = data["sample_submission"]

    schema = infer_schema(test_df, cfg)

    test_df = sort_by_voyage_time(test_df, schema)
    test_df = add_features(test_df, schema, cfg)

    if cfg["reconstruction"].get("use_kalman", True):
        test_df = apply_kalman_rts(test_df, schema, cfg)
    else:
        test_df["kf_lat"] = test_df[schema["lat_col"]]
        test_df["kf_lon"] = test_df[schema["lon_col"]]

    reconstructed = reconstruct_all(test_df, schema, cfg)

    reconstructed_path = output_dir / "reconstructed_test.csv"
    reconstructed.to_csv(reconstructed_path, index=False)

    submission_path = output_dir / cfg["submission"].get("filename", "submission.csv")
    submission = make_submission(
        reconstructed=reconstructed,
        sample_submission=sample_submission,
        schema=schema,
        output_path=submission_path,
    )

    summary = {
        "n_rows_test": int(len(test_df)),
        "n_target_rows": int((test_df[schema["target_col"]].astype(int) == 1).sum()),
        "n_submission_rows": int(len(submission)),
        "submission_path": str(submission_path),
        "reconstructed_path": str(reconstructed_path),
        "schema": schema,
    }

    save_json(summary, output_dir / "run_summary.json")

    return summary
