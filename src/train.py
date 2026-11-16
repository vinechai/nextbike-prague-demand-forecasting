"""train a lightgbm model on a features parquet and log to mlflow.

usage:
    python -m src.train \
        --features data/features_candidate.parquet \
        --output data/models/model_$(date +%Y%m%d).lgb \
        --cutoff 2026-07-26
"""

import argparse
import hashlib
import json
import os
from datetime import datetime
from pathlib import Path

import lightgbm as lgb
import mlflow
import numpy as np
import pandas as pd

FEATURE_COLS_FILE = os.environ.get("FEATURE_COLS_FILE", "data/feature_cols.json")

# matches notebook 03 final Optuna params (40 trials, val MAE objective)
_LGB_PARAMS = {
    "objective":         "regression_l1",
    "n_estimators":      1000,
    "learning_rate":     0.034,
    "num_leaves":        94,
    "min_child_samples": 72,
    "subsample":         0.747,
    "colsample_bytree":  0.554,
    "reg_lambda":        9.61,
    "reg_alpha":         0.0017,
    "random_state":      42,
    "n_jobs":            -1,
    "verbosity":         -1,
}


def load_feature_cols() -> list[str]:
    with open(FEATURE_COLS_FILE) as f:
        return json.load(f)


def train(
    features_path: str,
    output_path: str,
    train_cutoff: str,   # rows before this date are training, on/after are held out
    mlflow_run_name: str | None = None,
) -> dict:
    """train a lightgbm model. returns {"output": path, "train_mae": float, "val_mae": float}."""
    feature_cols = load_feature_cols()

    print(f"[train] loading {features_path}")
    df = pd.read_parquet(features_path)
    df["hour_local"] = pd.to_datetime(df.get("hour_local", df.get("hour", None)))

    cutoff = pd.Timestamp(train_cutoff)
    train_df = df[df["hour_local"] < cutoff].copy()
    val_df   = df[df["hour_local"] >= cutoff].copy()

    print(f"[train] train: {len(train_df):,} rows  val: {len(val_df):,} rows")

    present_cols = [c for c in feature_cols if c in df.columns]
    missing = [c for c in feature_cols if c not in df.columns]
    if missing:
        print(f"[train] missing features set to NaN: {missing}")

    target = "avg_available"
    X_train = train_df[present_cols]
    y_train = train_df[target]
    X_val   = val_df[present_cols]   if not val_df.empty else X_train[:0]
    y_val   = val_df[target]          if not val_df.empty else y_train[:0]

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    # station list — saved before training so it can be used by evaluate/promote
    from src.features import extract_station_stats_from_df
    station_uids = sorted(int(s) for s in train_df["station_uid"].unique())
    station_list_path = output_path.replace(".lgb", "_stations.json")
    with open(station_list_path, "w") as _f:
        json.dump(station_uids, _f)
    print(f"[train] {len(station_uids):,} stations -> {station_list_path}")

    # pre-compute per-station stats from training data before model training
    # (extract_station_stats_from_df needs train_df which has avail_std/reb_per_week columns)
    station_stats = extract_station_stats_from_df(train_df)

    mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI", "mlruns"))
    mlflow.set_experiment("nextbike_retrain")   # creates experiment dir if first run
    with mlflow.start_run(run_name=mlflow_run_name or "train_candidate"):
        mlflow.log_params({**_LGB_PARAMS, "train_rows": len(train_df), "val_rows": len(val_df)})

        model = lgb.LGBMRegressor(**_LGB_PARAMS)
        model.fit(X_train, y_train)

        train_mae = float(np.abs(model.predict(X_train) - y_train).mean())
        val_mae   = float(np.abs(model.predict(X_val) - y_val).mean()) if not val_df.empty else float("nan")

        mlflow.log_metrics({"train_mae": train_mae, "val_mae": val_mae})
        mlflow.log_artifact(output_path) if os.path.exists(output_path) else None

        print(f"[train] train MAE={train_mae:.4f}  val MAE={val_mae:.4f}")
        model.booster_.save_model(output_path)
        print(f"[train] saved → {output_path}")

        mlflow.log_artifact(output_path)

    # save station stats after model is written so we can embed its hash
    model_hash = hashlib.sha256(Path(output_path).read_bytes()).hexdigest()[:16]
    station_stats["_model_hash"] = model_hash
    stats_path = output_path.replace(".lgb", "_station_stats.json")
    with open(stats_path, "w") as _f:
        json.dump(station_stats, _f)
    print(f"[train] station stats (model_hash={model_hash}) -> {stats_path}")

    return {
        "output":      output_path,
        "stats_path":  stats_path,
        "train_mae":   train_mae,
        "val_mae":     val_mae,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", default="data/features_candidate.parquet")
    parser.add_argument("--output",   default=f"data/models/model_{datetime.now():%Y%m%d}.lgb")
    parser.add_argument("--cutoff",   default="2026-08-01")
    args = parser.parse_args()
    train(args.features, args.output, args.cutoff)
