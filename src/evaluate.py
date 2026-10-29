"""compare a candidate model against a reference model on a held-out window.

usage:
    python -m src.evaluate \
        --candidate data/models/model_20261029.lgb \
        --reference data/model.lgb \
        --features  data/features.parquet \
        --since 2026-08-01 \
        --until 2026-09-01
"""

import argparse
import json
import os

import lightgbm as lgb
import numpy as np
import pandas as pd

FEATURE_COLS_FILE = os.environ.get("FEATURE_COLS_FILE", "data/feature_cols.json")

_MIN_EVAL_ROWS = 10_000   # fewer rows means the time window is wrong


def _load_cols() -> list[str]:
    with open(FEATURE_COLS_FILE) as f:
        return json.load(f)


def evaluate(
    candidate_path: str,
    reference_path: str,
    features_path: str,
    since: str,
    until: str,
) -> dict:
    """returns {"candidate_mae", "candidate_r2", "reference_mae", "reference_r2", "is_better", "delta"}."""
    cols = _load_cols()

    df = pd.read_parquet(features_path)
    df["hour_local"] = pd.to_datetime(df.get("hour_local", df.get("hour", None)))
    window = df[(df["hour_local"] >= since) & (df["hour_local"] < until)].copy()
    window = window.dropna(subset=["avg_available"])

    print(
        f"[evaluate] eval [{since}, {until}): {len(window):,} rows, "
        f"{window['station_uid'].nunique()} stations, "
        f"range {window['hour_local'].min()} to {window['hour_local'].max()}"
    )

    if len(window) < _MIN_EVAL_ROWS:
        raise ValueError(
            f"eval set too small: {len(window):,} rows < {_MIN_EVAL_ROWS:,} minimum. "
            f"features file covers {df['hour_local'].min()} to {df['hour_local'].max()}. "
            f"build_features must cover both TRAIN_SINCE and EVAL_UNTIL."
        )

    present = [c for c in cols if c in window.columns]
    missing = [c for c in cols if c not in window.columns]
    if missing:
        print(f"[evaluate] missing features (NaN): {missing}")

    X = window[present]
    y = window["avg_available"].values
    ss_tot = float(np.sum((y - y.mean()) ** 2))

    def _metrics(pred):
        mae = float(np.abs(pred - y).mean())
        r2  = float(1.0 - np.sum((pred - y) ** 2) / ss_tot)
        return mae, r2

    cand = lgb.Booster(model_file=candidate_path)
    ref  = lgb.Booster(model_file=reference_path)

    cand_mae, cand_r2 = _metrics(cand.predict(X))
    ref_mae,  ref_r2  = _metrics(ref.predict(X))
    delta = ref_mae - cand_mae   # positive = candidate is better

    # naive baselines
    for lag_col in ("lag_24h", "lag_168h"):
        if lag_col in window.columns:
            vals = window[lag_col].values
            mask = ~np.isnan(vals)
            if mask.any():
                b_mae = float(np.abs(vals[mask] - y[mask]).mean())
                b_r2  = float(1.0 - np.sum((vals[mask] - y[mask])**2) /
                              np.sum((y[mask] - y[mask].mean())**2))
                print(f"[evaluate] naive {lag_col}: MAE={b_mae:.4f}  R2={b_r2:.4f}  (n={mask.sum():,})")

    result = {
        "candidate_mae": round(cand_mae, 4),
        "candidate_r2":  round(cand_r2,  4),
        "reference_mae": round(ref_mae,  4),
        "reference_r2":  round(ref_r2,   4),
        "is_better":     bool(cand_mae < ref_mae),
        "delta":         round(delta, 4),
    }

    status = "BETTER" if result["is_better"] else "WORSE"
    print(f"[evaluate] candidate: MAE={cand_mae:.4f}  R2={cand_r2:.4f}")
    print(f"[evaluate] reference: MAE={ref_mae:.4f}  R2={ref_r2:.4f}  -> candidate {status} by {abs(delta):.4f}")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate",  required=True)
    parser.add_argument("--reference",  default="data/model.lgb")
    parser.add_argument("--features",   default="data/features.parquet")
    parser.add_argument("--since",      default="2026-08-01")
    parser.add_argument("--until",      default="2026-09-01")
    args = parser.parse_args()
    result = evaluate(args.candidate, args.reference, args.features, args.since, args.until)
    print(result)
