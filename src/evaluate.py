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
    """returns {"candidate_mae", "reference_mae", "is_better", "delta"}."""
    cols = _load_cols()

    df = pd.read_parquet(features_path)
    df["hour_local"] = pd.to_datetime(df.get("hour_local", df.get("hour", None)))
    window = df[(df["hour_local"] >= since) & (df["hour_local"] < until)].copy()

    if window.empty:
        raise ValueError(f"no rows in [{since}, {until}) — check features file")

    present = [c for c in cols if c in window.columns]
    X = window[present]
    y = window["avg_available"]

    cand = lgb.Booster(model_file=candidate_path)
    ref  = lgb.Booster(model_file=reference_path)

    cand_mae = float(np.abs(cand.predict(X) - y).mean())
    ref_mae  = float(np.abs(ref.predict(X)  - y).mean())
    delta    = ref_mae - cand_mae   # positive means candidate is better

    result = {
        "candidate_mae": round(cand_mae, 4),
        "reference_mae": round(ref_mae,  4),
        "is_better":     bool(cand_mae < ref_mae),
        "delta":         round(delta, 4),
    }

    status = "BETTER" if result["is_better"] else "WORSE"
    print(
        f"[evaluate] candidate MAE={cand_mae:.4f}  reference MAE={ref_mae:.4f}"
        f"  → {status} by {abs(delta):.4f}"
    )
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
