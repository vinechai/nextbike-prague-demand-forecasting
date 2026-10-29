"""compare mart_station_hourly_features against features.parquet.

usage:
    cd dbt
    python check_mart.py

expects:
    - dbt_dev.mart_station_hourly_features already built (run `dbt run --profiles-dir .` first)
    - ../data/features.parquet from the training pipeline
    - DBT_DB_* env vars (or defaults: localhost:5432, user/pass nextbike, db nextbike)
"""

import os
import sys
import pandas as pd
import numpy as np

try:
    import psycopg2
except ImportError:
    print("psycopg2 not installed. run: pip install psycopg2-binary")
    sys.exit(1)

PARQUET = os.path.join(os.path.dirname(__file__), "..", "data", "features.parquet")

DB_PARAMS = {
    "host":     os.environ.get("DBT_DB_HOST",  "localhost"),
    "port":     int(os.environ.get("DBT_DB_PORT", "5432")),
    "user":     os.environ.get("DBT_DB_USER",  "nextbike"),
    "password": os.environ.get("DBT_DB_PASS",  "nextbike"),
    "dbname":   os.environ.get("DBT_DB_NAME",  "nextbike"),
}


def load_mart(conn) -> pd.DataFrame:
    sql = """
        select station_uid, hour_local, avg_available, lag_24h, lag_48h, lag_168h
        from dbt_dev.mart_station_hourly_features
        where hour_local >= '2026-04-01'
          and hour_local <  '2026-09-01'
        order by station_uid, hour_local
    """
    return pd.read_sql(sql, conn)


def load_parquet() -> pd.DataFrame:
    df = pd.read_parquet(PARQUET)
    # normalize column name for the target variable
    if "avg_available" not in df.columns and "avail" in df.columns:
        df = df.rename(columns={"avail": "avg_available"})
    # keep only columns that exist in both
    cols = ["station_uid", "hour", "avg_available", "lag_24h", "lag_48h", "lag_168h"]
    cols = [c for c in cols if c in df.columns]
    return df[cols].rename(columns={"hour": "hour_local"})


def compare(mart: pd.DataFrame, pq: pd.DataFrame) -> None:
    mart["hour_local"] = pd.to_datetime(mart["hour_local"]).dt.tz_localize(None)
    pq["hour_local"]   = pd.to_datetime(pq["hour_local"]).dt.tz_localize(None)

    merged = mart.merge(
        pq,
        on=["station_uid", "hour_local"],
        suffixes=("_mart", "_pq"),
        how="inner",
    )

    n = len(merged)
    print(f"\nmatched rows: {n:,}")
    if n == 0:
        print("no overlap — check that the mart was built and date ranges match")
        return

    for col in ["avg_available", "lag_24h", "lag_48h", "lag_168h"]:
        a = f"{col}_mart"
        b = f"{col}_pq"
        if a not in merged.columns or b not in merged.columns:
            continue
        diff = (merged[a] - merged[b]).dropna()
        pct_differ = (diff.abs() > 0.01).mean() * 100
        mae = diff.abs().mean()
        print(f"  {col:<20}  rows with diff>0.01: {pct_differ:5.1f}%   MAE: {mae:.4f}")

    print("""
notes:
  avg_available  — should be nearly identical (same aggregation)
  lag_*          — expected to differ for stations/periods with scraper gaps.
                   features.parquet uses time-offset joins; the mart uses lag() by rows.
""")


if __name__ == "__main__":
    if not os.path.exists(PARQUET):
        print(f"features.parquet not found at {PARQUET}")
        sys.exit(1)

    print(f"connecting to {DB_PARAMS['host']}:{DB_PARAMS['port']}/{DB_PARAMS['dbname']}...")
    try:
        conn = psycopg2.connect(**DB_PARAMS)
    except Exception as e:
        print(f"connection failed: {e}")
        sys.exit(1)

    print("loading mart...")
    try:
        mart = load_mart(conn)
    except Exception as e:
        print(f"mart query failed: {e}")
        print("did you run `dbt run --profiles-dir .` first?")
        sys.exit(1)
    finally:
        conn.close()

    print(f"mart rows: {len(mart):,}")

    print("loading features.parquet...")
    pq = load_parquet()
    print(f"parquet rows (filtered): {len(pq):,}")

    compare(mart, pq)
