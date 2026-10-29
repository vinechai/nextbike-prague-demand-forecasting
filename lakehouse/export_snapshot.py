"""export raw postgres tables to date-partitioned parquet files.

this step does NOT require java — it's pure pandas/psycopg2.

usage:
    python lakehouse/export_snapshot.py --since 2026-04-01 --until 2026-09-01

outputs go to lakehouse/raw/ (gitignored).
"""

import argparse
import os
from datetime import date, timedelta

import pandas as pd
import psycopg2

_RAW_DIR = os.path.join(os.path.dirname(__file__), "raw")

DB = {
    "host":     os.environ.get("DB_HOST", "localhost"),
    "port":     int(os.environ.get("DB_PORT", "5432")),
    "user":     os.environ.get("DB_USER", "nextbike"),
    "password": os.environ.get("DB_PASS", "nextbike"),
    "dbname":   os.environ.get("DB_NAME", "nextbike"),
}


def _export_snapshots(conn, d: date) -> int:
    next_d = d + timedelta(days=1)
    sql = """
        select id, station_uid, bikes_available, bikes_available_to_rent,
               free_racks, booked_bikes, scrape_time
        from station_snapshots
        where scrape_time >= %s and scrape_time < %s
    """
    df = pd.read_sql(sql, conn, params=(str(d), str(next_d)))
    if df.empty:
        return 0
    out_dir = os.path.join(_RAW_DIR, "station_snapshots", f"date={d}")
    os.makedirs(out_dir, exist_ok=True)
    df.to_parquet(os.path.join(out_dir, "part-0.parquet"), index=False)
    return len(df)


def _export_weather(conn, d: date) -> int:
    next_d = d + timedelta(days=1)
    sql = "select * from weather_hourly where hour >= %s and hour < %s"
    df = pd.read_sql(sql, conn, params=(str(d), str(next_d)))
    if df.empty:
        return 0
    out_dir = os.path.join(_RAW_DIR, "weather_hourly", f"date={d}")
    os.makedirs(out_dir, exist_ok=True)
    df.to_parquet(os.path.join(out_dir, "part-0.parquet"), index=False)
    return len(df)


def run(since: str, until: str) -> None:
    conn = psycopg2.connect(**DB)
    current = date.fromisoformat(since)
    end     = date.fromisoformat(until)
    total_snap = total_wx = 0
    while current < end:
        n_snap = _export_snapshots(conn, current)
        n_wx   = _export_weather(conn, current)
        if n_snap or n_wx:
            print(f"  {current}  snapshots={n_snap}  weather={n_wx}")
        total_snap += n_snap
        total_wx   += n_wx
        current += timedelta(days=1)
    conn.close()
    print(f"done — {total_snap:,} snapshot rows, {total_wx:,} weather rows exported to {_RAW_DIR}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--since", default="2026-04-01")
    parser.add_argument("--until", default="2026-09-01")
    args = parser.parse_args()
    run(args.since, args.until)
