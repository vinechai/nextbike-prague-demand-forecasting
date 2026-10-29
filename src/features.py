"""build a features dataframe from local postgres for a given date range.

simplified vs the full notebook pipeline:
  - rebalancing features (hours_since_reb, reb_per_week, avail_std, zero_rate_3d,
    near_zero_rate_3d) are set to NaN — they require the full heuristic pipeline
  - geo features (dist_metro_m etc.) are merged from a reference parquet
  - lag features use explicit time-offset merge to handle scraper gaps

usage:
    from src.features import build_features
    df = build_features(
        since="2026-07-01",
        until="2026-08-31",
        geo_parquet="data/features.parquet",
    )
"""

import os
from datetime import timedelta

import numpy as np
import pandas as pd

from src.db import get_conn

try:
    import holidays as _holidays_lib
    _CZ_HOLIDAYS = _holidays_lib.Czechia()
except ImportError:
    _CZ_HOLIDAYS = {}


_GEO_COLS = [
    "lat", "lng", "bike_racks", "district",
    "dist_metro_m", "n_metro_500m", "dist_tram_m", "n_tram_300m",
    "dist_cafe_m", "n_cafe_300m", "dist_park_m", "n_park_500m",
    "dist_office_m", "n_office_500m", "is_in_park", "elevation_m",
]

_WEATHER_COLS = [
    "temperature", "precipitation", "windspeed",
    "weathercode", "snowfall", "shortwave_radiation", "tsun",
]

_LAG_OFFSETS = {"lag_24h": 24, "lag_48h": 48, "lag_168h": 168}


def _query_hourly(since: str, until: str) -> pd.DataFrame:
    """aggregate snapshots to one row per (station_uid, hour_local)."""
    sql = """
        select
            station_uid,
            date_trunc('hour', scrape_time at time zone 'Europe/Prague') as hour_local,
            round(avg(bikes_available_to_rent)::numeric, 4) as avg_available
        from station_snapshots
        where scrape_time >= %s::timestamptz
          and scrape_time <  %s::timestamptz
          and bikes_available_to_rent >= 0
        group by station_uid, hour_local
        order by station_uid, hour_local
    """
    with get_conn() as conn:
        return pd.read_sql(sql, conn, params=(since, until))


def _query_weather(since: str, until: str) -> pd.DataFrame:
    sql = """
        select *
        from weather_hourly
        where hour >= %s::timestamptz
          and hour <  %s::timestamptz
        order by hour
    """
    with get_conn() as conn:
        df = pd.read_sql(sql, conn, params=(since, until))
    df["hour_local"] = (
        pd.to_datetime(df["hour"], utc=True)
        .dt.tz_convert("Europe/Prague")
        .dt.tz_localize(None)
        .dt.floor("h")
    )
    return df


def _add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    h = df["hour_local"]
    df["hour_of_day"] = h.dt.hour
    df["dow"]         = h.dt.dayofweek
    df["month"]       = h.dt.month
    df["is_weekend"]  = (df["dow"] >= 5).astype(int)
    df["is_holiday"]  = h.dt.date.map(
        lambda d: int(d in _CZ_HOLIDAYS)
    ).astype(int)

    # cyclical encoding
    df["hour_sin"]  = np.sin(2 * np.pi * df["hour_of_day"] / 24)
    df["hour_cos"]  = np.cos(2 * np.pi * df["hour_of_day"] / 24)
    df["dow_sin"]   = np.sin(2 * np.pi * df["dow"] / 7)
    df["dow_cos"]   = np.cos(2 * np.pi * df["dow"] / 7)
    df["month_sin"] = np.sin(2 * np.pi * (df["month"] - 1) / 12)
    df["month_cos"] = np.cos(2 * np.pi * (df["month"] - 1) / 12)
    return df


def _add_lag_features(df: pd.DataFrame) -> pd.DataFrame:
    """time-offset lags via timestamp merge — NULL when hour is missing (handles scraper gaps)."""
    df = df.sort_values(["station_uid", "hour_local"])
    for col, offset_h in _LAG_OFFSETS.items():
        shifted = df[["station_uid", "hour_local", "avg_available"]].copy()
        shifted["hour_local"] = shifted["hour_local"] + timedelta(hours=offset_h)
        shifted = shifted.rename(columns={"avg_available": col})
        df = df.merge(shifted, on=["station_uid", "hour_local"], how="left")
    # lag_24h_ratio: matches the inference API formula (lag_24h / station_mean + 0.1).
    # using per-station mean of lag_24h (no target in numerator) avoids training-inference skew.
    with np.errstate(invalid="ignore"):
        station_mean = df.groupby("station_uid")["avg_available"].transform("mean")
        df["lag_24h_ratio"] = df["lag_24h"] / (station_mean + 0.1)
    return df


def _add_geo_features(df: pd.DataFrame, geo_parquet: str) -> pd.DataFrame:
    """merge station-level geo features from a reference parquet."""
    if not os.path.exists(geo_parquet):
        for col in _GEO_COLS:
            df[col] = np.nan
        return df

    ref = pd.read_parquet(geo_parquet)
    geo_cols_present = [c for c in _GEO_COLS if c in ref.columns]
    if "station_uid" not in ref.columns:
        for col in _GEO_COLS:
            df[col] = np.nan
        return df

    geo = ref[["station_uid"] + geo_cols_present].drop_duplicates("station_uid")
    df = df.merge(geo, on="station_uid", how="left")
    for col in _GEO_COLS:
        if col not in df.columns:
            df[col] = np.nan
    return df


def build_features(
    since: str,
    until: str,
    geo_parquet: str = "data/features.parquet",
) -> pd.DataFrame:
    """build feature table for [since, until) from local postgres.

    rebalancing features are set to NaN (see module docstring).
    """
    print(f"[features] querying hourly data {since} → {until}")
    hourly = _query_hourly(since, until)
    print(f"[features] {len(hourly):,} station-hours loaded")

    hourly["hour_local"] = pd.to_datetime(hourly["hour_local"]).dt.tz_localize(None)

    print("[features] adding time features")
    hourly = _add_time_features(hourly)

    print("[features] joining weather")
    weather = _query_weather(since, until)
    weather_cols = [c for c in _WEATHER_COLS if c in weather.columns]
    hourly = hourly.merge(
        weather[["hour_local"] + weather_cols],
        on="hour_local",
        how="left",
    )
    # columns that might not exist in local db
    for col in _WEATHER_COLS:
        if col not in hourly.columns:
            hourly[col] = np.nan

    print("[features] adding lag features")
    hourly = _add_lag_features(hourly)

    print("[features] merging geo features")
    hourly = _add_geo_features(hourly, geo_parquet)

    # rebalancing features — not computed here
    for col in ["hours_since_reb", "reb_per_week", "avail_std",
                "zero_rate_3d", "near_zero_rate_3d"]:
        hourly[col] = np.nan

    print(f"[features] done: {len(hourly):,} rows, {hourly.shape[1]} columns")
    return hourly
