"""build a features dataframe from local postgres for a given date range.

vs the full notebook pipeline:
  - rebalancing features (hours_since_reb, reb_per_week, avail_std, zero_rate_3d,
    near_zero_rate_3d) are now computed using the same 4-rule heuristic as notebook 02
  - geo features (dist_metro_m etc.) are merged from a reference parquet
  - lag features use explicit time-offset merge to handle scraper gaps
  - only is_spot=TRUE stations are included (same as notebook)
  - only stations with mean availability >= 0.5 are included (same as notebook)

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

_OFF_PEAK = frozenset(range(0, 7)) | frozenset(range(21, 24))

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
    """aggregate snapshots to one row per (station_uid, hour_local), is_spot=TRUE only."""
    sql = """
        select
            ss.station_uid,
            date_trunc('hour', ss.scrape_time at time zone 'Europe/Prague') as hour_local,
            round(avg(ss.bikes_available_to_rent)::numeric, 4) as avg_available
        from station_snapshots ss
        join stations s on s.uid = ss.station_uid and s.is_spot = true
        where ss.scrape_time >= %s::timestamptz
          and ss.scrape_time <  %s::timestamptz
          and ss.bikes_available_to_rent >= 0
        group by ss.station_uid, hour_local
        order by ss.station_uid, hour_local
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

    df["hour_sin"]  = np.sin(2 * np.pi * df["hour_of_day"] / 24)
    df["hour_cos"]  = np.cos(2 * np.pi * df["hour_of_day"] / 24)
    df["dow_sin"]   = np.sin(2 * np.pi * df["dow"] / 7)
    df["dow_cos"]   = np.cos(2 * np.pi * df["dow"] / 7)
    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)
    return df


def _add_lag_features(
    df: pd.DataFrame,
    station_mean_avail: "pd.Series | None" = None,
) -> pd.DataFrame:
    """time-offset lags via timestamp merge with +/-1h tolerance for scraper gaps.

    matches the notebook: tries exact offset first, then -1h, then +1h.
    a gap of exactly 1h in the scrape series is still filled; larger gaps are NaN.

    station_mean_avail: optional Series indexed by station_uid for lag_24h_ratio.
    if None, computed from the build window (less accurate for short windows).
    """
    df = df.sort_values(["station_uid", "hour_local"])
    lag_src = df[["station_uid", "hour_local", "avg_available"]].copy()
    for col, offset_h in _LAG_OFFSETS.items():
        df[col] = np.nan
        for delta_h in (0, -1, 1):
            missing = df[col].isna()
            if not missing.any():
                break
            lookup = (
                lag_src.assign(hour_local=lag_src["hour_local"] + timedelta(hours=offset_h + delta_h))
                .rename(columns={"avg_available": col})
            )
            patch = (
                df.loc[missing, ["station_uid", "hour_local"]]
                .merge(lookup, on=["station_uid", "hour_local"], how="left")
            )
            df.loc[missing, col] = patch[col].values

    # lag_24h_ratio: lag_24h / (station_mean_avail + 0.1) — matches API formula.
    # use pre-computed station means from the reference parquet when available (broader window);
    # fall back to computing from this build window (leakage-guarded to exclude last 6 weeks).
    if station_mean_avail is not None:
        sm = df["station_uid"].map(station_mean_avail)
    else:
        safe_end = df["hour_local"].max() - pd.Timedelta(weeks=6)
        safe_df  = df[df["hour_local"] < safe_end] if (df["hour_local"].min() < safe_end) else df
        sm = safe_df.groupby("station_uid")["avg_available"].mean().reindex(df["station_uid"]).values
    with np.errstate(invalid="ignore"):
        df["lag_24h_ratio"] = df["lag_24h"] / (sm + 0.1)
    return df


def _load_station_stats(geo_parquet: str) -> "dict[str, pd.Series]":
    """load per-station statistics from the reference parquet.

    returns a dict of Series indexed by station_uid:
      'mean_avail': used for lag_24h_ratio denominator
      'avail_std':  station volatility feature
      'reb_per_week': rebalancing frequency feature

    if geo_parquet doesn't contain these columns, returns empty dict.
    the API predictor reads the same values from features.parquet at startup.
    """
    stats: dict = {}
    if not os.path.exists(geo_parquet):
        return stats
    try:
        ref = pd.read_parquet(geo_parquet, columns=["station_uid", "avg_available",
                                                     "avail_std", "reb_per_week"])
    except Exception:
        try:
            ref = pd.read_parquet(geo_parquet, columns=["station_uid", "avg_available"])
        except Exception:
            return stats
    if "avg_available" in ref.columns:
        stats["mean_avail"] = ref.groupby("station_uid")["avg_available"].mean()
    if "avail_std" in ref.columns:
        stats["avail_std"] = ref.groupby("station_uid")["avail_std"].mean()
    if "reb_per_week" in ref.columns:
        stats["reb_per_week"] = ref.groupby("station_uid")["reb_per_week"].median()
    return stats


def _add_geo_features(df: pd.DataFrame, geo_parquet: str) -> pd.DataFrame:
    """merge station-level geo features.

    prefers data/station_geo.parquet (all 1,513 is_spot stations, pure geo only)
    over the full features.parquet to avoid accidentally merging time-varying columns.
    """
    # prefer the dedicated geo parquet if it exists
    geo_path = "data/station_geo.parquet"
    if not os.path.exists(geo_path):
        geo_path = geo_parquet
    if not os.path.exists(geo_path):
        for col in _GEO_COLS:
            df[col] = np.nan
        return df

    ref = pd.read_parquet(geo_path)
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


def _detect_rebalancing(hourly_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """detect rebalancing events using the same 4-rule heuristic as notebook 02.

    returns:
        reb_events: DataFrame with columns [station_uid, reb_hour] — one row per event
        distorted: DataFrame with columns [station_uid, hour_local] — hours to exclude
    """
    import sys
    import os as _os
    # add project root so utils.py is importable
    _proj_root = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
    if _proj_root not in sys.path:
        sys.path.insert(0, _proj_root)
    from utils import flag_rebalancing

    # rules 1 + 2: trip-based (uses bike_positions table)
    sql_trips = """
        WITH ordered AS (
            SELECT
                bike_id,
                station_uid                                                           AS to_station,
                first_seen_at                                                         AS arrive_time,
                LAG(station_uid)  OVER (PARTITION BY bike_id ORDER BY first_seen_at) AS from_station,
                LAG(last_seen_at) OVER (PARTITION BY bike_id ORDER BY first_seen_at) AS depart_time
            FROM bike_positions
        )
        SELECT o.bike_id, o.from_station, o.to_station, o.depart_time, o.arrive_time,
               sf.name AS name_from, st.name AS name_to
        FROM ordered o
        JOIN stations sf ON sf.uid = o.from_station
        JOIN stations st ON st.uid = o.to_station
        WHERE o.from_station IS NOT NULL
          AND sf.is_spot = TRUE AND st.is_spot = TRUE
    """
    with get_conn() as conn:
        trips = pd.read_sql(sql_trips, conn)

    trips["arrive_time"] = pd.to_datetime(trips["arrive_time"], utc=True)
    trips["depart_time"] = pd.to_datetime(trips["depart_time"], utc=True)
    trips = flag_rebalancing(trips)

    # rule 1: batch >= 3 bikes or known uphill pair, outside peak hours
    reb_base = trips[trips["is_rebalancing"]][["to_station", "arrive_time"]].drop_duplicates().copy()
    reb_base["reb_hour"] = (
        reb_base["arrive_time"].dt.tz_convert("Europe/Prague")
        .dt.floor("h").dt.tz_localize(None)
    )

    # rule 2: destination mass arrival >= 5 bikes, off-peak only
    trips["arrive_hour_local"] = trips["arrive_time"].dt.tz_convert("Europe/Prague").dt.hour
    off_peak_trips = trips[trips["arrive_hour_local"].isin(_OFF_PEAK)]
    dest_counts = (
        off_peak_trips.groupby(["to_station", "arrive_time"])["bike_id"]
        .count().reset_index(name="n_dest")
    )
    mass_dest = dest_counts[dest_counts["n_dest"] >= 5][["to_station", "arrive_time"]].copy()
    mass_dest["reb_hour"] = (
        mass_dest["arrive_time"].dt.tz_convert("Europe/Prague")
        .dt.floor("h").dt.tz_localize(None)
    )

    events = pd.concat(
        [reb_base[["to_station", "reb_hour"]], mass_dest[["to_station", "reb_hour"]]],
        ignore_index=True,
    ).drop_duplicates(subset=["to_station", "reb_hour"])

    # rule 4: slow delivery — rise >= 8 bikes in 1h, off-peak
    h_sorted = hourly_df.sort_values(["station_uid", "hour_local"])
    avail_rise = h_sorted.groupby("station_uid")["avg_available"].diff()
    off_peak_hr = h_sorted["hour_local"].dt.hour.isin(_OFF_PEAK)
    slow = h_sorted[(avail_rise >= 8) & off_peak_hr][["station_uid", "hour_local"]].copy()
    slow.columns = ["to_station", "reb_hour"]
    events = pd.concat([events, slow], ignore_index=True).drop_duplicates(subset=["to_station", "reb_hour"])

    # rule 3: large availability drop >= 5 bikes (collection by truck)
    avail_diff = h_sorted.groupby("station_uid")["avg_available"].diff()
    if "n_metro_500m" in h_sorted.columns:
        is_metro = h_sorted["n_metro_500m"] >= 1
        rule3_window = off_peak_hr | ~is_metro
    else:
        rule3_window = off_peak_hr
    drops = h_sorted[(avail_diff <= -5) & rule3_window][["station_uid", "hour_local"]].copy()
    drops.columns = ["to_station", "reb_hour"]
    events = pd.concat([events, drops], ignore_index=True).drop_duplicates(subset=["to_station", "reb_hour"])

    # expand each event to cover arrival hour + 3 following hours
    expanded = pd.concat(
        [events.assign(hour_local=events["reb_hour"] + pd.Timedelta(hours=off))
         for off in range(4)],
        ignore_index=True,
    )[["to_station", "hour_local"]].drop_duplicates()
    expanded = expanded.rename(columns={"to_station": "station_uid"})

    reb_events = events.rename(columns={"to_station": "station_uid"})
    print(f"[rebalancing] {len(reb_events):,} events -> {len(expanded):,} distorted station-hours")
    return reb_events, expanded


def _add_rebalancing_features(
    df: pd.DataFrame,
    reb_events: pd.DataFrame,
    distorted: pd.DataFrame,
) -> pd.DataFrame:
    """add hours_since_reb, reb_per_week, avail_std, zero_rate_3d, near_zero_rate_3d.

    reb_events: columns [station_uid, reb_hour]
    distorted:  columns [station_uid, hour_local]
    """
    # hours_since_reb: time since last event at this station, capped at 24h
    reb_times = (
        reb_events[["station_uid", "reb_hour"]]
        .drop_duplicates()
        .rename(columns={"reb_hour": "last_reb_hour"})
        .sort_values("last_reb_hour")
        .reset_index(drop=True)
    )
    feat_sorted = df.sort_values("hour_local").reset_index(drop=True)
    merged = pd.merge_asof(
        feat_sorted,
        reb_times,
        left_on="hour_local",
        right_on="last_reb_hour",
        by="station_uid",
        direction="backward",
    )
    merged["hours_since_reb"] = (merged["hour_local"] - merged["last_reb_hour"]) / pd.Timedelta(hours=1)
    merged.loc[merged["hours_since_reb"] > 24, "hours_since_reb"] = np.nan
    df = merged.drop(columns=["last_reb_hour"]).sort_values(["station_uid", "hour_local"]).reset_index(drop=True)

    # replace lag_24h with lag_48h when the 24h look-back hour was right after a reb event
    # (rebalancing-inflated availability in the look-back distorts the lag value)
    past_hsr = df.groupby("station_uid")["hours_since_reb"].shift(24)
    contaminated = past_hsr.notna() & (past_hsr < 4)
    n_cont = int(contaminated.sum())
    if n_cont > 0:
        df.loc[contaminated, "lag_24h"] = df.loc[contaminated, "lag_48h"]
        print(f"[rebalancing] replaced {n_cont:,} contaminated lag_24h values with lag_48h")

    # reb_per_week: per-station rebalancing frequency
    total_weeks = max((df["hour_local"].max() - df["hour_local"].min()).days / 7, 1)
    station_reb = (
        reb_events.groupby("station_uid").size()
        .reset_index(name="reb_count")
    )
    station_reb["reb_per_week"] = station_reb["reb_count"] / total_weeks
    df = df.merge(station_reb[["station_uid", "reb_per_week"]], on="station_uid", how="left")
    df["reb_per_week"] = df["reb_per_week"].fillna(0)

    # avail_std: per-station volatility on training-safe period (exclude last 6 weeks)
    safe_end = df["hour_local"].max() - pd.Timedelta(weeks=6)
    safe_df  = df[df["hour_local"] < safe_end] if (df["hour_local"].min() < safe_end) else df
    avail_std = (
        safe_df.groupby("station_uid")["avg_available"]
        .std()
        .reset_index(name="avail_std")
    )
    df = df.merge(avail_std, on="station_uid", how="left")

    # zero_rate_3d and near_zero_rate_3d: backward-looking rolling rates (no leakage by design)
    df = df.sort_values(["station_uid", "hour_local"]).reset_index(drop=True)
    df["_is_zero"]    = (df["avg_available"] < 0.5).astype(float)
    df["_near_zero"]  = (df["avg_available"] < 1.0).astype(float)
    df["zero_rate_3d"] = (
        df.groupby("station_uid")["_is_zero"]
        .transform(lambda s: s.shift(1).rolling(72, min_periods=9).mean())
    )
    df["near_zero_rate_3d"] = (
        df.groupby("station_uid")["_near_zero"]
        .transform(lambda s: s.shift(1).rolling(72, min_periods=9).mean())
    )
    df = df.drop(columns=["_is_zero", "_near_zero"])
    return df


def compute_station_stats(since: str, until: str) -> "dict[int, dict]":
    """compute per-station training statistics from the DB for [since, until).

    returns {station_uid: {station_mean_avail, avail_std, reb_per_week}}
    suitable for saving alongside a trained model.

    reb_per_week uses the same slow-rise rule (rise >= 8 bikes, off-peak) as an
    approximation of the full 4-rule heuristic — accurate enough for a stable
    station-level feature.
    """
    hourly = _query_hourly(since, until)
    hourly["hour_local"] = pd.to_datetime(hourly["hour_local"]).dt.tz_localize(None)

    agg = hourly.groupby("station_uid")["avg_available"].agg(
        station_mean_avail="mean",
        avail_std="std",
    ).reset_index()

    total_weeks = max((pd.Timestamp(until) - pd.Timestamp(since)).days / 7, 1)
    h_sorted = hourly.sort_values(["station_uid", "hour_local"])
    off_peak  = h_sorted["hour_local"].dt.hour.isin(_OFF_PEAK)
    rises     = h_sorted.groupby("station_uid")["avg_available"].diff()
    reb_approx = (
        h_sorted[(rises >= 8) & off_peak]
        .groupby("station_uid").size()
        .reset_index(name="reb_count")
    )
    reb_approx["reb_per_week"] = reb_approx["reb_count"] / total_weeks
    agg = agg.merge(reb_approx[["station_uid", "reb_per_week"]], on="station_uid", how="left")
    agg["reb_per_week"] = agg["reb_per_week"].fillna(0)

    return {
        int(row["station_uid"]): {
            "station_mean_avail": float(row["station_mean_avail"]),
            "avail_std":          float(row["avail_std"]) if not pd.isna(row["avail_std"]) else 0.0,
            "reb_per_week":       float(row["reb_per_week"]),
        }
        for _, row in agg.iterrows()
    }


def extract_station_stats_from_df(df: pd.DataFrame) -> "dict[int, dict]":
    """extract per-station stats from an already-built features DataFrame.

    suitable for calling in train.py after the train/val split, so that stats
    are computed from training rows only and saved alongside the model.
    """
    agg = df.groupby("station_uid").agg(
        station_mean_avail=("avg_available", "mean"),
    )
    if "avail_std" in df.columns:
        agg["avail_std"] = df.groupby("station_uid")["avail_std"].first()
    else:
        agg["avail_std"] = df.groupby("station_uid")["avg_available"].std()

    if "reb_per_week" in df.columns:
        agg["reb_per_week"] = df.groupby("station_uid")["reb_per_week"].first()
    else:
        agg["reb_per_week"] = 0.0

    return {
        int(uid): {
            "station_mean_avail": float(row["station_mean_avail"]),
            "avail_std":          float(row["avail_std"])    if not pd.isna(row["avail_std"])    else 0.0,
            "reb_per_week":       float(row["reb_per_week"]) if not pd.isna(row["reb_per_week"]) else 0.0,
        }
        for uid, row in agg.iterrows()
    }


def build_features(
    since: str,
    until: str,
    geo_parquet: str = "data/features.parquet",
    precomputed_stats: "dict | None" = None,
) -> pd.DataFrame:
    """build feature table for [since, until) from local postgres.

    station filter: is_spot=TRUE (enforced in SQL) and mean availability >= 0.5
    over the build window — same rule as notebook 02.

    precomputed_stats: optional dict {station_uid: {station_mean_avail, avail_std, reb_per_week}}
      as returned by extract_station_stats_from_df on the training slice.
      when provided, station-level stats come from the training window (no eval leakage).
      when None, stats are computed from the build window with a 6-week leakage guard.
    """
    print(f"[features] querying hourly data {since} to {until}")
    hourly = _query_hourly(since, until)
    print(f"[features] {len(hourly):,} station-hours loaded ({hourly['station_uid'].nunique():,} stations)")

    hourly["hour_local"] = pd.to_datetime(hourly["hour_local"]).dt.tz_localize(None)

    # explicit mean >= 0.5 filter — notebook rule, applied to the build window
    station_mean_all = hourly.groupby("station_uid")["avg_available"].mean()
    active_uids = station_mean_all[station_mean_all >= 0.5].index
    n_before = hourly["station_uid"].nunique()
    hourly = hourly[hourly["station_uid"].isin(active_uids)].copy()
    print(f"[features] mean>=0.5 filter: kept {hourly['station_uid'].nunique():,} stations (dropped {n_before - hourly['station_uid'].nunique():,})")

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
    for col in _WEATHER_COLS:
        if col not in hourly.columns:
            hourly[col] = np.nan

    # build station_mean_avail Series for lag_24h_ratio
    if precomputed_stats is not None:
        station_mean_avail = pd.Series(
            {uid: v["station_mean_avail"] for uid, v in precomputed_stats.items()}
        )
        print("[features] using precomputed station means from training window")
    else:
        station_mean_avail = None  # computed inside _add_lag_features with 6-week guard

    print("[features] adding lag features")
    hourly = _add_lag_features(hourly, station_mean_avail=station_mean_avail)

    print("[features] merging geo features")
    hourly = _add_geo_features(hourly, geo_parquet)

    print("[features] detecting rebalancing events")
    try:
        reb_events, distorted = _detect_rebalancing(hourly)
        hourly = _add_rebalancing_features(hourly, reb_events, distorted)
    except Exception as exc:
        print(f"[features] rebalancing detection failed ({exc}) — setting to NaN")
        for col in ["hours_since_reb", "avail_std", "reb_per_week", "zero_rate_3d", "near_zero_rate_3d"]:
            hourly[col] = np.nan

    # override avail_std and reb_per_week from training stats when provided
    if precomputed_stats is not None:
        avail_std_map    = {uid: v["avail_std"]    for uid, v in precomputed_stats.items()}
        reb_per_week_map = {uid: v["reb_per_week"] for uid, v in precomputed_stats.items()}
        hourly["avail_std"]    = hourly["station_uid"].map(avail_std_map)
        hourly["reb_per_week"] = hourly["station_uid"].map(reb_per_week_map).fillna(0)
        print("[features] overrode avail_std/reb_per_week from precomputed training stats")

    print(f"[features] done: {len(hourly):,} rows, {hourly.shape[1]} columns")
    return hourly
