"""shared fixtures for the test suite."""
import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def tiny_hourly():
    """three stations, 10 days of hourly data, no gaps."""
    stations  = [10, 20, 30]
    hours     = pd.date_range("2026-07-01", periods=240, freq="h")  # 10 days
    rows = []
    rng = np.random.default_rng(42)
    for s in stations:
        base = rng.uniform(1.5, 4.0)
        for h in hours:
            rows.append({
                "station_uid": s,
                "hour_local":  h,
                "avg_available": round(base + rng.normal(0, 0.5), 2),
            })
    return pd.DataFrame(rows)


@pytest.fixture
def tiny_hourly_with_gap(tiny_hourly):
    """same as tiny_hourly but station 10 is missing all of 2026-07-05."""
    mask = ~(
        (tiny_hourly["station_uid"] == 10) &
        (tiny_hourly["hour_local"].dt.date == pd.Timestamp("2026-07-05").date())
    )
    return tiny_hourly[mask].reset_index(drop=True)
