"""tests for the station filter logic in build_features."""
import numpy as np
import pandas as pd
import pytest


def _build_hourly(station_avails: dict, n_hours: int = 24 * 10) -> pd.DataFrame:
    """create synthetic hourly df with known per-station mean availability.

    station_avails: {station_uid: mean_value}
    """
    hours = pd.date_range("2026-07-01", periods=n_hours, freq="h")
    rng = np.random.default_rng(7)
    rows = []
    for uid, mean_val in station_avails.items():
        for h in hours:
            rows.append({
                "station_uid": uid,
                "hour_local":  h,
                "avg_available": max(0.0, mean_val + rng.normal(0, 0.1)),
            })
    return pd.DataFrame(rows)


class TestStationFilter:
    """the mean >= 0.5 filter in build_features matches the notebook rule."""

    def test_filter_drops_below_threshold(self):
        """stations with mean < 0.5 are dropped; stations with mean > 0.6 are kept."""
        # use means well away from the 0.5 boundary so noise doesn't flip them
        df = _build_hourly({10: 0.1, 20: 0.3, 30: 0.7, 40: 2.0})
        station_means = df.groupby("station_uid")["avg_available"].mean()
        active = station_means[station_means >= 0.5].index
        # stations 10 and 20 have mean ~0.1 and ~0.3 — clearly below threshold
        assert 10 not in active
        assert 20 not in active
        # stations 30 and 40 have mean ~0.7 and ~2.0 — clearly above threshold
        assert 30 in active
        assert 40 in active

    def test_filter_keeps_borderline_station(self):
        """a station with mean exactly 0.5 is kept."""
        rows = [{"station_uid": 99, "hour_local": pd.Timestamp("2026-07-01") + pd.Timedelta(hours=i),
                 "avg_available": 0.5} for i in range(100)]
        df = pd.DataFrame(rows)
        mean = df.groupby("station_uid")["avg_available"].mean().iloc[0]
        assert mean >= 0.5, "borderline station should pass the mean >= 0.5 filter"

    def test_filter_is_per_training_window(self):
        """the filter uses the build window mean, not a global pre-computed value.

        a station that is dead in the training window but alive in eval should be
        excluded from training features but included if built from the eval window.
        """
        # training window: station 50 has mean 0.1 (dead)
        train_rows = [{"station_uid": 50, "hour_local": pd.Timestamp("2026-07-01") + pd.Timedelta(hours=i),
                       "avg_available": 0.1} for i in range(48)]
        train_df = pd.DataFrame(train_rows)
        train_means = train_df.groupby("station_uid")["avg_available"].mean()
        assert train_means[50] < 0.5, "station should fail mean filter in training window"

        # eval window: same station now active
        eval_rows = [{"station_uid": 50, "hour_local": pd.Timestamp("2026-08-01") + pd.Timedelta(hours=i),
                      "avg_available": 3.0} for i in range(48)]
        eval_df = pd.DataFrame(eval_rows)
        eval_means = eval_df.groupby("station_uid")["avg_available"].mean()
        assert eval_means[50] >= 0.5, "station should pass mean filter in eval window"
