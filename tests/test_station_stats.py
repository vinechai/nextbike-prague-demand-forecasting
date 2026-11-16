"""tests for extract_station_stats_from_df and precomputed_stats — no DB required."""

import numpy as np
import pandas as pd
import pytest

from src.features import _add_lag_features, extract_station_stats_from_df


def _make_df(uid: int, avail_values: list, reb_per_week: float = 0.5, avail_std: float = 1.0) -> pd.DataFrame:
    """build a minimal features DataFrame for one station."""
    n = len(avail_values)
    return pd.DataFrame({
        "station_uid": uid,
        "hour_local":  pd.date_range("2026-01-01", periods=n, freq="h"),
        "avg_available": avail_values,
        "reb_per_week":  reb_per_week,
        "avail_std":     avail_std,
    })


class TestExtractStationStats:
    def test_mean_avail_correct(self):
        df = _make_df(uid=1, avail_values=[2.0, 4.0, 6.0])
        stats = extract_station_stats_from_df(df)
        assert stats[1]["station_mean_avail"] == pytest.approx(4.0)

    def test_avail_std_from_column(self):
        df = _make_df(uid=1, avail_values=[1.0, 2.0, 3.0], avail_std=0.75)
        stats = extract_station_stats_from_df(df)
        # should take the pre-computed avail_std column, not recompute
        assert stats[1]["avail_std"] == pytest.approx(0.75)

    def test_avail_std_fallback_when_missing(self):
        df = _make_df(uid=1, avail_values=[1.0, 2.0, 3.0])
        df = df.drop(columns=["avail_std"])
        stats = extract_station_stats_from_df(df)
        expected = float(pd.Series([1.0, 2.0, 3.0]).std())
        assert stats[1]["avail_std"] == pytest.approx(expected, rel=1e-5)

    def test_reb_per_week_from_column(self):
        df = _make_df(uid=1, avail_values=[1.0, 2.0], reb_per_week=1.23)
        stats = extract_station_stats_from_df(df)
        assert stats[1]["reb_per_week"] == pytest.approx(1.23)

    def test_multiple_stations(self):
        df = pd.concat([
            _make_df(uid=10, avail_values=[1.0, 1.0, 1.0], avail_std=0.1),
            _make_df(uid=20, avail_values=[5.0, 5.0, 5.0], avail_std=0.9),
        ], ignore_index=True)
        stats = extract_station_stats_from_df(df)
        assert set(stats.keys()) == {10, 20}
        assert stats[10]["station_mean_avail"] == pytest.approx(1.0)
        assert stats[20]["station_mean_avail"] == pytest.approx(5.0)
        assert stats[10]["avail_std"] == pytest.approx(0.1)
        assert stats[20]["avail_std"] == pytest.approx(0.9)

    def test_different_windows_give_different_stats(self):
        """stats computed from two different data windows must differ for the same station."""
        # window A: low availability
        df_a = _make_df(uid=1, avail_values=[1.0, 1.5, 0.8, 1.2, 1.1], reb_per_week=0.1, avail_std=0.2)
        # window B: high availability
        df_b = _make_df(uid=1, avail_values=[8.0, 9.0, 7.5, 8.5, 9.5], reb_per_week=1.5, avail_std=2.1)

        stats_a = extract_station_stats_from_df(df_a)
        stats_b = extract_station_stats_from_df(df_b)

        assert stats_a[1]["station_mean_avail"] != stats_b[1]["station_mean_avail"]
        assert stats_a[1]["avail_std"] != stats_b[1]["avail_std"]
        assert stats_a[1]["reb_per_week"] != stats_b[1]["reb_per_week"]

    def test_nan_avail_std_becomes_zero(self):
        """a single-row station has NaN std — should not raise, should return 0.0."""
        df = _make_df(uid=1, avail_values=[3.0])
        df = df.drop(columns=["avail_std"])
        stats = extract_station_stats_from_df(df)
        # std of a single value is NaN; function should coerce to 0.0
        assert stats[1]["avail_std"] == pytest.approx(0.0)

    def test_all_values_are_floats(self):
        df = _make_df(uid=5, avail_values=[2.0, 3.0, 4.0])
        stats = extract_station_stats_from_df(df)
        for key, val in stats[5].items():
            assert isinstance(val, float), f"{key} is not float: {type(val)}"


class TestPrecomputedStatsApplied:
    """verify that precomputed training stats are used for eval-window features.

    tests _add_lag_features directly (no DB required).
    this guards against the eval-window leakage bug: without precomputed_stats,
    lag_24h_ratio uses the eval window's station mean instead of the training mean.
    """

    def _make_windowed_df(self, uid: int, avail_values: list) -> pd.DataFrame:
        """build a DataFrame spanning enough hours for lag_24h lookbacks to resolve.

        rows at the START have no lookbacks (NaN lag).
        rows in EVAL_SLICE (index 24+) have resolved lag_24h.
        """
        n = len(avail_values)
        return pd.DataFrame({
            "station_uid": uid,
            "hour_local":  pd.date_range("2026-08-01", periods=n, freq="h"),
            "avg_available": avail_values,
        })

    def test_training_mean_used_for_lag_ratio(self):
        """lag_24h_ratio must use training station_mean_avail, not eval window mean."""
        # training mean is 1.0 (low-availability)
        training_mean = pd.Series({1: 1.0})

        # 25 rows: first 24 are lookback history (avail=2.0), row 24 is the eval row
        avail = [2.0] * 25
        # eval row (index 24) has much higher availability
        avail[24] = 9.0
        df = self._make_windowed_df(uid=1, avail_values=avail)

        result_train_stats = _add_lag_features(df.copy(), station_mean_avail=training_mean)
        result_no_stats    = _add_lag_features(df.copy(), station_mean_avail=None)

        # eval row: lag_24h = avail[0] = 2.0 (the value 24h before)
        lag_val = result_train_stats["lag_24h"].iloc[24]
        assert not np.isnan(lag_val), "lag_24h should resolve at row 24"

        # with training stats: ratio = lag_24h / (1.0 + 0.1)
        expected_train = lag_val / (1.0 + 0.1)
        assert result_train_stats["lag_24h_ratio"].iloc[24] == pytest.approx(expected_train, rel=1e-5)

        # without stats: 6-week guard computes mean from the df itself (mean ≈ 2.0)
        # the ratio will be different from the training-stats version
        ratio_no_stats = result_no_stats["lag_24h_ratio"].iloc[24]
        assert not np.isnan(ratio_no_stats)
        assert result_train_stats["lag_24h_ratio"].iloc[24] != pytest.approx(ratio_no_stats, rel=0.01)

    def test_eval_rows_match_saved_training_stats(self):
        """stats extracted from train_df and applied to eval_df must match the saved values."""
        # training data: 5 rows, low availability
        train_df = _make_df(uid=1, avail_values=[1.0, 1.5, 0.8, 1.2, 1.0], avail_std=0.25, reb_per_week=0.5)
        train_stats = extract_station_stats_from_df(train_df)
        training_mean_avail = train_stats[1]["station_mean_avail"]

        # eval window: 25 rows at higher availability, so lookbacks can resolve
        eval_df = self._make_windowed_df(uid=1, avail_values=[2.0] * 25)
        training_mean = pd.Series({1: training_mean_avail})
        result = _add_lag_features(eval_df.copy(), station_mean_avail=training_mean)

        # lag_24h at row 24 = avg_available 24h back = 2.0
        lag_val = result["lag_24h"].iloc[24]
        assert not np.isnan(lag_val)

        expected_ratio = lag_val / (training_mean_avail + 0.1)
        assert result["lag_24h_ratio"].iloc[24] == pytest.approx(expected_ratio, rel=1e-5)
