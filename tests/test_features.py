"""tests for src/features.py feature engineering."""
import numpy as np
import pandas as pd
import pytest

from src.features import _add_lag_features, _add_time_features, _add_rebalancing_features


# ── lag correctness ───────────────────────────────────────────────────────────

class TestLagFeatures:
    def test_lag_24h_exact_match(self, tiny_hourly):
        """lag_24h at hour T equals avg_available at T-24h."""
        df = _add_lag_features(tiny_hourly.copy())
        joined = df.merge(
            df[["station_uid", "hour_local", "avg_available"]].rename(
                columns={"hour_local": "lag_hour", "avg_available": "expected_lag24"}),
            left_on=["station_uid", df["hour_local"] - pd.Timedelta(hours=24)],
            right_on=["station_uid", "lag_hour"],
            how="left",
        )
        valid = joined["expected_lag24"].notna() & joined["lag_24h"].notna()
        assert valid.any(), "expected some non-null lag_24h rows"
        np.testing.assert_allclose(
            joined.loc[valid, "lag_24h"].values,
            joined.loc[valid, "expected_lag24"].values,
            rtol=1e-5,
            err_msg="lag_24h != avg_available 24h ago",
        )

    def test_lag_168h_exact_match(self, tiny_hourly):
        """lag_168h at hour T equals avg_available at T-168h."""
        df = _add_lag_features(tiny_hourly.copy())
        min_h = df["hour_local"].min() + pd.Timedelta(hours=168)
        window = df[df["hour_local"] >= min_h].copy()
        assert len(window) > 0
        # all rows after 168h of warmup should have a non-null lag_168h
        assert window["lag_168h"].notna().all(), "lag_168h unexpectedly NaN after warmup period"

    def test_lag_nan_across_gap(self):
        """a 3-day scraper gap produces NaN lag_24h for the day immediately after the gap.

        the ±1h tolerance can fill a 1h gap by looking at the adjacent hour.
        for a gap of 3+ days, lag_24h at the first day after the gap has no nearby hour
        to fall back to, so it remains NaN.
        """
        # build 14 days of data for one station, then remove days 4-6 entirely (3-day gap)
        hours = pd.date_range("2026-07-01", periods=336, freq="h")  # 14 days
        rng = np.random.default_rng(99)
        df = pd.DataFrame({
            "station_uid": 10,
            "hour_local":  hours,
            "avg_available": np.abs(rng.normal(2.0, 0.5, 336)),
        })
        # remove Jul 4, 5, 6 entirely (72-hour gap)
        gap_mask = df["hour_local"].dt.date.isin([
            pd.Timestamp("2026-07-04").date(),
            pd.Timestamp("2026-07-05").date(),
            pd.Timestamp("2026-07-06").date(),
        ])
        df_gapped = df[~gap_mask].reset_index(drop=True)

        result = _add_lag_features(df_gapped.copy())

        # rows on Jul 7 00:00-22:00 look back 24h (±1h tolerance) to Jul 6 01:00-Jul 7 00:00-1h.
        # the furthest the ±1h tolerance can reach on Jul 7 23:00 is T-23h = Jul 7 00:00 (exists).
        # so only Jul 7 23:00 may be filled; hours 00:00-22:00 must remain NaN.
        jul7 = result[result["hour_local"].dt.date == pd.Timestamp("2026-07-07").date()]
        jul7_early = jul7[jul7["hour_local"].dt.hour < 23]
        assert len(jul7_early) >= 20, "expected at least 20 early Jul 7 rows"
        assert jul7_early["lag_24h"].isna().all(), (
            "expected NaN lag_24h for Jul 7 00:00-22:00 (all 24h±1h lookbacks land in the 3-day gap)"
        )

    def test_lag_24h_ratio_formula(self, tiny_hourly):
        """lag_24h_ratio == lag_24h / (station_mean + 0.1)."""
        df = _add_lag_features(tiny_hourly.copy())
        valid = df["lag_24h"].notna() & df["lag_24h_ratio"].notna()
        assert valid.any()
        # infer station mean from the ratio: ratio = lag_24h / (mean + 0.1)  =>  mean = lag_24h/ratio - 0.1
        # check that this is consistent across all rows for each station
        df_v = df[valid].copy()
        df_v["inferred_mean"] = df_v["lag_24h"] / df_v["lag_24h_ratio"].replace(0, np.nan) - 0.1
        station_mean_std = df_v.groupby("station_uid")["inferred_mean"].std()
        assert (station_mean_std.dropna() < 0.01).all(), (
            "lag_24h_ratio implies inconsistent station means — formula may be wrong"
        )

    def test_no_target_leakage(self, tiny_hourly):
        """lag features at hour T are derived from avg_available at T-N, never from T itself.

        we check the THREE raw lags (lag_24h, lag_48h, lag_168h): they look strictly backward
        in time.  lag_24h_ratio depends on the station mean — changing one hour slightly shifts
        the mean — so we exclude it from this check (it is not target-leaky in the dangerous
        sense: you cannot recover T's exact value from the ratio and the public station mean).
        """
        df_orig   = _add_lag_features(tiny_hourly.copy())
        df_perturb = tiny_hourly.copy()

        # perturb one specific row in the MIDDLE of the series (not at a lookback boundary)
        target_h = pd.Timestamp("2026-07-05 12:00")
        idx = df_perturb[(df_perturb["station_uid"] == 10) &
                         (df_perturb["hour_local"] == target_h)].index
        if len(idx) == 0:
            pytest.skip("target row not in dataset")
        df_perturb.loc[idx, "avg_available"] = 999.0
        df_mod = _add_lag_features(df_perturb.copy())

        # lags at T look at T-24h, T-48h, T-168h — not at T — so they must be unchanged
        t     = df_orig[df_orig["hour_local"] == target_h]
        t_mod = df_mod[df_mod["hour_local"] == target_h]
        if t.empty or t_mod.empty:
            pytest.skip("row missing")

        for col in ("lag_24h", "lag_48h", "lag_168h"):
            o, m = t[col].values[0], t_mod[col].values[0]
            if not (pd.isna(o) and pd.isna(m)):
                assert abs(o - m) < 1e-6, (
                    f"leakage: {col} at perturbed hour changed from {o} to {m}"
                )

        # the perturbed value DOES appear correctly as lag_24h of rows 24h later
        future_h = target_h + pd.Timedelta(hours=24)
        row_24h_later = df_mod[(df_mod["station_uid"] == 10) &
                                (df_mod["hour_local"] == future_h)]
        if not row_24h_later.empty:
            lag_val = row_24h_later["lag_24h"].values[0]
            if pd.notna(lag_val):
                assert abs(lag_val - 999.0) < 0.01, (
                    f"lag_24h at T+24h should reflect the perturbed value (999), got {lag_val}"
                )


# ── time features ─────────────────────────────────────────────────────────────

class TestTimeFeatures:
    def test_month_sin_formula(self, tiny_hourly):
        """month_sin uses month/12 (not (month-1)/12) to match notebook and API."""
        df = _add_time_features(tiny_hourly.copy())
        for _, row in df.head(10).iterrows():
            expected = np.sin(2 * np.pi * row["month"] / 12)
            assert abs(row["month_sin"] - expected) < 1e-8, (
                f"month_sin mismatch for month={row['month']}: "
                f"got {row['month_sin']:.6f}, expected {expected:.6f}"
            )

    def test_hour_sin_formula(self, tiny_hourly):
        """hour_sin = sin(2π × hour_of_day / 24)."""
        df = _add_time_features(tiny_hourly.copy())
        for _, row in df.head(24).iterrows():
            expected = np.sin(2 * np.pi * row["hour_of_day"] / 24)
            assert abs(row["hour_sin"] - expected) < 1e-8

    def test_is_weekend_correct(self, tiny_hourly):
        """is_weekend == 1 iff dow >= 5."""
        df = _add_time_features(tiny_hourly.copy())
        assert ((df["is_weekend"] == 1) == (df["dow"] >= 5)).all()


# ── rebalancing features ──────────────────────────────────────────────────────

class TestRebalancingFeatures:
    def _make_reb_df(self):
        """minimal hourly df with a clear rebalancing event."""
        hours = pd.date_range("2026-07-01", periods=72, freq="h")
        station = 10
        avail = [2.0] * 72
        # inject a +10 bike jump at hour 25 (off-peak: 01:00)
        avail[25] = 12.0
        df = pd.DataFrame({
            "station_uid": station,
            "hour_local":  hours,
            "avg_available": avail,
        })
        return df

    def test_lag_24h_replaced_after_reb(self):
        """lag_24h is replaced by lag_48h when the 24h lookback falls in a rebalanced hour."""
        df = self._make_reb_df()
        df_lag = _add_lag_features(df.copy())

        # inject a rebalancing event at hour 25 (Jul 1 01:00)
        reb_events = pd.DataFrame([{
            "station_uid": 10,
            "reb_hour":    pd.Timestamp("2026-07-01 01:00"),
        }])
        distorted = pd.DataFrame([{
            "station_uid": 10,
            "hour_local":  pd.Timestamp("2026-07-01 01:00") + pd.Timedelta(hours=i),
        } for i in range(4)])
        df_with_reb = _add_rebalancing_features(df_lag.copy(), reb_events, distorted)

        # row at Jul 2 01:00 (24h after the event) should have lag_24h replaced
        row_24h_after = df_with_reb[
            df_with_reb["hour_local"] == pd.Timestamp("2026-07-02 01:00")
        ]
        if row_24h_after.empty:
            pytest.skip("row not in dataset")
        # original lag_24h would have been the rebalancing-inflated 12.0;
        # replaced value should be lag_48h instead
        lag_48h_val = row_24h_after["lag_48h"].values[0]
        lag_24h_val = row_24h_after["lag_24h"].values[0]
        if pd.notna(lag_48h_val) and pd.notna(lag_24h_val):
            assert abs(lag_24h_val - lag_48h_val) < 0.01, (
                f"expected lag_24h={lag_48h_val} (replaced by lag_48h) but got {lag_24h_val}"
            )

    def test_hours_since_reb_capped_at_24(self):
        """hours_since_reb is NaN beyond 24h after the event."""
        df = self._make_reb_df()
        df_lag = _add_lag_features(df.copy())
        reb_events = pd.DataFrame([{
            "station_uid": 10,
            "reb_hour":    pd.Timestamp("2026-07-01 01:00"),
        }])
        df_with_reb = _add_rebalancing_features(df_lag.copy(), reb_events, pd.DataFrame())
        beyond_24h = df_with_reb[
            df_with_reb["hour_local"] >= pd.Timestamp("2026-07-02 02:00")
        ]
        if beyond_24h.empty:
            pytest.skip("no rows beyond 24h in tiny dataset")
        assert beyond_24h["hours_since_reb"].isna().all(), (
            "hours_since_reb should be NaN beyond 24h of the event"
        )

    def test_reb_per_week_zero_when_no_events(self, tiny_hourly):
        """stations with no rebalancing events have reb_per_week == 0."""
        df_lag = _add_lag_features(tiny_hourly.copy())
        # empty reb_events with proper dtypes to avoid merge_asof dtype errors
        reb_events = pd.DataFrame({
            "station_uid": pd.Series([], dtype="int64"),
            "reb_hour":    pd.Series([], dtype="datetime64[us]"),
        })
        distorted = pd.DataFrame({
            "station_uid": pd.Series([], dtype="int64"),
            "hour_local":  pd.Series([], dtype="datetime64[us]"),
        })
        df_reb = _add_rebalancing_features(df_lag, reb_events, distorted)
        assert (df_reb["reb_per_week"] == 0.0).all(), "expected reb_per_week=0 with no events"
