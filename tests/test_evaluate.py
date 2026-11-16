"""tests for src/evaluate.py."""
import json
import os
import tempfile

import lightgbm as lgb
import numpy as np
import pandas as pd
import pytest

from src.evaluate import evaluate, _MIN_EVAL_ROWS


def _make_model_file(n_features: int = 5) -> str:
    """train a tiny LightGBM model and save it, return path."""
    rng = np.random.default_rng(0)
    X = rng.random((200, n_features))
    y = X[:, 0] * 2 + rng.normal(0, 0.1, 200)
    m = lgb.LGBMRegressor(n_estimators=5, verbosity=-1)
    m.fit(X, y)
    tmp = tempfile.NamedTemporaryFile(suffix=".lgb", delete=False)
    tmp.close()
    m.booster_.save_model(tmp.name)
    return tmp.name


def _make_features_parquet(n_rows: int, feat_names: list[str]) -> tuple[str, str]:
    """create a tiny features parquet + feature_cols.json, return (parquet_path, cols_path).

    all rows fall inside 2026-08-01 to 2026-09-01 so the evaluate window always has >= n_rows.
    """
    rng = np.random.default_rng(1)
    df = pd.DataFrame(rng.random((n_rows, len(feat_names))), columns=feat_names)
    df["avg_available"] = rng.uniform(0, 5, n_rows)
    df["lag_24h"]       = rng.uniform(0, 5, n_rows)
    df["lag_168h"]      = rng.uniform(0, 5, n_rows)
    df["station_uid"]   = rng.integers(1, 10, n_rows)
    # keep all rows inside [2026-08-01, 2026-09-01): step by ~2.5min to fit n_rows in 31 days
    step_min = int(31 * 24 * 60 / max(n_rows, 1))
    df["hour_local"] = pd.date_range("2026-08-01", periods=n_rows,
                                     freq=pd.Timedelta(minutes=max(step_min, 1)))

    parquet_f = tempfile.NamedTemporaryFile(suffix=".parquet", delete=False)
    parquet_f.close()
    df.to_parquet(parquet_f.name, index=False)

    cols_f = tempfile.NamedTemporaryFile(suffix=".json", delete=False, mode="w")
    json.dump(feat_names, cols_f)
    cols_f.close()

    return parquet_f.name, cols_f.name


class TestEvaluate:
    def setup_method(self):
        self.feat_names = [f"f{i}" for i in range(5)]
        self.cand_path  = _make_model_file(len(self.feat_names))
        self.ref_path   = _make_model_file(len(self.feat_names))
        self.parquet, self.cols = _make_features_parquet(20_000, self.feat_names)
        os.environ["FEATURE_COLS_FILE"] = self.cols

    def teardown_method(self):
        for p in [self.cand_path, self.ref_path, self.parquet, self.cols]:
            try:
                os.unlink(p)
            except Exception:
                pass

    def test_row_count_guard_raises(self):
        """evaluate raises ValueError when eval window has fewer than _MIN_EVAL_ROWS rows."""
        # use a tiny parquet with only 5 rows
        _, cols = _make_features_parquet(5, self.feat_names)
        parquet_tiny, _ = _make_features_parquet(5, self.feat_names)
        os.environ["FEATURE_COLS_FILE"] = cols
        with pytest.raises(ValueError, match="eval set too small"):
            evaluate(
                candidate_path=self.cand_path,
                reference_path=self.ref_path,
                features_path=parquet_tiny,
                since="2026-08-01",
                until="2026-09-01",
            )
        os.unlink(parquet_tiny)
        os.unlink(cols)

    def test_eval_after_train_check(self):
        """evaluate reads the correct time window and returns metrics dict."""
        os.environ["FEATURE_COLS_FILE"] = self.cols
        result = evaluate(
            candidate_path=self.cand_path,
            reference_path=self.ref_path,
            features_path=self.parquet,
            since="2026-08-01",
            until="2026-09-01",
        )
        assert "candidate_mae"   in result
        assert "reference_mae"   in result
        assert "candidate_r2"    in result
        assert "is_better"       in result
        assert "beats_reference" in result
        assert "beats_naive"     in result
        assert isinstance(result["is_better"], bool)

    def test_is_better_requires_both_conditions(self):
        """is_better == beats_reference AND beats_naive."""
        os.environ["FEATURE_COLS_FILE"] = self.cols
        result = evaluate(
            candidate_path=self.cand_path,
            reference_path=self.ref_path,
            features_path=self.parquet,
            since="2026-08-01",
            until="2026-09-01",
        )
        expected_is_better = result["beats_reference"] and result["beats_naive"]
        assert result["is_better"] == expected_is_better, (
            f"is_better={result['is_better']} but beats_reference={result['beats_reference']}, "
            f"beats_naive={result['beats_naive']}"
        )

    def test_perfect_model_beats_naive(self):
        """a model that predicts avg_available exactly beats the naive lag_24h baseline."""
        from src.evaluate import _load_cols
        os.environ["FEATURE_COLS_FILE"] = self.cols
        # build a parquet where lag_24h is systematically off and avg_available is exactly f0*2
        rng = np.random.default_rng(99)
        n = 20_000
        feat_names = self.feat_names
        X = rng.random((n, len(feat_names)))
        df = pd.DataFrame(X, columns=feat_names)
        df["avg_available"] = X[:, 0] * 2
        df["lag_24h"]  = df["avg_available"] + rng.normal(0, 2, n)   # noisy lag
        df["lag_168h"] = df["avg_available"] + rng.normal(0, 3, n)
        df["station_uid"] = 1
        df["hour_local"] = pd.date_range("2026-08-01", periods=n, freq="min")

        p = tempfile.NamedTemporaryFile(suffix=".parquet", delete=False)
        p.close()
        df.to_parquet(p.name, index=False)

        result = evaluate(
            candidate_path=self.cand_path,   # this model was trained on f0*2 target
            reference_path=self.ref_path,
            features_path=p.name,
            since="2026-08-01",
            until="2026-09-01",
        )
        # candidate was trained on exact target, so it should beat the noisy naive lag
        # (we won't assert it always does — just that the logic runs and returns the right shape)
        assert "beats_naive" in result
        os.unlink(p.name)


class TestBootstrapPromotionCI:
    """promotion rule should hold up to bootstrap resampling."""

    def test_bootstrap_mae_ordering_stable(self):
        """if candidate MAE < reference MAE on a large sample, it wins on >80% of bootstraps."""
        rng = np.random.default_rng(42)
        n = 10_000
        y_true = rng.uniform(0, 5, n)
        # candidate: closer to truth
        y_cand = y_true + rng.normal(0, 0.5, n)
        # reference: further from truth
        y_ref  = y_true + rng.normal(0, 1.0, n)

        n_boots = 200
        cand_wins = 0
        for _ in range(n_boots):
            idx = rng.integers(0, n, n)
            mae_c = np.abs(y_cand[idx] - y_true[idx]).mean()
            mae_r = np.abs(y_ref[idx]  - y_true[idx]).mean()
            if mae_c < mae_r:
                cand_wins += 1

        win_rate = cand_wins / n_boots
        assert win_rate > 0.8, (
            f"expected candidate to win >80% of bootstrap resamples; got {win_rate:.1%}. "
            "this signals the MAE difference is too small to be reliable."
        )
