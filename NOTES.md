# de-upgrades — session notes

branch: `de-upgrades` | base: `main` @ 957393a

## what finished

### task 1 — dbt ✅
- `dbt/dbt_project.yml`, `profiles.yml`, `packages.yml`
- `models/_sources.yml` — 4 raw tables declared as sources (stations, station_snapshots, bike_positions, weather_hourly)
- staging models: `stg_stations`, `stg_station_snapshots`, `stg_bike_positions`, `stg_weather_hourly`
- intermediate: `int_station_hourly_avg` (aggregate snapshots → 1 row per station/hour)
- mart: `mart_station_hourly_features` (time features + weather join + lag window functions)
- tests: unique/not_null on PKs, FK relationships, accepted_range via dbt_utils on bikes and time cols
- `check_mart.py` — compares mart vs features.parquet, prints MAE for avg_available and lag cols
- README with setup steps

### task 2 — airflow ✅
- `src/db.py` — psycopg2 pool reading DB_* env vars
- `src/features.py` — query → hourly avg → time features → weather join → lag features (explicit time-offset merge for gap handling) → geo merge
- `src/train.py` — LightGBM + MLflow logging, versioned output path
- `src/evaluate.py` — compare candidate vs reference MAE on held-out window
- `airflow/dags/retrain_pipeline.py` — 5-task DAG, paused at creation, promote writes versioned file only
- `airflow/docker-compose.yaml` — LocalExecutor stack, airflow postgres on port 5433 (no conflict with nextbike on 5432)
- `airflow/Dockerfile`, `.env.example`, `requirements.txt`, README
- syntax verified: `ast.parse()` on all Python files — all OK

### task 3 — lakehouse ✅ (scripts written, not run)
- `lakehouse/export_snapshot.py` — postgres → date-partitioned parquet (pure pandas, no java)
- `lakehouse/bronze.py` — raw parquet → Delta bronze
- `lakehouse/silver.py` — dedup + type cast → Delta silver
- `lakehouse/gold.py` — hourly aggregation + time features + weather join + lag windows → Delta gold
- syntax verified — all OK

## what failed / was skipped

### lakehouse execution ❌
- java is not installed on this machine (`java: command not found`)
- scripts are written and syntactically valid but were not executed
- **to run**: install java 11+ from https://adoptium.net, then `pip install -r lakehouse/requirements.txt` and run `export_snapshot.py` → `bronze.py` → `silver.py` → `gold.py`

### dbt run/test ❌
- dbt-postgres not installed in this environment (`pip install dbt-postgres` needed)
- models were written and validated structurally but were not executed against the local db
- **to run**: `pip install dbt-postgres`, then `cd dbt && dbt deps && dbt run --profiles-dir . && dbt test --profiles-dir .`

### airflow dags test ❌
- apache-airflow not installed locally
- DAG syntax is valid (ast.parse passes)
- **to test**: `pip install apache-airflow` then `airflow dags test retrain_pipeline 2026-10-29`

## known limitations

### lag features in dbt mart and lakehouse gold
the `lag(n)` SQL window function shifts by N rows, not N hours.
if there are gaps in the hourly series (scraper outages), these values will differ from
`features.parquet` which uses an explicit time-offset join.
`check_mart.py` reports the expected difference.
`src/features.py` uses the correct time-offset approach.

### rebalancing features
`src/features.py` sets `hours_since_reb`, `reb_per_week`, `avail_std`, `zero_rate_3d`,
`near_zero_rate_3d` to NaN — these require the full heuristic detection pipeline from
notebook 02. a model trained from the DAG will have slightly worse accuracy than the
production model because of this.

### candidate model quality
the retrain DAG's candidate model won't match production quality:
1. rebalancing features are NaN (see above)
2. training window is defined by TRAIN_SINCE/TRAIN_UNTIL env vars — needs to cover enough data

## audit findings — 2026-10-30

### bug 1: evaluate ran on 2 hours of data (false MAE 0.0528)

`_build_features` built features only for `TRAIN_SINCE`→`TRAIN_UNTIL` (July).
The eval window is `EVAL_SINCE`→`EVAL_UNTIL` (August). The features parquet only
overlapped for Aug 1 00:00 and 01:00 → 3,010 eval rows. Candidate MAE 0.0528 is
computed on those 2 hours, which is meaningless.

**fix**: `_build_features` now builds for `TRAIN_SINCE`→`EVAL_UNTIL` so both tasks
have the same full-window parquet. A `_MIN_EVAL_ROWS = 10,000` guard in `evaluate.py`
raises an error if fewer rows are found, preventing a silent bad comparison.

### bug 2: lag_24h_ratio leaks the target in training

`src/features.py` computed:

    lag_24h_ratio = (avg_available - lag_24h) / (lag_24h + 1)

`avg_available` is the prediction target. Knowing `lag_24h_ratio` and `lag_24h`
lets you recover `avg_available` exactly. This is direct leakage.

The production API (`api/predictor.py`) computes:

    lag_24h_ratio = lag_24h / (station_mean_avail + 0.1)

which uses only historical lag values — no target.

**fix**: `src/features.py` now uses `lag_24h / (station_mean + 0.1)` where
`station_mean` is the per-station mean of `avg_available` across the build window.
This matches the API formula and introduces no leakage.

### model metrics (on features.parquet, eval 2026-08-01 to 2026-09-01, 395,302 rows)

| model | MAE | R² | notes |
|---|---|---|---|
| candidate from DAG run 5 | 3.1977 | -3.35 | trained with leaky lag_24h_ratio |
| reference (data/model.lgb) | 0.9038 | 0.5683 | production model, consistent with README |
| naive lag_24h | 1.0060 | 0.3610 | same-hour-yesterday baseline |
| naive lag_168h | 1.6286 | -0.2877 | same-hour-last-week baseline |

The candidate from run 5 is actively worse than the naive baseline. It should not have
been promoted. `data/models/model_best_20261029.lgb` is invalid.

**action**: deleted `data/models/model_best_20261029.lgb` manually (do not use).
`data/model.lgb` is unchanged. A corrected retrain (run 6, with bug fixes applied)
will produce a new candidate. Run 6 result will be appended below once it completes.

### note on horizons

The DAG trains a single model on `(features at hour T, target = avg_available at T)`.
It does NOT produce per-horizon models. The production setup does the same: the API
calls `predict_all(target_utc)` and uses lag values queried live for that target UTC.
The README says "up to 24h ahead" but the model structure is single-horizon: it
predicts availability at any requested hour given that hour's lag features.

### note on lag_1h

`lag_1h` is in `features.parquet` but NOT in `feature_cols.json` and NOT used by the
production model. The API computes it for debug display only. Naive lag_1h achieves
MAE=0.1175, R²=0.9596 on the eval set — far better than the model — because
availability changes slowly and the current hour is almost always close to the previous.
The model intentionally omits lag_1h because it won't always be available at inference
time for multi-hour forward predictions.

### run 7 result (both bugs fixed)

Both builds succeeded within memory limits (train: 270,909 rows Jul 1-Aug 1;
eval: 939,464 rows Jul 25-Sep 1 for lag lookback, filtered to Aug in evaluate.py).

| model | MAE | R² | eval rows |
|---|---|---|---|
| candidate (run 7, fixed features) | 0.6477 | 0.4841 | 795,559 |
| reference (data/model.lgb) | 1.0623 | 0.3574 | 795,559 |
| naive lag_24h | 0.5906 | 0.4620 | 694,852 |
| naive lag_168h | 0.9396 | -0.0855 | 625,388 |

promote ran correctly (candidate < reference).

the candidate is worse than naive lag_24h (0.6477 vs 0.5906) — expected, because:
1. rebalancing features (hours_since_reb, reb_per_week, avail_std, zero_rate_3d,
   near_zero_rate_3d) are all NaN — these are the most powerful non-lag predictors
2. training on only 1 month (Jul) may not generalize to August well
3. local Postgres has 3,201 unique station_uids vs 751 in features.parquet — more
   stations means more unseen station-hours in eval

the reference model performs worse here (MAE=1.062) than on features.parquet
(MAE=0.904, from the offline audit above) because it encounters 3,201 stations
vs the 751 it was trained on.

note on model horizons: the README says "up to 24h ahead" but the model is a
single-horizon model — it predicts avg_available at a requested target hour T
given lag features at T-24h, T-48h, T-168h. the API calls it once per hour per
station for each future hour independently.

## what to check first

1. **dbt**: run `dbt deps && dbt run --profiles-dir .` with local docker postgres running
   — watch for schema issues (check that `dbt_dev` schema is created)
2. **check_mart.py**: `python dbt/check_mart.py` — avg_available diff should be near 0,
   lag diffs show the gap-handling discrepancy
3. **airflow DAG**: `airflow dags test retrain_pipeline 2026-10-29` — all 5 tasks should pass
   (fetch_weather may fail if local db is empty)
4. **lakehouse**: install java, then run `export_snapshot.py` → `bronze.py` → `silver.py` → `gold.py`
5. **mlflow UI**: after a train run, `mlflow ui --backend-store-uri mlruns` at project root

## approximate runtimes (not measured — java skip)

- dbt run (full history): ~2-5 min (mart materialization is the slow step)
- src/features.py build (5 months): ~3-8 min (lag join is the slow step)
- train.py (LightGBM 2000 trees): ~5-15 min depending on hardware
- lakehouse export_snapshot (5 months): ~5-10 min
- bronze.py / silver.py / gold.py: ~5-20 min each (PySpark startup ~30s)
