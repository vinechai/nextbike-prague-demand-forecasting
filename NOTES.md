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
