# how it works

A top-to-bottom explanation of the pipeline — what each tool does, why it's there,
how the pieces connect, and what to say about it in an interview.

---

## data sources

Two tables live in a local Docker Postgres (`nextbike` database, port 5432):

- **`station_snapshots`** — raw scraper output: one row per station per scrape
  (~every 5 minutes), columns `station_uid`, `scrape_time`, `bikes_available_to_rent`
- **`weather_hourly`** — hourly Open-Meteo weather data for Prague: temperature,
  precipitation, wind, radiation, etc.

Everything downstream reads from these two tables. Nothing ever writes back to them
(the scraper is a separate process).

---

## dbt — SQL transformations

### what dbt is

dbt (data build tool) is a framework that turns `.sql` files into a pipeline.
You write `SELECT` statements, dbt figures out the execution order from `{{ ref() }}`
calls between models, and runs them against the database. Think of it as `make` for SQL.

### the four layers

```
sources (raw tables)
  └─ staging      (clean types, rename columns)
       └─ intermediate   (aggregate: snapshots → one row per station/hour)
            └─ mart      (final feature table: time + weather + lag features)
```

**sources** (`dbt/models/_sources.yml`) — declarations that tell dbt "this table
already exists in Postgres; don't try to create it." No SQL, just YAML.

**staging** (`stg_station_snapshots`, `stg_stations`, `stg_weather_hourly`,
`stg_bike_positions`) — thin views that cast columns to the right types and rename
anything ugly. Materialized as VIEWs so they don't take up space.

**intermediate** (`int_station_hourly_avg`) — aggregates the ~12 scraper snapshots per
hour into one row: `avg_available`, `min_available`, `max_available`, `n_snapshots`.
Materialized as a TABLE with an index on `(station_uid, hour_local)` because the mart
joins against it three times (for the 24h, 48h, 168h lags).

**mart** (`mart_station_hourly_features`) — the final ML feature table. Adds:
- time features: `hour_of_day`, `dow`, `month`, `is_weekend`, `is_holiday`,
  cyclical sin/cos encodings
- weather: joined from `stg_weather_hourly` on `hour_local`
- lag features: `lag_24h`, `lag_48h`, `lag_168h` — the value of `avg_available`
  24/48/168 hours before each row

### lag features: why timestamp join instead of `lag(n)`

The naive SQL approach is `lag(avg_available, 24) OVER (PARTITION BY station_uid
ORDER BY hour_local)`. This looks backwards 24 *rows*, not 24 *hours*. If the scraper
was down for 3 hours, that function returns the value from 21 hours ago.

The mart uses a LEFT JOIN instead:

```sql
left join hourly lag24
    on  lag24.station_uid = h.station_uid
    and lag24.hour_local  = h.hour_local - interval '24 hours'
```

If the 24-hours-ago row doesn't exist, the join returns NULL — which is honest.
The model can handle NULLs; wrong lag values would silently corrupt training data.

### dbt tests

`dbt test` runs 45 data quality checks defined in YAML:
- `unique` and `not_null` on primary keys
- `accepted_range` (from the `dbt_utils` package) on `bikes_available_to_rent`
  (0–100) and time columns

### useful commands

```bash
cd dbt
.venv-dbt/Scripts/dbt run --profiles-dir .      # materialize all models
.venv-dbt/Scripts/dbt test --profiles-dir .     # run 45 tests
.venv-dbt/Scripts/dbt docs generate --profiles-dir .
.venv-dbt/Scripts/dbt docs serve --port 8888    # lineage graph at localhost:8888
```

---

## Airflow — scheduling and orchestration

### what Airflow is

Airflow is a workflow scheduler. You write a DAG (Directed Acyclic Graph) in Python —
a set of tasks with dependencies — and Airflow runs it on a schedule, retries failures,
and keeps a UI showing what happened.

### the DAG

`airflow/dags/retrain_pipeline.py` defines five tasks:

```
fetch_weather → build_features → train_candidate → evaluate → branch_promote
                                                                 ↓         ↓
                                                             promote   skip_promote
```

- **fetch_weather** — checks that `weather_hourly` has data for the training window.
  In production this would call Open-Meteo; here it just validates.
- **build_features** — calls `src/features.py` to query Postgres and produce
  `data/features_candidate.parquet`
- **train_candidate** — calls `src/train.py`, trains LightGBM 2000 trees on the
  parquet, logs params and metrics to MLflow, saves the model file
- **evaluate** — calls `src/evaluate.py`, compares candidate vs reference MAE on a
  held-out window
- **branch_promote** — a `BranchPythonOperator`: reads the `is_better` XCom from
  evaluate, routes to `promote` or `skip_promote`
- **promote** — copies the candidate to `data/models/model_best_<date>.lgb`.
  Deliberately does NOT overwrite `data/model.lgb` — that's a manual step after
  human review.

The DAG runs on `0 3 * * 1` (Mondays at 03:00 UTC). It was manually triggered for
the verified run.

### how Airflow runs

This stack uses **LocalExecutor**: the scheduler runs each task as a subprocess on the
same machine. One `docker compose up` starts two containers:
- `airflow-webserver` — the UI at localhost:8080
- `airflow-scheduler` — parses DAGs and spawns task processes

A third container `af-postgres` stores Airflow's own metadata (DAG run history,
task states, XComs). This is separate from the nextbike Postgres.

The `src/` and `data/` directories are mounted as Docker volumes, so edits on the
host take effect immediately without rebuilding.

### how to see task logs

Airflow UI → DAGs → `retrain_pipeline` → Grid view → click a task cell → Logs tab.
Or directly in `airflow/logs/dag_id=retrain_pipeline/run_id=.../task_id=.../attempt=1.log`.

---

## MLflow — experiment tracking

MLflow is the model registry. `src/train.py` calls:

```python
mlflow.set_tracking_uri("mlruns")   # local filesystem backend
mlflow.set_experiment("nextbike_retrain")
with mlflow.start_run(run_name="retrain_20261029"):
    mlflow.log_params({...})
    mlflow.log_metrics({"train_mae": ..., "val_mae": ...})
    mlflow.log_artifact(output_path)
```

Each training run creates a directory under `mlruns/<experiment_id>/<run_id>/` with
the params, metrics, and a copy of the model file. You can browse all runs:

```bash
mlflow ui --backend-store-uri mlruns/
# open http://localhost:5000
```

The `MLFLOW_ALLOW_FILE_STORE=true` env var is needed because MLflow ≥ 2.20 put the
local filesystem backend in maintenance mode; without it, the container raises an
exception.

---

## Delta Lake — lakehouse (scripts written, Java not yet installed)

The `lakehouse/` scripts implement a bronze/silver/gold pipeline using PySpark and
Delta Lake:

- **`export_snapshot.py`** — reads Postgres and writes date-partitioned Parquet to
  `data/lakehouse/raw/` (no Java needed, pure pandas)
- **`bronze.py`** — converts raw Parquet to Delta format, keeping every original
  column unchanged (raw, append-only)
- **`silver.py`** — deduplicates, casts types, filters invalid rows; stored as Delta
  with `MERGE INTO` upsert semantics
- **`gold.py`** — hourly aggregation + time features + weather join + lag windows;
  the final analytical layer, stored as Delta

Delta Lake is a storage format on top of Parquet that adds ACID transactions and
time-travel (you can query any past version). PySpark reads/writes it via the
`delta-spark` package, which requires Java.

**To run**: install JDK 17 from https://adoptium.net, then:
```bash
pip install -r lakehouse/requirements.txt
python lakehouse/export_snapshot.py
python lakehouse/bronze.py
python lakehouse/silver.py
python lakehouse/gold.py
```

---

## duplication between `src/features.py` and the dbt mart

Both produce a features table with the same columns from the same source tables.
This is intentional, not an accident:

| | dbt mart | `src/features.py` |
|---|---|---|
| **runs in** | Postgres (SQL) | Python (pandas) |
| **used by** | exploration, `check_mart.py`, BI tools | Airflow DAG, offline training |
| **lag implementation** | timestamp LEFT JOIN | timestamp merge (same logic) |
| **rebalancing features** | not included | set to NaN (full heuristic not ported) |

The dbt mart is the *documented, queryable, testable* version of the feature logic.
The `src/features.py` is the *runtime* version used inside the pipeline. They were
written to produce the same output so that `check_mart.py` can validate them against
each other (`avg_available` MAE < 0.01; lag MAE < 0.05 from gap-boundary differences).

In a real team you'd pick one path. Here having both serves as a cross-check: if
`src/features.py` produces different numbers than the mart, something changed and
you want to know.

---

## how to explain this in an interview

**"Why dbt?"**
> The feature logic lives in SQL and needs to be testable, documented, and runnable
> by anyone with a database connection. dbt gives you schema tests, a lineage graph,
> and materializations out of the box. The mart was also a good way to validate that
> the Python pipeline produces the same numbers.

**"Why Airflow?"**
> It's the standard for ML pipelines that need retries, scheduling, and an audit trail.
> The DAG makes the dependency between fetch → build → train → evaluate → promote
> explicit and observable. You can look at the Airflow UI and see exactly what ran,
> what failed, and why.

**"Why not just a cron script?"**
> A cron script would work. The DAG adds: dependency tracking (build won't run if
> fetch failed), retries on transient failures, XCom to pass paths between tasks,
> and a UI to see historical runs. For a portfolio project the overhead is worth it
> as a demonstration.

**"What's MLflow for?"**
> Parameter and metric logging. After five retrains you want to compare runs:
> did changing the training window improve validation MAE? MLflow gives you a UI
> for that. `log_artifact` also keeps a copy of each model file so you can
> reproduce any run.

**"What would you do differently in production?"**
> Several things: (1) replace the filesystem MLflow backend with a SQLite or
> PostgreSQL backend, (2) port the rebalancing feature logic from the notebook into
> `src/features.py` so the retrain model matches production quality, (3) add
> monitoring for data drift on the scraper output, (4) use a proper secrets manager
> instead of `.env` files.

**"The dbt mart and features.py do the same thing — isn't that redundant?"**
> Yes, deliberately. The mart is the contract: it's tested, documented, and anyone
> can query it. The Python version is the runtime: it runs inside Docker without a
> dbt installation. The cross-check (`check_mart.py`) is the value — it proves
> the two implementations agree, which means the SQL logic and the pipeline logic
> can't silently diverge.
