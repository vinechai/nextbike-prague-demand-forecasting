"""retrain pipeline DAG — left paused at creation (DAGS_ARE_PAUSED_AT_CREATION=true).

flow:
    fetch_weather → build_features → train_candidate → evaluate → promote_or_skip

promote writes a versioned model file but never overwrites data/model.lgb.

to test locally (without docker):
    pip install apache-airflow lightgbm mlflow pandas pyarrow psycopg2-binary holidays
    export AIRFLOW__CORE__DAGS_FOLDER=$(pwd)/airflow/dags
    airflow dags test retrain_pipeline 2026-10-29
"""

import os
import shutil
from datetime import datetime, timedelta

from airflow import DAG
from airflow.operators.python import PythonOperator, BranchPythonOperator
from airflow.operators.empty import EmptyOperator

# data paths — relative to /opt/airflow inside the container,
# or to the project root when running locally
_DATA_DIR    = os.environ.get("AIRFLOW_DATA_DIR", "data")
_MODELS_DIR  = os.path.join(_DATA_DIR, "models")
_GEO_PARQUET = os.path.join(_DATA_DIR, "features.parquet")
_REF_MODEL   = os.path.join(_DATA_DIR, "model.lgb")

_SINCE = os.environ.get("TRAIN_SINCE", "2026-04-01")
_UNTIL = os.environ.get("TRAIN_UNTIL", "2026-08-01")
_EVAL_SINCE = os.environ.get("EVAL_SINCE",  "2026-08-01")
_EVAL_UNTIL = os.environ.get("EVAL_UNTIL",  "2026-09-01")


default_args = {
    "owner": "nextbike",
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
}


# ── task functions ──────────────────────────────────────────────────────────────

def _fetch_weather(**ctx):
    """check that weather data covers the training window.

    in production this would call the open-meteo api and insert missing rows.
    here we just verify that at least one row exists in weather_hourly.
    """
    import sys
    sys.path.insert(0, "/opt/airflow")

    from src.db import get_conn

    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute("select count(*) from weather_hourly where hour >= %s", (_SINCE,))
            n = cur.fetchone()[0]

    if n == 0:
        raise ValueError(f"no weather rows since {_SINCE} — run backfill_weather.py first")
    print(f"[fetch_weather] {n} weather rows found since {_SINCE}")


def _build_features(**ctx):
    """query db and build a candidate features parquet."""
    import sys
    sys.path.insert(0, "/opt/airflow")

    from src.features import build_features

    out = os.path.join(_DATA_DIR, "features_candidate.parquet")
    df = build_features(since=_SINCE, until=_UNTIL, geo_parquet=_GEO_PARQUET)
    df.to_parquet(out, index=False)
    print(f"[build_features] saved {len(df):,} rows → {out}")
    return out


def _train_candidate(**ctx):
    """train a lightgbm model on features_candidate.parquet."""
    import sys
    sys.path.insert(0, "/opt/airflow")

    from src.train import train

    run_date = ctx["ds_nodash"]   # e.g. "20261029"
    os.makedirs(_MODELS_DIR, exist_ok=True)
    out = os.path.join(_MODELS_DIR, f"model_{run_date}.lgb")
    features_path = os.path.join(_DATA_DIR, "features_candidate.parquet")

    result = train(
        features_path=features_path,
        output_path=out,
        train_cutoff=_UNTIL,
        mlflow_run_name=f"retrain_{run_date}",
    )
    ctx["ti"].xcom_push(key="candidate_path", value=out)
    ctx["ti"].xcom_push(key="train_mae",      value=result["train_mae"])
    return out


def _evaluate(**ctx):
    """compare candidate vs reference model on the eval window."""
    import sys
    sys.path.insert(0, "/opt/airflow")

    from src.evaluate import evaluate

    ti = ctx["ti"]
    candidate = ti.xcom_pull(task_ids="train_candidate", key="candidate_path")

    if not os.path.exists(_REF_MODEL):
        print(f"[evaluate] no reference model at {_REF_MODEL} — treating candidate as better")
        ti.xcom_push(key="is_better", value=True)
        return

    features_path = os.path.join(_DATA_DIR, "features_candidate.parquet")
    result = evaluate(
        candidate_path=candidate,
        reference_path=_REF_MODEL,
        features_path=features_path,
        since=_EVAL_SINCE,
        until=_EVAL_UNTIL,
    )
    ti.xcom_push(key="is_better",      value=result["is_better"])
    ti.xcom_push(key="candidate_mae",  value=result["candidate_mae"])
    ti.xcom_push(key="reference_mae",  value=result["reference_mae"])


def _branch_promote(**ctx):
    is_better = ctx["ti"].xcom_pull(task_ids="evaluate", key="is_better")
    return "promote" if is_better else "skip_promote"


def _promote(**ctx):
    """copy candidate to data/models/model_best_<date>.lgb.

    deliberately does NOT overwrite data/model.lgb — that's the production
    model and should only be updated manually after human review.
    """
    ti = ctx["ti"]
    candidate = ti.xcom_pull(task_ids="train_candidate", key="candidate_path")
    run_date   = ctx["ds_nodash"]
    best_path  = os.path.join(_MODELS_DIR, f"model_best_{run_date}.lgb")
    shutil.copy2(candidate, best_path)
    print(f"[promote] candidate is better → saved as {best_path}")
    print("[promote] to deploy: manually copy to data/model.lgb and restart the API")


# ── DAG definition ──────────────────────────────────────────────────────────────

with DAG(
    dag_id="retrain_pipeline",
    default_args=default_args,
    description="fetch weather → build features → train → evaluate → promote if better",
    schedule="0 3 * * 1",   # mondays at 03:00 UTC
    start_date=datetime(2026, 10, 1),
    catchup=False,
    tags=["nextbike", "ml"],
) as dag:

    fetch_weather = PythonOperator(
        task_id="fetch_weather",
        python_callable=_fetch_weather,
    )

    build_features_task = PythonOperator(
        task_id="build_features",
        python_callable=_build_features,
    )

    train_candidate = PythonOperator(
        task_id="train_candidate",
        python_callable=_train_candidate,
    )

    evaluate = PythonOperator(
        task_id="evaluate",
        python_callable=_evaluate,
    )

    branch = BranchPythonOperator(
        task_id="branch_promote",
        python_callable=_branch_promote,
    )

    promote = PythonOperator(
        task_id="promote",
        python_callable=_promote,
    )

    skip_promote = EmptyOperator(task_id="skip_promote")

    (
        fetch_weather
        >> build_features_task
        >> train_candidate
        >> evaluate
        >> branch
        >> [promote, skip_promote]
    )
