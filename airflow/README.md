# airflow — nextbike retrain pipeline

weekly DAG that retrains a LightGBM model and promotes it only if it beats the current one.

## flow

```
fetch_weather → build_features → train_candidate → evaluate → promote (if better)
                                                             ↘ skip_promote (if worse)
```

- **fetch_weather**: verify weather rows exist; in production would backfill from open-meteo
- **build_features**: query local postgres → `data/features_candidate.parquet`
  - note: rebalancing features (`hours_since_reb`, etc.) are set to NaN — they need the full heuristic pipeline
- **train_candidate**: LightGBM with the same params as the production model, logged to MLflow
- **evaluate**: compare candidate vs `data/model.lgb` on the eval window (env var EVAL_SINCE/EVAL_UNTIL)
- **promote**: if better, copy to `data/models/model_best_<date>.lgb`; never overwrites `data/model.lgb`

mlflow runs land in `mlruns/` (gitignored).

## setup

```bash
cp .env.example .env            # fill in db connection if different from defaults
docker compose up airflow-init  # first run only
docker compose up -d
# webserver at http://localhost:8080 (admin/admin)
```

the DAG is created **paused** — enable it in the UI when ready.

to test locally without docker (requires airflow + deps installed):

```bash
pip install -r requirements.txt
export AIRFLOW__CORE__DAGS_FOLDER=$(pwd)/airflow/dags
airflow dags test retrain_pipeline 2026-10-29
```

## notes

- `data/model.lgb` is never overwritten automatically — deploy manually after review
- the DAG container mounts `../src` and `../data` so all pipeline code and data are shared
- if the nextbike postgres is not on `host.docker.internal:5432`, edit `NEXTBIKE_DB_HOST` in `.env`
