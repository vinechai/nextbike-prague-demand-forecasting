# nextbike prague — demand forecasting

End-to-end ML project: scrape live bike-sharing data, build a feature pipeline, train a demand forecast model, and serve predictions through a live API and map dashboard.

**live demo**: [nextbike-prague-demand-forecasting.streamlit.app](https://nextbike-prague-demand-forecasting.streamlit.app/)

## stack

python · lightgbm · fastapi · streamlit · postgresql (supabase) · dbt · airflow · github actions

## what it does

- scrapes nextbike prague every 10 minutes via github actions, stores in supabase
- builds hourly features: lag availability, time, weather (open-meteo), geospatial (osm)
- trains a lightgbm model to predict avg bikes per station for any requested hour
- serves predictions via fastapi, displays them on a live pydeck map

## architecture

```
github actions scraper (every 10 min)
       |
supabase postgres (production DB, read-only for training)
       |
       +--- api/predictor.py        live feature assembly + model inference
       |
       +--- src/features.py         offline feature pipeline
               |
               dbt mart              SQL view of the same features (for inspection)
               |
               src/train.py          lightgbm training + mlflow logging
               |
               src/evaluate.py       MAE comparison vs reference + naive lag_24h
               |
               airflow retrain DAG   weekly retraining (paused, run manually to promote)
```

## model

test set: 2026-07-26 to 2026-08-30 · 437k rows · 760 stations (is\_spot=TRUE, mean availability >= 0.5)

| model | features | test MAE | test R² |
|---|---|---|---|
| naive (lag\_24h) | — | 1.009 | 0.359 |
| naive (lag\_168h) | — | 1.629 | -0.288 |
| lightgbm | time + lag | 0.896 | 0.566 |
| lightgbm | + geo | 0.893 | 0.573 |
| lightgbm | + weather | 0.897 | 0.567 |
| **lightgbm** | **+ geo + weather (reference)** | **0.893** | **0.580** |
| xgboost | all features (300k sample) | 0.973 | 0.578 |
| random forest | all features (300k sample) | 0.973 | 0.581 |

**retrain comparison (same hyperparameters, Mar–Jul vs Mar–Aug training):** candidate MAE 0.893 / R² 0.577 vs reference MAE 0.904 / R² 0.568 on 395k identical rows. the 0.011 MAE gap is smaller than natural run-to-run variance and should not be interpreted as a real improvement.

**main difficulty**: operators move bikes by truck outside of organic demand. detecting rebalancing from raw scraper data is the core data engineering challenge — we built 4 heuristic detection rules but some events still slip through and distort lag features. a clean dataset with operator truck logs would make modeling significantly easier.

beats the naive lag\_24h baseline by ~11% (MAE 0.893 vs 1.009). lag\_24h (same hour yesterday) is the single most important feature.

## running locally

```bash
docker compose up -d           # local postgres (nextbike-postgres-1)
pip install -r requirements.txt
cp .env.example .env           # fill in SUPABASE_DATABASE_URL
python sync_supabase.py        # pull data from supabase into local docker
uvicorn api.main:app --reload  # api at localhost:8000
streamlit run dashboard/app.py # dashboard at localhost:8501
```

## known limitations

- **rebalancing detection is approximate.** four heuristic rules flag rebalancing events from raw scraper data. operator truck logs would make this exact; without them, some events slip through and distort lag features. a cross-validated AUC score for the detection itself was never computed.
- **scraper gaps create NaN lags.** the scraper runs every 10 minutes via github actions; CI outages leave gaps. `lag_24h` falls back to `lag_48h` when the 24h look-back hour was post-rebalancing; otherwise NaN propagates. ~2.5% of rows have NaN lags in the test period.
- **feature parity is approximate for three station-level features.** `avail_std`, `reb_per_week`, and `lag_24h_ratio` show corr 0.79–0.93 between `src/features.py` output and the reference parquet, because the reference used a 6-month training window while the pipeline computes them from the build window. see `docs/PARITY_REPORT.md`.
- **Airflow DAG runs end-to-end but candidate never promoted.** the two scheduled runs (Oct 19 and Oct 26, executed Nov 1–2 2026) completed all tasks and ended in skip_promote. the logs say: `"[evaluate] candidate: MAE=1.1785 R2=0.3967"`, `"[evaluate] naive lag_24h: MAE=1.0727 beats_naive=False"`, `"[evaluate] promote? False (must beat BOTH reference AND naive lag_24h)"`. the candidate beat the reference model (MAE 1.1785 < 1.2110) but not the naive lag_24h baseline, so promotion was correctly withheld. the eval window (Aug–Sep 2026) differs from the original test set, so the numbers are not comparable to the 0.893 benchmark above.
- **Delta Lake / Spark not run.** `lakehouse/` is a learning exercise. it requires Java 11+ and was never executed against real data.
- **Databricks not used.** nothing in this project connects to Databricks.

## tests

```bash
pytest tests/ -v
```

41 tests across five files: lag correctness, ±1h tolerance, no target leakage, month formula, rebalancing detection, station filter, evaluate row-count guard, bootstrap promotion CI, station stats extraction (including precomputed training stats), parity CSV tolerances. all pass. parity tests use a committed fixture so they never skip in CI.

ci: `.github/workflows/pytest.yml` runs on every push/PR with a postgres service container.

## structure

```
scrape_prague.py         scraper (github actions, every 10 min)
notebooks/
  01_eda.ipynb           rebalancing detection, demand patterns
  02_features.ipynb      feature engineering
  03_model.ipynb         training, evaluation, comparison
api/
  main.py                fastapi endpoints
  predictor.py           inference: lag queries, feature assembly, model
dashboard/app.py         streamlit map + station detail
src/
  features.py            offline feature pipeline (mirrors api/predictor.py formulas)
  train.py               lightgbm training + mlflow logging
  evaluate.py            candidate vs reference comparison
  db.py                  psycopg2 connection pool
data/
  model.lgb              trained model (reference, do not overwrite)
  features.parquet        precomputed feature table (lag baseline + active station list)
  feature_cols.json       feature list for inference
  station_geo.parquet     geo features for 1,513 is_spot=TRUE stations
db/schema.sql            postgres schema
tests/                   pytest suite (19 tests)
.github/workflows/
  pytest.yml             ci: pytest on every push
```

## data engineering extras (branch `de-upgrades`)

- **dbt/** — SQL transformations on raw postgres tables: 4 source declarations, staging views,
  hourly aggregation, `mart_station_hourly_features`. lags in the mart use a timestamp join to handle
  scraper gaps correctly. run with `cd dbt && dbt run --profiles-dir . && dbt test --profiles-dir .`.
  see [dbt/README.md](dbt/README.md).

- **airflow/** — weekly retrain DAG (monday 13:00 Europe/Prague): fetch\_weather → build\_features →
  train\_candidate → evaluate → promote if beats reference AND naive lag\_24h. never overwrites
  `data/model.lgb` automatically — promotion writes a versioned file only.
  run with docker compose in `airflow/`. see [airflow/README.md](airflow/README.md).

- **lakehouse/** — PySpark / Delta Lake bronze/silver/gold pipeline. learning exercise — the dataset
  fits in pandas at this scale. **not run**: requires java 11+.
  see [lakehouse/README.md](lakehouse/README.md).
