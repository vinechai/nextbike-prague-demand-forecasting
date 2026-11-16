# dbt — nextbike prague

SQL transformations on top of the raw postgres tables.

## models

```
models/
  _sources.yml                    declare the 4 raw tables as sources
  staging/
    stg_stations.sql              cleaned stations; is_bike rows dropped
    stg_station_snapshots.sql     snapshots + hour_local column
    stg_bike_positions.sql        bike stays + duration_minutes
    stg_weather_hourly.sql        weather + hour_local column
  intermediate/
    int_station_hourly_avg.sql    aggregate snapshots → 1 row per (station, hour)
  marts/
    mart_station_hourly_features.sql  full feature table (see note on lags below)
```

**lag note**: lag columns in the mart use timestamp self-joins
(`lag24.hour_local = h.hour_local - interval '24 hours'`), not row-based `lag(n)`.
this matches `src/features.py` and handles scraper gaps correctly — a gap produces
NaN instead of silently returning the wrong hour's value. see `docs/DBT_GUIDE.md`.

## setup

```bash
pip install dbt-postgres
cd dbt
dbt deps               # install dbt_utils
dbt run --profiles-dir .
dbt test --profiles-dir .
dbt docs generate --profiles-dir . && dbt docs serve --profiles-dir .
```

env vars for the db connection (defaults match local docker compose):

| var | default |
|-----|---------|
| DBT_DB_HOST | localhost |
| DBT_DB_PORT | 5432 |
| DBT_DB_USER | nextbike |
| DBT_DB_PASS | nextbike |
| DBT_DB_NAME | nextbike |

## compare against features.parquet

after `dbt run`, run:

```bash
python check_mart.py
```
