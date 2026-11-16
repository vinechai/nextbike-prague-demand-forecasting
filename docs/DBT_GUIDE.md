# dbt guide

dbt transforms raw postgres tables into model-ready features.
the mart is used for **inspection and debugging only** — `src/features.py` is the
source of truth for the features the model is actually trained on.

---

## commands

```bash
cd dbt

# run all models
dbt run --profiles-dir .

# run tests
dbt test --profiles-dir .

# generate and open the documentation site
dbt docs generate --profiles-dir .
dbt docs serve --profiles-dir . --port 8081
# then open http://localhost:8081
```

to run a single model:
```bash
dbt run --profiles-dir . --select mart_station_hourly_features
```

---

## models (6 total)

### staging — 4 views

views run on top of the raw tables without storing any data.
"view" means postgres executes the SQL every time something queries the view.

| model | source table | purpose |
|---|---|---|
| `stg_stations` | `stations` | drops virtual `is_bike` rows, extracts district number from name |
| `stg_station_snapshots` | `station_snapshots` | drops negative bike counts, adds `hour_local` (Prague time) |
| `stg_bike_positions` | `bike_positions` | adds `duration_minutes` from first/last seen timestamps |
| `stg_weather_hourly` | `weather_hourly` | adds `hour_local` (Prague time) alongside the UTC column |

### intermediate — 1 table

stores the result to disk so the mart doesn't re-aggregate millions of snapshots.

| model | purpose |
|---|---|
| `int_station_hourly_avg` | averages bike counts across all scrapes in each (station, hour). one row per station-hour. 3.8 M rows. takes ~75s to build. |

### mart — 1 table

| model | purpose |
|---|---|
| `mart_station_hourly_features` | adds lag features, time features, and weather joins to `int_station_hourly_avg`. one row per station-hour with all model input columns. 3.8 M rows. takes ~207s to build (three self-joins, no index on station_uid+hour_local). |

---

## lag implementation

the mart's lag features use **timestamp self-joins**, not row-based `LAG()`:

```sql
LEFT JOIN int_station_hourly_avg lag24
  ON lag24.station_uid = h.station_uid
  AND lag24.hour_local = h.hour_local - interval '24 hours'
```

this is correct — it handles scraper gaps. a row-based `LAG()` would silently
skip the gap and return the wrong hour. `src/features.py` uses the same
timestamp approach.

---

## tests (45 total)

### source tests (18) — on raw tables

| table | column | tests |
|---|---|---|
| `stations` | uid | unique, not_null |
| `stations` | name, lat, lng | not_null (3) |
| `station_snapshots` | id | unique, not_null |
| `station_snapshots` | station_uid | not_null, relationships to stations.uid |
| `station_snapshots` | bikes_available_to_rent | not_null |
| `station_snapshots` | scrape_time | not_null |
| `bike_positions` | id | unique, not_null |
| `bike_positions` | bike_id | not_null |
| `bike_positions` | station_uid | not_null, relationships to stations.uid |
| `weather_hourly` | hour | unique, not_null |

### model tests (27) — on transformed data

| model | column | tests |
|---|---|---|
| `stg_stations` | station_id | unique, not_null |
| `stg_stations` | lat, lng | not_null (2) |
| `stg_station_snapshots` | snapshot_id | unique, not_null |
| `stg_station_snapshots` | station_uid, hour_local | not_null (2) |
| `stg_station_snapshots` | bikes_available_to_rent | not_null, accepted_range 0–200 |
| `stg_bike_positions` | position_id | unique, not_null |
| `stg_bike_positions` | duration_minutes | accepted_range ≥ 0 |
| `stg_weather_hourly` | hour_utc | unique, not_null |
| `stg_weather_hourly` | hour_local | not_null |
| `int_station_hourly_avg` | station_uid, hour_local | not_null (2) |
| `int_station_hourly_avg` | avg_available | not_null, accepted_range ≥ 0 |
| `mart_station_hourly_features` | station_uid, hour_local | not_null (2) |
| `mart_station_hourly_features` | avg_available | not_null, accepted_range ≥ 0 |
| `mart_station_hourly_features` | hour_of_day | accepted_range 0–23 |
| `mart_station_hourly_features` | dow | accepted_range 0–6 |
| `mart_station_hourly_features` | month | accepted_range 1–12 |

---

## environment / credentials

connection settings are read from environment variables. there are no hardcoded
secrets in `profiles.yml`.

```bash
export DBT_DB_HOST=localhost
export DBT_DB_PORT=5432
export DBT_DB_NAME=nextbike
export DBT_DB_USER=nextbike
export DBT_DB_PASS=nextbike    # local docker default
```

`dbt/profiles.yml` is listed in `.gitignore` — it is not committed.
to recreate it locally, copy the template:

```bash
cp dbt/profiles.yml.example dbt/profiles.yml
# then set environment variables above
```
