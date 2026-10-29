# lakehouse — nextbike prague

bronze / silver / gold Delta Lake tables built with PySpark.

> **this is a learning exercise**, not a production requirement.
> the dataset is ~3.7M station-hour rows — pandas handles it fine.
> pyspark adds value at 100M+ rows or when you need ACID table history, schema evolution, or
> a time-travel audit log. it's included here to demonstrate the pattern.

## requirements

- **java 11+** (required by pyspark — not installed on the dev machine at time of writing)
- python 3.11+
- `pip install -r requirements.txt`

## layout

```
lakehouse/
  raw/                       parquet snapshots exported from postgres
    station_snapshots/date=YYYY-MM-DD/*.parquet
    weather_hourly/date=YYYY-MM-DD/*.parquet
  delta/
    bronze/                  raw data as Delta tables (append-only)
    silver/                  cleaned, deduped
    gold/                    hourly station features (ready for ML)
```

## pipeline

```bash
# 1. export raw data from local postgres to parquet
python lakehouse/export_snapshot.py --since 2026-04-01 --until 2026-09-01

# 2. write bronze Delta tables
python lakehouse/bronze.py

# 3. clean and deduplicate → silver
python lakehouse/silver.py

# 4. aggregate to hourly features → gold
python lakehouse/gold.py
```

## notes

- Delta tables land in `lakehouse/delta/` (gitignored — too large for repo)
- the `export_snapshot.py` script reads from local docker postgres (no supabase)
- if Java is not installed: install it from https://adoptium.net then re-run
