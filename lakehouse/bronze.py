"""write raw parquet exports to Delta bronze tables.

requires java 11+ and pyspark / delta-spark.
run export_snapshot.py first to generate the raw parquet files.

usage:
    python lakehouse/bronze.py
"""

import os

_HERE  = os.path.dirname(os.path.abspath(__file__))
_RAW   = os.path.join(_HERE, "raw")
_DELTA = os.path.join(_HERE, "delta", "bronze")


def _get_spark():
    try:
        from pyspark.sql import SparkSession
        from delta import configure_spark_with_delta_pip
    except ImportError:
        raise SystemExit(
            "pyspark / delta-spark not installed.\n"
            "run: pip install -r lakehouse/requirements.txt\n"
            "also requires java 11+ — see lakehouse/README.md"
        )

    builder = (
        SparkSession.builder
        .appName("nextbike-bronze")
        .config("spark.sql.extensions", "io.delta.sql.DeltaSparkSessionExtension")
        .config("spark.sql.catalog.spark_catalog", "org.apache.spark.sql.delta.catalog.DeltaCatalog")
        .config("spark.driver.memory", "2g")
    )
    return configure_spark_with_delta_pip(builder).getOrCreate()


def ingest_table(spark, table_name: str) -> None:
    src = os.path.join(_RAW, table_name)
    dst = os.path.join(_DELTA, table_name)

    if not os.path.exists(src):
        print(f"[bronze] {table_name}: no raw data at {src} — skip")
        return

    # read all date partitions
    df = spark.read.parquet(src)
    n  = df.count()
    print(f"[bronze] {table_name}: {n:,} rows → {dst}")

    (
        df.write
        .format("delta")
        .mode("overwrite")   # safe for initial load; use append + dedup for incremental
        .option("overwriteSchema", "true")
        .save(dst)
    )
    print(f"[bronze] {table_name}: written")


if __name__ == "__main__":
    spark = _get_spark()
    for table in ["station_snapshots", "weather_hourly"]:
        ingest_table(spark, table)
    spark.stop()
