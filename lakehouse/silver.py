"""clean and deduplicate bronze → silver Delta tables.

silver rules:
  station_snapshots:
    - drop duplicate (station_uid, scrape_time) rows (keep first)
    - drop bikes_available_to_rent < 0 (sensor error)
    - cast types, add hour_local column
  weather_hourly:
    - deduplicate on hour
    - coerce nulls to NaN-friendly float

requires java 11+ and pyspark / delta-spark.

usage:
    python lakehouse/silver.py
"""

import os

_HERE   = os.path.dirname(os.path.abspath(__file__))
_BRONZE = os.path.join(_HERE, "delta", "bronze")
_SILVER = os.path.join(_HERE, "delta", "silver")


def _get_spark():
    try:
        from pyspark.sql import SparkSession
        from delta import configure_spark_with_delta_pip
    except ImportError:
        raise SystemExit("pyspark/delta-spark not installed — see lakehouse/README.md")

    builder = (
        SparkSession.builder
        .appName("nextbike-silver")
        .config("spark.sql.extensions", "io.delta.sql.DeltaSparkSessionExtension")
        .config("spark.sql.catalog.spark_catalog", "org.apache.spark.sql.delta.catalog.DeltaCatalog")
        .config("spark.driver.memory", "2g")
    )
    return configure_spark_with_delta_pip(builder).getOrCreate()


def silver_snapshots(spark) -> None:
    from pyspark.sql import functions as F

    src = os.path.join(_BRONZE, "station_snapshots")
    dst = os.path.join(_SILVER, "station_snapshots")

    if not os.path.exists(src):
        print("[silver] station_snapshots: no bronze data — skip")
        return

    df = spark.read.format("delta").load(src)

    df = (
        df
        .filter(F.col("bikes_available_to_rent") >= 0)
        .dropDuplicates(["station_uid", "scrape_time"])
        .withColumn(
            "hour_local",
            F.date_trunc("hour", F.to_timestamp(F.col("scrape_time")).cast("timestamp"))
            # note: full Prague tz conversion requires spark-timezone support
            # simplified here; for exact Prague local time add: .withColumn(..., F.from_utc_timestamp(..., "Europe/Prague"))
        )
        .orderBy("station_uid", "scrape_time")
    )

    n = df.count()
    print(f"[silver] station_snapshots: {n:,} rows → {dst}")
    df.write.format("delta").mode("overwrite").option("overwriteSchema", "true").save(dst)


def silver_weather(spark) -> None:
    from pyspark.sql import functions as F

    src = os.path.join(_BRONZE, "weather_hourly")
    dst = os.path.join(_SILVER, "weather_hourly")

    if not os.path.exists(src):
        print("[silver] weather_hourly: no bronze data — skip")
        return

    df = spark.read.format("delta").load(src)
    df = df.dropDuplicates(["hour"]).orderBy("hour")

    n = df.count()
    print(f"[silver] weather_hourly: {n:,} rows → {dst}")
    df.write.format("delta").mode("overwrite").option("overwriteSchema", "true").save(dst)


if __name__ == "__main__":
    spark = _get_spark()
    silver_snapshots(spark)
    silver_weather(spark)
    spark.stop()
