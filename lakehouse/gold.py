"""aggregate silver → gold: one row per (station_uid, hour) with time + weather features.

this is the PySpark equivalent of int_station_hourly_avg + mart_station_hourly_features
from the dbt pipeline. outputs a Delta table ready for ML.

requires java 11+ and pyspark / delta-spark.

usage:
    python lakehouse/gold.py
"""

import os

import numpy as np

_HERE   = os.path.dirname(os.path.abspath(__file__))
_SILVER = os.path.join(_HERE, "delta", "silver")
_GOLD   = os.path.join(_HERE, "delta", "gold")


def _get_spark():
    try:
        from pyspark.sql import SparkSession
        from delta import configure_spark_with_delta_pip
    except ImportError:
        raise SystemExit("pyspark/delta-spark not installed — see lakehouse/README.md")

    builder = (
        SparkSession.builder
        .appName("nextbike-gold")
        .config("spark.sql.extensions", "io.delta.sql.DeltaSparkSessionExtension")
        .config("spark.sql.catalog.spark_catalog", "org.apache.spark.sql.delta.catalog.DeltaCatalog")
        .config("spark.driver.memory", "2g")
    )
    return configure_spark_with_delta_pip(builder).getOrCreate()


def gold_station_hourly(spark) -> None:
    from pyspark.sql import functions as F
    from pyspark.sql.window import Window

    snap_path = os.path.join(_SILVER, "station_snapshots")
    wx_path   = os.path.join(_SILVER, "weather_hourly")
    dst       = os.path.join(_GOLD,   "station_hourly_features")

    if not os.path.exists(snap_path):
        print("[gold] station_snapshots silver not found — skip")
        return

    snapshots = spark.read.format("delta").load(snap_path)
    weather   = spark.read.format("delta").load(wx_path) if os.path.exists(wx_path) else None

    # aggregate to hourly avg
    hourly = (
        snapshots
        .groupBy("station_uid", "hour_local")
        .agg(
            F.round(F.avg("bikes_available_to_rent"), 4).alias("avg_available"),
            F.min("bikes_available_to_rent").alias("min_available"),
            F.max("bikes_available_to_rent").alias("max_available"),
            F.count("*").alias("n_snapshots"),
        )
    )

    # time features
    hourly = (
        hourly
        .withColumn("hour_of_day", F.hour("hour_local"))
        .withColumn("dow",         F.dayofweek("hour_local") - 1)   # 0=Mon
        .withColumn("month",       F.month("hour_local"))
        .withColumn("is_weekend",  (F.dayofweek("hour_local").isin([1, 7])).cast("integer"))
        # cyclical
        .withColumn("hour_sin",  F.sin(2 * np.pi * F.col("hour_of_day") / 24))
        .withColumn("hour_cos",  F.cos(2 * np.pi * F.col("hour_of_day") / 24))
        .withColumn("dow_sin",   F.sin(2 * np.pi * F.col("dow") / 7))
        .withColumn("dow_cos",   F.cos(2 * np.pi * F.col("dow") / 7))
        .withColumn("month_sin", F.sin(2 * np.pi * (F.col("month") - 1) / 12))
        .withColumn("month_cos", F.cos(2 * np.pi * (F.col("month") - 1) / 12))
    )

    # lag features — row-based window (same limitation as dbt mart)
    w = Window.partitionBy("station_uid").orderBy("hour_local")
    hourly = (
        hourly
        .withColumn("lag_24h",  F.lag("avg_available", 24).over(w))
        .withColumn("lag_48h",  F.lag("avg_available", 48).over(w))
        .withColumn("lag_168h", F.lag("avg_available", 168).over(w))
    )

    # join weather
    if weather is not None:
        wx_cols = ["hour"] + [c for c in weather.columns if c != "hour"]
        weather = weather.withColumnRenamed("hour", "hour_utc")
        # simplified: join on truncated utc hour (not exact Prague local conversion)
        weather = weather.withColumn(
            "hour_local_approx",
            F.date_trunc("hour", F.to_timestamp("hour_utc"))
        )
        hourly = hourly.join(
            weather.drop("hour_utc"),
            hourly["hour_local"] == weather["hour_local_approx"],
            how="left",
        ).drop("hour_local_approx")

    n = hourly.count()
    print(f"[gold] station_hourly_features: {n:,} rows → {dst}")

    (
        hourly
        .write
        .format("delta")
        .mode("overwrite")
        .option("overwriteSchema", "true")
        .partitionBy("month")
        .save(dst)
    )
    print("[gold] done")


if __name__ == "__main__":
    spark = _get_spark()
    gold_station_hourly(spark)
    spark.stop()
