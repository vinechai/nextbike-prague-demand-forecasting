-- mart: one row per (station, Prague local hour) with time, weather, geo, and lag features
--
-- lag columns use SQL window functions (lag by N rows).
-- if there are gaps in the hourly series (scraper outages), these will differ from
-- features.parquet which uses an explicit time-offset join to handle gaps.
-- the check_mart.py script reports the size of that difference.
with hourly as (
    select * from {{ ref('int_station_hourly_avg') }}
),

stations as (
    select * from {{ ref('stg_stations') }}
),

weather as (
    select * from {{ ref('stg_weather_hourly') }}
),

time_features as (
    select
        h.station_uid,
        h.hour_local,
        h.avg_available,
        h.min_available,
        h.max_available,
        h.n_snapshots,

        -- time features
        extract(hour  from h.hour_local)::integer as hour_of_day,
        extract(dow   from h.hour_local)::integer as dow,
        extract(month from h.hour_local)::integer as month,
        case when extract(dow from h.hour_local) in (0, 6) then 1 else 0 end as is_weekend,

        -- lag features via window function (row-based, not time-based — see note above)
        lag(h.avg_available, 24)  over w as lag_24h,
        lag(h.avg_available, 48)  over w as lag_48h,
        lag(h.avg_available, 168) over w as lag_168h

    from hourly h
    window w as (partition by h.station_uid order by h.hour_local)
),

joined as (
    select
        t.*,
        s.lat,
        s.lng,
        s.bike_racks,
        s.district,

        -- weather at same Prague local hour
        w.temperature,
        w.precipitation,
        w.windspeed,
        w.weathercode,
        w.snowfall

    from time_features t
    left join stations s on s.station_id = t.station_uid
    left join weather  w on w.hour_local  = t.hour_local
)

select * from joined
