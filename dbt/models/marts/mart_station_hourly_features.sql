-- mart: one row per (station, Prague local hour) with time, weather, geo, and lag features
--
-- lag features use timestamp self-joins (not row-based window functions).
-- if there is no row for exactly 24/48/168 hours earlier (scraper gap), the join
-- returns NULL — which is honest. row-based lag() would silently return the wrong row.
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

        -- lag features via timestamp join — NULL when the hour is missing (correct for gaps)
        lag24.avg_available  as lag_24h,
        lag48.avg_available  as lag_48h,
        lag168.avg_available as lag_168h

    from hourly h
    left join hourly lag24
        on  lag24.station_uid = h.station_uid
        and lag24.hour_local  = h.hour_local - interval '24 hours'
    left join hourly lag48
        on  lag48.station_uid = h.station_uid
        and lag48.hour_local  = h.hour_local - interval '48 hours'
    left join hourly lag168
        on  lag168.station_uid = h.station_uid
        and lag168.hour_local  = h.hour_local - interval '168 hours'
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
