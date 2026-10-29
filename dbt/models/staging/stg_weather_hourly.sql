with source as (
    select * from {{ source('nextbike', 'weather_hourly') }}
),

cleaned as (
    select
        hour                                                          as hour_utc,
        -- convert to Prague local time for joining with station data
        (hour at time zone 'UTC' at time zone 'Europe/Prague')        as hour_local,
        temperature,
        precipitation,
        windspeed,
        weathercode,
        snowfall
    from source
)

select * from cleaned
