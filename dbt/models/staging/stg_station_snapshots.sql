with source as (
    select * from {{ source('nextbike', 'station_snapshots') }}
),

cleaned as (
    select
        id                          as snapshot_id,
        station_uid,
        bikes_available,
        bikes_available_to_rent,
        free_racks,
        booked_bikes,
        scrape_time,
        -- floor scrape time to the Prague local hour for grouping
        date_trunc(
            'hour',
            scrape_time at time zone 'Europe/Prague'
        )                           as hour_local
    from source
    where bikes_available_to_rent >= 0   -- drop rare -1 sentinel values
)

select * from cleaned
