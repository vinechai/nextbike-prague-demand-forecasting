-- aggregate scraper snapshots to one row per (station, hour)
-- multiple snapshots fall in the same hour, so we average them
with snapshots as (
    select * from {{ ref('stg_station_snapshots') }}
),

aggregated as (
    select
        station_uid,
        hour_local,
        round(avg(bikes_available_to_rent)::numeric, 4) as avg_available,
        min(bikes_available_to_rent)                    as min_available,
        max(bikes_available_to_rent)                    as max_available,
        count(*)                                        as n_snapshots
    from snapshots
    group by station_uid, hour_local
)

select * from aggregated
