{{
    config(
        materialized = 'table',
        post_hook = "CREATE INDEX IF NOT EXISTS {{ this.name }}_uid_hour
                     ON {{ this }} (station_uid, hour_local)"
    )
}}
-- aggregate scraper snapshots to one row per (station, hour)
-- materialized as a table so the lag self-joins in the mart are fast
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
