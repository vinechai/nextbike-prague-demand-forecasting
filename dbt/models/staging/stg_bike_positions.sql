with source as (
    select * from {{ source('nextbike', 'bike_positions') }}
),

cleaned as (
    select
        id                 as position_id,
        bike_id,
        station_uid,
        lat,
        lng,
        first_seen_at,
        last_seen_at,
        -- how long the bike stayed at this station in minutes
        extract(epoch from (last_seen_at - first_seen_at)) / 60.0 as duration_minutes
    from source
    where last_seen_at >= first_seen_at   -- sanity check
)

select * from cleaned
