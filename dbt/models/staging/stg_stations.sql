with source as (
    select * from {{ source('nextbike', 'stations') }}
),

cleaned as (
    select
        uid                 as station_id,
        name,
        lat,
        lng,
        coalesce(bike_racks, 0)  as bike_racks,
        is_spot,
        is_bike,
        first_seen_at,
        -- extract district number from name prefix "P<n>" (e.g. "P1-123" → 1)
        case
            when name ~ '^P[0-9]+'
            then cast(substring(name from '[0-9]+') as integer)
            else null
        end as district
    from source
    where is_spot = true   -- drop free-floating virtual spots
)

select * from cleaned
