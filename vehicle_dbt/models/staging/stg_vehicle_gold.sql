{{ config(materialized='view') }}

SELECT
    vehicle_id,
    avg_speed,
    max_speed,
    avg_fuel_level,
    avg_battery,
    max_engine_temperature,
    total_events
FROM {{ source('vehicle_platform','vehicle_gold') }}
