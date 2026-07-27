{{ config(materialized='table') }}

SELECT

    vehicle_id,

    avg_speed,

    max_speed,

    avg_fuel_level,

    avg_battery,

    max_engine_temperature,

    total_events

FROM {{ ref('int_vehicle_health') }}
