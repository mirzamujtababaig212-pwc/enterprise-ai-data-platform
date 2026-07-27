{{ config(materialized='table') }}

SELECT

    vehicle_id,

    avg_speed,

    max_speed,

    avg_fuel_level,

    avg_battery,

    max_engine_temperature,

    total_events,

    CASE

        WHEN avg_battery < 20 THEN 'Critical'

        WHEN avg_battery < 50 THEN 'Warning'

        ELSE 'Healthy'

    END AS battery_health,

    CASE

        WHEN avg_fuel_level < 15 THEN 'Low'

        ELSE 'Normal'

    END AS fuel_health

FROM {{ ref('stg_vehicle_gold') }}
