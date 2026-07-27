{{ config(materialized='table') }}

SELECT

    battery_health,

    fuel_health,

    COUNT(*) AS vehicle_count,

    ROUND(AVG(avg_speed)::numeric,2) AS avg_speed,

    ROUND(AVG(avg_battery)::numeric,2) AS avg_battery,

    ROUND(AVG(avg_fuel_level)::numeric,2) AS avg_fuel

FROM {{ ref('int_vehicle_health') }}

GROUP BY

    battery_health,
    fuel_health
