{{ config(materialized='table') }}

SELECT DISTINCT
    vehicle_id,

    CASE
        WHEN avg_fuel_level >= 70 THEN 'GOOD'
        WHEN avg_fuel_level >= 40 THEN 'MEDIUM'
        ELSE 'LOW'
    END AS fuel_health,

    CASE
        WHEN avg_battery >= 70 THEN 'GOOD'
        WHEN avg_battery >= 40 THEN 'MEDIUM'
        ELSE 'LOW'
    END AS battery_health

FROM {{ ref('stg_vehicle_gold') }}
