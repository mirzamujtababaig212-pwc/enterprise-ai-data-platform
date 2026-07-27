{{ config(materialized='table') }}

SELECT

    battery_health,

    fuel_health,

    COUNT(*) AS vehicles,

    AVG(avg_speed) AS fleet_avg_speed,

    AVG(avg_fuel_level) AS fleet_avg_fuel,

    AVG(avg_battery) AS fleet_avg_battery

FROM {{ ref('int_vehicle_health') }}

GROUP BY
    battery_health,
    fuel_health
