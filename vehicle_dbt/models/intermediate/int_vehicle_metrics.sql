select
    vehicle_id,

    avg_speed,

    max_speed,

    avg_fuel_level,

    avg_battery,

    max_engine_temperature,

    total_events,

    case
        when avg_speed > 100 then 'High'
        when avg_speed > 70 then 'Medium'
        else 'Low'
    end as driving_category,

    case
        when avg_battery < 20 then 'Critical'
        when avg_battery < 50 then 'Warning'
        else 'Healthy'
    end as battery_health

from {{ ref('stg_vehicle_gold') }}
