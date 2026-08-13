{{ config(
    materialized='table'
) }}

SELECT
    CAST(1 AS INT) AS id,
    'dbt_spark' AS engine,
    'hive_metastore' AS catalog,
    'delta' AS storage_format

UNION ALL

SELECT
    CAST(2 AS INT) AS id,
    'spark_4' AS engine,
    'hive_metastore' AS catalog,
    'delta' AS storage_format

UNION ALL

SELECT
    CAST(3 AS INT) AS id,
    'vehicle_platform' AS engine,
    'persistent' AS catalog,
    'delta' AS storage_format
