import os

from dotenv import load_dotenv

load_dotenv()


class SnowflakeConfig:
    """
    Centralized Snowflake connection configuration.

    Credentials and connection details are resolved from environment
    variables so they are not hardcoded in pipeline configuration files.
    """

    ACCOUNT = os.getenv("SNOWFLAKE_ACCOUNT")
    USER = os.getenv("SNOWFLAKE_USER")
    PASSWORD = os.getenv("SNOWFLAKE_PASSWORD")

    DATABASE = os.getenv("SNOWFLAKE_DATABASE")
    SCHEMA = os.getenv("SNOWFLAKE_SCHEMA")
    WAREHOUSE = os.getenv("SNOWFLAKE_WAREHOUSE")
    ROLE = os.getenv("SNOWFLAKE_ROLE")

    @classmethod
    def options(cls):
        """
        Return Spark Snowflake connector options.

        Empty values are excluded so callers can override or provide
        only the configuration required by their environment.
        """

        values = {
            "sfURL": cls.ACCOUNT,
            "sfUser": cls.USER,
            "sfPassword": cls.PASSWORD,
            "sfDatabase": cls.DATABASE,
            "sfSchema": cls.SCHEMA,
            "sfWarehouse": cls.WAREHOUSE,
            "sfRole": cls.ROLE,
        }

        return {key: value for key, value in values.items() if value is not None and value != ""}
