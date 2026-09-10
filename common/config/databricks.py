import os

from dotenv import load_dotenv

load_dotenv()


class DatabricksConfig:
    """
    Centralized Databricks workspace configuration.

    Connection details are resolved from environment variables so
    workspace-specific configuration remains external to application code.
    """

    HOST = os.getenv("DATABRICKS_HOST")
    TOKEN = os.getenv("DATABRICKS_TOKEN")

    @classmethod
    def options(cls):
        """
        Return configured Databricks connection values.

        Empty values are excluded so callers can provide or override
        configuration explicitly.
        """

        values = {
            "host": cls.HOST,
            "token": cls.TOKEN,
        }

        return {key: value for key, value in values.items() if value is not None and value != ""}
