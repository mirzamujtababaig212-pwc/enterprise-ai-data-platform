import os

from dotenv import load_dotenv


load_dotenv()


class FabricConfig:
    """
    Centralized Microsoft Fabric configuration.

    Values are resolved from environment variables so environment-
    specific workspace and lakehouse configuration remains external
    to application code.
    """

    WORKSPACE = os.getenv("FABRIC_WORKSPACE")
    LAKEHOUSE = os.getenv("FABRIC_LAKEHOUSE")

    @classmethod
    def options(cls):
        """
        Return configured Fabric values.

        Empty values are excluded.
        """

        values = {
            "workspace": cls.WORKSPACE,
            "lakehouse": cls.LAKEHOUSE,
        }

        return {key: value for key, value in values.items() if value is not None and value != ""}
