import os

from dotenv import load_dotenv

load_dotenv()


class PostgresConfig:
    HOST = os.getenv("POSTGRES_HOST", "localhost")
    PORT = os.getenv("POSTGRES_PORT", "5432")
    DATABASE = os.getenv("POSTGRES_DB", "vehicle_platform")
    USER = os.getenv("POSTGRES_USER", "postgres")
    PASSWORD = os.getenv("POSTGRES_PASSWORD")

    TABLE = os.getenv(
        "POSTGRES_TABLE",
        "vehicle_metrics",
    )

    URL = f"jdbc:postgresql://" f"{HOST}:{PORT}/{DATABASE}"

    PROPERTIES = {
        "user": USER,
        "password": PASSWORD,
        "driver": "org.postgresql.Driver",
    }
