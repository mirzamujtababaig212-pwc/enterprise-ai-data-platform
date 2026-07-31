import os

from dotenv import load_dotenv

load_dotenv()


class PostgresConfig:
    HOST = os.getenv("POSTGRES_HOST")
    PORT = os.getenv("POSTGRES_PORT")
    DATABASE = os.getenv("POSTGRES_DB")
    USER = os.getenv("POSTGRES_USER")
    PASSWORD = os.getenv("POSTGRES_PASSWORD")
    URL = f"jdbc:postgresql://{HOST}:{PORT}/{DATABASE}"
    TABLE = os.getenv("POSTGRES_TABLE", "gold")
    PROPERTIES = {"user": USER, "password": PASSWORD, "driver": "org.postgresql.Driver"}
