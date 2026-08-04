from common.config.settings import Settings
from common.exceptions.database import DatabaseException
from common.logging.logger import get_logger

logger = get_logger(__name__)


class PostgresWriter:
    @staticmethod
    def write_table(df, table_name, mode="append"):
        try:
            logger.info(f"Writing {table_name}")
            (
                df.write.format("jdbc")
                .option(
                    "url",
                    f"jdbc:postgresql://"
                    f"{Settings.postgres.HOST}:"
                    f"{Settings.postgres.PORT}/"
                    f"{Settings.postgres.DATABASE}",
                )
                .option("dbtable", table_name)
                .option("user", Settings.postgres.USER)
                .option("password", Settings.postgres.PASSWORD)
                .mode(mode)
                .save()
            )
            logger.info("Write successful.")
        except Exception as ex:
            logger.error(str(ex))
            raise DatabaseException(str(ex)) from ex
