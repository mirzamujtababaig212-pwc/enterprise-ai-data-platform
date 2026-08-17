from pathlib import Path

from common.logging.logger import get_logger
from common.spark.spark_builder import SparkSessionBuilder
from common.writers.delta_writer import DeltaWriter

logger = get_logger(__name__)
PROJECT_ROOT = Path(__file__).resolve().parents[2]
INPUT_PATH = PROJECT_ROOT / "data" / "vehicle_events.csv"
BRONZE_PATH = PROJECT_ROOT / "data" / "bronze" / "vehicle_events"
BRONZE_TABLE = "bronze.vehicle_events"


def main() -> None:
    spark = SparkSessionBuilder.build("VehicleBronzeIngestion")
    try:
        logger.info("Starting Bronze ingestion")
        logger.info("Input path=%s", INPUT_PATH)
        logger.info("Bronze path=%s", BRONZE_PATH)
        if not INPUT_PATH.exists():
            raise FileNotFoundError(f"Input file does not exist: {INPUT_PATH}")
        df = spark.read.option("header", True).option("inferSchema", True).csv(str(INPUT_PATH))
        logger.info("Input schema:")
        df.printSchema()
        logger.info("Input row count=%s", df.count())
        writer = DeltaWriter(
            table=BRONZE_TABLE,
            path=str(BRONZE_PATH),
            mode="overwrite",
        )
        writer.write(df)
        logger.info("Bronze ingestion completed successfully")
    finally:
        spark.stop()


if __name__ == "__main__":
    main()
