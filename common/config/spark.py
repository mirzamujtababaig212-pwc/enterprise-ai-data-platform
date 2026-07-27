import os

from dotenv import load_dotenv

load_dotenv()

class SparkConfig:
    CHECKPOINT_DIR = os.getenv(
        "SPARK_CHECKPOINT_DIR"
    )
