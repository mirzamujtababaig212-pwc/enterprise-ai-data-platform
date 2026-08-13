from common.factories.pipeline_factory import PipelineFactory
from common.spark.spark_builder import SparkSessionBuilder


def main():
    spark = SparkSessionBuilder.build("VerifySilverPipelineConfig")

    try:
        pipeline = PipelineFactory.get_pipeline(
            "silver",
            spark,
        )

        print("=" * 80)
        print("PIPELINE")
        print("=" * 80)
        print(type(pipeline).__name__)

        print()
        print("=" * 80)
        print("READER")
        print("=" * 80)
        print(type(pipeline.reader).__name__)
        print(vars(pipeline.reader))

        print()
        print("=" * 80)
        print("WRITER")
        print("=" * 80)
        print(type(pipeline.writer).__name__)
        print(vars(pipeline.writer))

        print()
        print("=" * 80)
        print("TRANSFORMER")
        print("=" * 80)
        print(type(pipeline.transformer).__name__)

        print()
        print("=" * 80)
        print("VALIDATOR")
        print("=" * 80)
        print(type(pipeline.validator).__name__)

        print()
        print("=" * 80)
        print("CONFIGURATION VALIDATION")
        print("=" * 80)

        assert pipeline.writer.table == "silver.vehicle_events"

        assert pipeline.writer.path == ("/home/annie/enterprise_ai_platform/" "data/silver_delta")

        assert pipeline.writer.checkpoint == (
            "/home/annie/enterprise_ai_platform/" "spark/checkpoints/silver_delta"
        )

        print("Writer configuration: OK")

    finally:
        spark.stop()


if __name__ == "__main__":
    main()
