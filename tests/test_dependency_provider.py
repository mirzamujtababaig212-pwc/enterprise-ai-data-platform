from unittest.mock import patch

from common.config.settings import Settings
from common.dependency_provider import DependencyProvider


class TestDependencyProvider:

    # =========================================================
    # Readers
    # =========================================================

    def test_bronze_reader(self):
        reader = DependencyProvider.bronze_reader()
        assert reader is not None

    def test_silver_batch_reader(self):
        reader = DependencyProvider.silver_batch_reader()
        assert reader is not None

    def test_silver_stream_reader(self):
        reader = DependencyProvider.silver_stream_reader()
        assert reader is not None

    def test_gold_reader(self):
        reader = DependencyProvider.gold_reader()
        assert reader is not None

    # =========================================================
    # Writers
    # =========================================================

    @patch("common.dependency_provider.DeltaWriter")
    def test_bronze_writer(self, mock_delta_writer):
        DependencyProvider.bronze_writer()

        mock_delta_writer.assert_called_once()

        kwargs = mock_delta_writer.call_args.kwargs

        assert kwargs["table"]
        assert kwargs["path"]
        assert kwargs["checkpoint"]
        assert kwargs["mode"] == "append"

    @patch("common.dependency_provider.DeltaWriter")
    def test_silver_writer(self, mock_delta_writer):
        DependencyProvider.silver_writer()

        mock_delta_writer.assert_called_once()

        kwargs = mock_delta_writer.call_args.kwargs

        assert kwargs["table"]
        assert kwargs["path"]
        assert kwargs["checkpoint"]
        assert kwargs["mode"] == "append"

    @patch("common.dependency_provider.PostgresWriter")
    def test_gold_writer(self, mock_postgres_writer):
        DependencyProvider.gold_writer()

        mock_postgres_writer.assert_called_once_with(
            url=Settings.postgres.URL,
            table=Settings.postgres.TABLE,
            properties=Settings.postgres.PROPERTIES,
        )

    # =========================================================
    # Validators
    # =========================================================

    def test_bronze_validator(self):
        validator = DependencyProvider.bronze_validator()
        assert validator is not None

    def test_silver_validator(self):
        validator = DependencyProvider.silver_validator()
        assert validator is not None

    def test_gold_validator(self):
        validator = DependencyProvider.gold_validator()
        assert validator is not None

    # =========================================================
    # Transformers
    # =========================================================

    def test_bronze_transformer(self):
        transformer = DependencyProvider.bronze_transformer()
        assert transformer is not None

    def test_silver_transformer(self):
        transformer = DependencyProvider.silver_transformer()
        assert transformer is not None

    def test_gold_transformer(self):
        transformer = DependencyProvider.gold_transformer()
        assert transformer is not None

    # =========================================================
    # Metrics
    # =========================================================

    def test_metrics(self):
        metrics = DependencyProvider.metrics()
        assert metrics is not None

    # =========================================================
    # DLQ
    # =========================================================

    def test_bronze_dlq(self):
        dlq = DependencyProvider.bronze_dlq()
        assert dlq is not None

    def test_silver_dlq(self):
        dlq = DependencyProvider.silver_dlq()
        assert dlq is not None

    def test_gold_dlq(self):
        dlq = DependencyProvider.gold_dlq()
        assert dlq is not None
