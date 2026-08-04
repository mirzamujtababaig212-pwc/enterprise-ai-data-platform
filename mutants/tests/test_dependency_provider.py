from unittest.mock import patch

from common.config.settings import Settings
from common.dependency_provider import DependencyProvider


class TestDependencyProvider:

    #############################
    # Readers
    #############################
    @patch("common.dependency_provider.KafkaReader")
    def test_bronze_reader(self, mock_reader):
        DependencyProvider.bronze_reader()
        mock_reader.assert_called_once()
        args, kwargs = mock_reader.call_args
        assert len(args) == 1
        assert kwargs == {}
        assert args[0] == Settings.kafka.options

    @patch("common.dependency_provider.ParquetReader")
    def test_silver_reader(self, mock_reader):
        DependencyProvider.silver_reader()
        mock_reader.assert_called_once()
        args, kwargs = mock_reader.call_args
        assert args == ()
        assert "path" in kwargs
        assert "schema" in kwargs

    @patch("common.dependency_provider.ParquetReader")
    def test_gold_reader(self, mock_reader):
        DependencyProvider.gold_reader()
        mock_reader.assert_called_once()
        args, kwargs = mock_reader.call_args
        assert args == ()
        assert "path" in kwargs
        assert "schema" in kwargs

    #############################
    # Writers
    #############################

    @patch("common.dependency_provider.DeltaWriter")
    def test_bronze_writer(self, mock_writer):
        DependencyProvider.bronze_writer()
        mock_writer.assert_called_once()
        args, kwargs = mock_writer.call_args
        assert args == ()
        assert "table" in kwargs
        assert "checkpoint" in kwargs

    @patch("common.dependency_provider.DeltaWriter")
    def test_silver_writer(self, mock_writer):
        DependencyProvider.silver_writer()
        mock_writer.assert_called_once()
        args, kwargs = mock_writer.call_args
        assert args == ()
        assert "table" in kwargs
        assert "checkpoint" in kwargs

    @patch("common.dependency_provider.PostgresWriter")
    def test_gold_writer(self, mock_writer):
        DependencyProvider.gold_writer()
        mock_writer.assert_called_once()
        args, kwargs = mock_writer.call_args
        assert args == ()
        assert "url" in kwargs
        assert "table" in kwargs
        assert "properties" in kwargs

    #############################
    # Validators
    #############################

    @patch("common.dependency_provider.CompositeValidator")
    @patch("common.dependency_provider.DuplicateValidator")
    @patch("common.dependency_provider.NullValidator")
    @patch("common.dependency_provider.SchemaValidator")
    def test_bronze_validator(
        self,
        mock_schema,
        mock_null,
        mock_duplicate,
        mock_composite,
    ):
        DependencyProvider.bronze_validator()
        mock_schema.assert_called_once()
        mock_null.assert_called_once()
        mock_duplicate.assert_called_once()
        mock_composite.assert_called_once()

    @patch("common.dependency_provider.CompositeValidator")
    @patch("common.dependency_provider.DuplicateValidator")
    @patch("common.dependency_provider.BusinessRuleValidator")
    @patch("common.dependency_provider.SchemaValidator")
    def test_silver_validator(
        self,
        mock_schema,
        mock_business,
        mock_duplicate,
        mock_composite,
    ):
        DependencyProvider.silver_validator()
        mock_schema.assert_called_once()
        mock_business.assert_called_once()
        mock_duplicate.assert_called_once()
        mock_composite.assert_called_once()

    @patch("common.dependency_provider.NoOpValidator")
    def test_gold_validator(self, mock_validator):
        DependencyProvider.gold_validator()
        mock_validator.assert_called_once_with()

    #############################
    # Transformers
    #############################

    @patch("common.dependency_provider.BronzeTransformer")
    def test_bronze_transformer(self, mock_transformer):
        DependencyProvider.bronze_transformer()
        mock_transformer.assert_called_once_with()

    @patch("common.dependency_provider.SilverTransformer")
    def test_silver_transformer(self, mock_transformer):
        DependencyProvider.silver_transformer()
        mock_transformer.assert_called_once_with()

    @patch("common.dependency_provider.GoldTransformer")
    def test_gold_transformer(self, mock_transformer):
        DependencyProvider.gold_transformer()
        mock_transformer.assert_called_once_with()

    #############################
    # Metrics
    #############################

    @patch("common.dependency_provider.MetricsCollector")
    def test_metrics(self, mock_metrics):
        DependencyProvider.metrics()
        mock_metrics.assert_called_once_with()

    #############################
    # DLQ
    #############################

    @patch("common.dependency_provider.DeltaDLQ")
    def test_bronze_dlq(self, mock_dlq):
        DependencyProvider.bronze_dlq()
        mock_dlq.assert_called_once()
        args, kwargs = mock_dlq.call_args
        assert args == ()
        assert "table" in kwargs

    @patch("common.dependency_provider.DeltaDLQ")
    def test_silver_dlq(self, mock_dlq):
        DependencyProvider.silver_dlq()
        mock_dlq.assert_called_once()
        args, kwargs = mock_dlq.call_args
        assert args == ()
        assert "table" in kwargs

    @patch("common.dependency_provider.NoOpDLQ")
    def test_gold_dlq(self, mock_dlq):
        DependencyProvider.gold_dlq()
        mock_dlq.assert_called_once_with()
