from unittest.mock import MagicMock, patch

from common.spark.spark_builder import SparkSessionBuilder


class FakeBuilder:
    def __init__(self):
        self._options = {}
        self.config_calls = []
        self._spark = MagicMock()

    def appName(self, value):
        return self

    def master(self, value):
        return self

    def config(self, key, value):
        self._options[key] = value
        self.config_calls.append((key, value))
        return self

    def getOrCreate(self):
        self.get_or_create_calls = getattr(self, "get_or_create_calls", 0) + 1
        return self._spark


@patch("common.spark.spark_builder.configure_spark_with_delta_pip")
@patch("common.spark.spark_builder.SparkSession")
def test_build_configures_s3a_dependencies_without_kafka_by_default(
    mock_spark,
    mock_delta,
    tmp_path,
    monkeypatch,
):
    monkeypatch.setenv("ENTERPRISE_AI_PLATFORM_ROOT", str(tmp_path))

    builder = FakeBuilder()
    mock_spark.builder = builder

    delta_configured_builder = FakeBuilder()
    delta_configured_builder._options = {
        "spark.jars.packages": SparkSessionBuilder.DELTA_PACKAGE,
    }
    mock_delta.return_value = delta_configured_builder

    result = SparkSessionBuilder.build("S3ARegressionTest")

    expected_packages = (
        f"{SparkSessionBuilder.DELTA_PACKAGE}," f"{SparkSessionBuilder.HADOOP_AWS_PACKAGE}"
    )

    assert delta_configured_builder._options["spark.jars.packages"] == expected_packages
    assert SparkSessionBuilder.KAFKA_PACKAGE not in expected_packages

    assert "spark.hadoop.fs.s3a.aws.credentials.provider" in delta_configured_builder._options

    assert delta_configured_builder._options["spark.hadoop.fs.s3a.aws.credentials.provider"] == (
        "org.apache.hadoop.fs.s3a.auth.IAMInstanceCredentialsProvider,"
        "software.amazon.awssdk.auth.credentials.ProfileCredentialsProvider"
    )

    package_calls = [
        value
        for key, value in delta_configured_builder.config_calls
        if key == "spark.jars.packages"
    ]

    assert package_calls == [expected_packages]

    mock_delta.assert_called_once_with(builder)
    assert result == delta_configured_builder._spark
    assert delta_configured_builder.get_or_create_calls == 1
    delta_configured_builder._spark.sparkContext.setLogLevel.assert_called_once_with("WARN")

    assert delta_configured_builder._options["spark.hadoop.fs.s3a.impl"] == (
        "org.apache.hadoop.fs.s3a.S3AFileSystem"
    )


@patch("common.spark.spark_builder.configure_spark_with_delta_pip")
@patch("common.spark.spark_builder.SparkSession")
def test_build_adds_kafka_when_explicitly_enabled(
    mock_spark,
    mock_delta,
    tmp_path,
    monkeypatch,
):
    monkeypatch.setenv("ENTERPRISE_AI_PLATFORM_ROOT", str(tmp_path))

    builder = FakeBuilder()
    mock_spark.builder = builder

    delta_configured_builder = FakeBuilder()
    delta_configured_builder._options = {
        "spark.jars.packages": SparkSessionBuilder.DELTA_PACKAGE,
    }
    mock_delta.return_value = delta_configured_builder

    SparkSessionBuilder.build(
        "KafkaRegressionTest",
        include_kafka=True,
    )

    expected_base_packages = (
        f"{SparkSessionBuilder.DELTA_PACKAGE}," f"{SparkSessionBuilder.HADOOP_AWS_PACKAGE}"
    )
    expected_packages = f"{expected_base_packages},{SparkSessionBuilder.KAFKA_PACKAGE}"

    assert delta_configured_builder._options["spark.jars.packages"] == expected_packages

    assert delta_configured_builder._options["spark.hadoop.fs.s3a.impl"] == (
        "org.apache.hadoop.fs.s3a.S3AFileSystem"
    )

    assert delta_configured_builder._options["spark.hadoop.fs.s3a.aws.credentials.provider"] == (
        "org.apache.hadoop.fs.s3a.auth.IAMInstanceCredentialsProvider,"
        "software.amazon.awssdk.auth.credentials.ProfileCredentialsProvider"
    )

    package_calls = [
        value
        for key, value in delta_configured_builder.config_calls
        if key == "spark.jars.packages"
    ]

    assert package_calls == [
        expected_base_packages,
        expected_packages,
    ]

    mock_delta.assert_called_once_with(builder)
    assert delta_configured_builder.get_or_create_calls == 1
    delta_configured_builder._spark.sparkContext.setLogLevel.assert_called_once_with("WARN")


@patch("common.spark.spark_builder.configure_spark_with_delta_pip")
@patch("common.spark.spark_builder.SparkSession")
def test_build_isolates_dev_warehouse_and_metastore(
    mock_spark,
    mock_delta,
    tmp_path,
    monkeypatch,
):
    monkeypatch.setenv("ENTERPRISE_AI_PLATFORM_ROOT", str(tmp_path))
    monkeypatch.setenv("APP_ENV", "dev")

    builder = FakeBuilder()
    mock_spark.builder = builder

    delta_configured_builder = FakeBuilder()
    delta_configured_builder._options = {
        "spark.jars.packages": SparkSessionBuilder.DELTA_PACKAGE,
    }
    mock_delta.return_value = delta_configured_builder

    SparkSessionBuilder.build("DevCatalogIsolationTest")

    assert builder._options["spark.sql.warehouse.dir"] == str(tmp_path / "spark-warehouse" / "dev")
    assert builder._options["spark.hadoop.javax.jdo.option.ConnectionURL"] == (
        f"jdbc:derby:;databaseName={tmp_path / 'metastore_db' / 'dev'};create=true"
    )


@patch("common.spark.spark_builder.configure_spark_with_delta_pip")
@patch("common.spark.spark_builder.SparkSession")
def test_build_isolates_aws_warehouse_and_metastore(
    mock_spark,
    mock_delta,
    tmp_path,
    monkeypatch,
):
    monkeypatch.setenv("ENTERPRISE_AI_PLATFORM_ROOT", str(tmp_path))
    monkeypatch.setenv("APP_ENV", "aws")

    builder = FakeBuilder()
    mock_spark.builder = builder

    delta_configured_builder = FakeBuilder()
    delta_configured_builder._options = {
        "spark.jars.packages": SparkSessionBuilder.DELTA_PACKAGE,
    }
    mock_delta.return_value = delta_configured_builder

    SparkSessionBuilder.build("AwsCatalogIsolationTest")

    assert builder._options["spark.sql.warehouse.dir"] == str(tmp_path / "spark-warehouse" / "aws")
    assert builder._options["spark.hadoop.javax.jdo.option.ConnectionURL"] == (
        f"jdbc:derby:;databaseName={tmp_path / 'metastore_db' / 'aws'};create=true"
    )
