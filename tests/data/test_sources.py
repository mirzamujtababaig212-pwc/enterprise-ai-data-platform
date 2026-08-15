from ai_platform.data.contracts import DataSource
from ai_platform.data.sources import DataSourceConfig


def test_data_source_config_implements_protocol() -> None:
    source = DataSourceConfig(
        name="customer_csv",
        source_type="local_file",
        config={
            "path": "/tmp/customers.csv",
            "format": "csv",
        },
    )

    assert isinstance(source, DataSource)

    assert source.name == "customer_csv"
    assert source.source_type == "local_file"

    assert source.get_config() == {
        "path": "/tmp/customers.csv",
        "format": "csv",
    }


def test_data_source_config_returns_copy() -> None:
    source = DataSourceConfig(
        name="customer_csv",
        source_type="local_file",
        config={
            "path": "/tmp/customers.csv",
        },
    )

    config = source.get_config()

    config["path"] = "/tmp/modified.csv"

    assert source.get_config() == {
        "path": "/tmp/customers.csv",
    }


def test_data_source_config_is_immutable() -> None:
    source = DataSourceConfig(
        name="customer_csv",
        source_type="local_file",
    )

    try:
        source.name = "changed"
    except AttributeError:
        pass
    else:
        raise AssertionError("DataSourceConfig should be immutable.")
