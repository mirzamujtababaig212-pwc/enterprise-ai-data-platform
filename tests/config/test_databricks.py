from common.config.databricks import DatabricksConfig


def test_databricks_config_options_excludes_empty_values(monkeypatch):
    monkeypatch.setattr(
        DatabricksConfig,
        "HOST",
        "https://workspace.example.com",
    )
    monkeypatch.setattr(
        DatabricksConfig,
        "TOKEN",
        "",
    )

    assert DatabricksConfig.options() == {
        "host": "https://workspace.example.com",
    }


def test_databricks_config_options_returns_configured_values(monkeypatch):
    monkeypatch.setattr(
        DatabricksConfig,
        "HOST",
        "https://workspace.example.com",
    )
    monkeypatch.setattr(
        DatabricksConfig,
        "TOKEN",
        "test-token",
    )

    assert DatabricksConfig.options() == {
        "host": "https://workspace.example.com",
        "token": "test-token",
    }
