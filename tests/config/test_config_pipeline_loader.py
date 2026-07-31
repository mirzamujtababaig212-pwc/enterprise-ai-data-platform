from unittest.mock import mock_open, patch

from common.config.pipeline_loader import PipelineLoader


@patch("common.config.pipeline_loader.yaml.safe_load")
@patch("builtins.open", new_callable=mock_open, read_data="pipeline: bronze")
def test_pipeline_loader(mock_file, mock_yaml):
    mock_yaml.return_value = {"pipeline": "bronze"}
    result = PipelineLoader.load("bronze")
    mock_file.assert_called_once_with("config/pipelines/bronze.yaml")
    mock_yaml.assert_called_once()
    assert result["pipeline"] == "bronze"
