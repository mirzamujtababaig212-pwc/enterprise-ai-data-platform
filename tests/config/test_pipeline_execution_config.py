import pytest

from common.config.pipeline_execution_config import PipelineExecutionConfig


def test_pipeline_execution_defaults_to_ecs() -> None:
    config = PipelineExecutionConfig()

    assert config.runtime == "ecs"


def test_pipeline_execution_accepts_glue() -> None:
    config = PipelineExecutionConfig(runtime="glue")

    assert config.runtime == "glue"


@pytest.mark.parametrize(
    "runtime, expected",
    [
        ("ECS", "ecs"),
        (" Glue ", "glue"),
    ],
)
def test_pipeline_execution_normalizes_runtime(
    runtime: str,
    expected: str,
) -> None:
    config = PipelineExecutionConfig(runtime=runtime)

    assert config.runtime == expected


@pytest.mark.parametrize(
    "runtime",
    [
        "",
        "   ",
    ],
)
def test_pipeline_execution_rejects_empty_runtime(
    runtime: str,
) -> None:
    with pytest.raises(
        ValueError,
        match="Pipeline execution runtime cannot be empty",
    ):
        PipelineExecutionConfig(runtime=runtime)


@pytest.mark.parametrize(
    "runtime",
    [
        "spark",
        "batch",
        "docker",
    ],
)
def test_pipeline_execution_rejects_unsupported_runtime(
    runtime: str,
) -> None:
    with pytest.raises(
        ValueError,
        match="Unsupported pipeline execution runtime",
    ):
        PipelineExecutionConfig(runtime=runtime)


def test_pipeline_execution_is_immutable() -> None:
    config = PipelineExecutionConfig(runtime="glue")

    with pytest.raises(AttributeError):
        config.runtime = "ecs"
