from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class PipelineExecutionConfig:
    """
    Configuration describing where a pipeline is executed.

    This is intentionally separate from PipelineRuntimeConfig.

    PipelineRuntimeConfig controls pipeline behavior such as retries,
    validation, checkpoints, metrics, and output mode.

    PipelineExecutionConfig controls the execution environment.
    """

    runtime: str = "ecs"

    SUPPORTED_RUNTIMES = frozenset({"ecs", "glue"})

    def __post_init__(self) -> None:
        runtime = self.runtime.strip().lower()

        if not runtime:
            raise ValueError("Pipeline execution runtime cannot be empty.")

        if runtime not in self.SUPPORTED_RUNTIMES:
            supported = ", ".join(sorted(self.SUPPORTED_RUNTIMES))
            raise ValueError(
                f"Unsupported pipeline execution runtime '{self.runtime}'. "
                f"Supported runtimes: {supported}."
            )

        object.__setattr__(self, "runtime", runtime)
