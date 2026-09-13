from __future__ import annotations

from typing import Protocol, Sequence

from rag.evaluation.external.models import (
    ExternalEvaluationResult,
    ExternalEvaluationSample,
)


class RagasMetric(Protocol):
    """Minimal RAGAS Faithfulness interface required by the adapter."""

    name: str

    async def ascore(
        self,
        *,
        user_input: str,
        response: str,
        retrieved_contexts: list[str],
    ) -> object: ...


class RagasFaithfulnessAdapter:
    """Adapt Deldai evaluation samples to RAGAS Faithfulness."""

    provider = "ragas"
    evaluator = "faithfulness"

    def __init__(
        self,
        metric: RagasMetric | None = None,
        *,
        llm: object | None = None,
    ) -> None:
        if metric is not None and llm is not None:
            raise ValueError("metric and llm must not be provided together")

        self._metric = metric
        self._llm = llm

    def _get_metric(self) -> RagasMetric:
        if self._metric is not None:
            return self._metric

        try:
            from ragas.metrics.collections import Faithfulness
        except ImportError as exc:
            raise ImportError(
                "RAGAS is required for the RAGAS evaluation adapter. "
                'Install the optional dependency with: pip install -e ".[ragas]"'
            ) from exc

        if self._llm is None:
            raise ValueError(
                "An LLM is required for RAGAS Faithfulness evaluation. "
                "Provide a metric for testing or configure a RAGAS-compatible LLM."
            )

        self._metric = Faithfulness(llm=self._llm)

        return self._metric

    async def evaluate(
        self,
        samples: Sequence[ExternalEvaluationSample],
    ) -> ExternalEvaluationResult:
        if not samples:
            raise ValueError("samples must not be empty")

        metric = self._get_metric()

        scores: list[float] = []

        for sample in samples:
            score_result = await metric.ascore(
                user_input=sample.query,
                response=sample.response,
                retrieved_contexts=list(sample.retrieved_contexts),
            )

            score = getattr(score_result, "value", score_result)

            if not isinstance(score, (int, float)):
                raise TypeError(
                    "RAGAS faithfulness metric returned a non-numeric score: "
                    f"{type(score).__name__}"
                )

            scores.append(float(score))

        mean_score = sum(scores) / len(scores)

        return ExternalEvaluationResult(
            provider=self.provider,
            evaluator=self.evaluator,
            metrics={
                "faithfulness": mean_score,
            },
            evaluated_samples=len(scores),
        )
