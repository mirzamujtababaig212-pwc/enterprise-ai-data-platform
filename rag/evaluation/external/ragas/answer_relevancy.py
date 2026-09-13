from __future__ import annotations

from typing import Protocol, Sequence

from rag.evaluation.external.models import (
    ExternalEvaluationResult,
    ExternalEvaluationSample,
)


class RagasAnswerRelevancyMetric(Protocol):
    """Minimal RAGAS AnswerRelevancy interface required by the adapter."""

    name: str

    async def ascore(
        self,
        user_input: str,
        response: str,
    ) -> object: ...


class RagasAnswerRelevancyAdapter:
    """Adapt Deldai evaluation samples to RAGAS AnswerRelevancy."""

    provider = "ragas"
    evaluator = "answer_relevancy"

    def __init__(
        self,
        metric: RagasAnswerRelevancyMetric | None = None,
        *,
        llm: object | None = None,
        embeddings: object | None = None,
    ) -> None:
        if metric is not None and (llm is not None or embeddings is not None):
            raise ValueError("metric must not be provided with llm or embeddings")

        if (llm is None) != (embeddings is None):
            raise ValueError("llm and embeddings must be provided together")

        self._metric = metric
        self._llm = llm
        self._embeddings = embeddings

    def _get_metric(self) -> RagasAnswerRelevancyMetric:
        if self._metric is not None:
            return self._metric

        try:
            from ragas.metrics.collections import AnswerRelevancy
        except ImportError as exc:
            raise ImportError(
                "RAGAS is required for the RAGAS evaluation adapter. "
                'Install the optional dependency with: pip install -e ".[ragas]"'
            ) from exc

        if self._llm is None or self._embeddings is None:
            raise ValueError(
                "An LLM and embeddings are required for RAGAS "
                "AnswerRelevancy evaluation. Provide a metric for testing "
                "or configure RAGAS-compatible LLM and embeddings."
            )

        self._metric = AnswerRelevancy(
            llm=self._llm,
            embeddings=self._embeddings,
        )

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
            )

            score = getattr(score_result, "value", score_result)

            if not isinstance(score, (int, float)):
                raise TypeError(
                    "RAGAS answer relevancy metric returned a non-numeric "
                    f"score: {type(score).__name__}"
                )

            scores.append(float(score))

        mean_score = sum(scores) / len(scores)

        return ExternalEvaluationResult(
            provider=self.provider,
            evaluator=self.evaluator,
            metrics={
                "answer_relevancy": mean_score,
            },
            evaluated_samples=len(scores),
        )
