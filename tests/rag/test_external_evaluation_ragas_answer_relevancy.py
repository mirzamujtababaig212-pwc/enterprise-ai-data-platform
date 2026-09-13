from __future__ import annotations

import pytest

from rag.evaluation.external.models import ExternalEvaluationSample
from rag.evaluation.external.ragas import RagasAnswerRelevancyAdapter


class FakeAnswerRelevancyMetric:
    name = "answer_relevancy"

    def __init__(self, scores: list[float | object]) -> None:
        self.scores = scores
        self.calls: list[tuple[str, str]] = []

    async def ascore(self, user_input: str, response: str) -> object:
        self.calls.append((user_input, response))
        return self.scores[len(self.calls) - 1]


def sample(
    query: str = "What is DELDAI?",
    response: str = "An enterprise AI platform.",
) -> ExternalEvaluationSample:
    return ExternalEvaluationSample(
        query=query,
        retrieved_contexts=("DELDAI is an enterprise AI platform.",),
        response=response,
    )


@pytest.mark.asyncio
async def test_answer_relevancy_maps_query_and_response() -> None:
    metric = FakeAnswerRelevancyMetric([0.8])

    adapter = RagasAnswerRelevancyAdapter(metric=metric)

    result = await adapter.evaluate([sample()])

    assert metric.calls == [("What is DELDAI?", "An enterprise AI platform.")]
    assert result.provider == "ragas"
    assert result.evaluator == "answer_relevancy"
    assert result.metrics == {"answer_relevancy": 0.8}
    assert result.evaluated_samples == 1


@pytest.mark.asyncio
async def test_answer_relevancy_aggregates_scores() -> None:
    metric = FakeAnswerRelevancyMetric([0.8, 0.6])

    adapter = RagasAnswerRelevancyAdapter(metric=metric)

    result = await adapter.evaluate(
        [
            sample("Question one", "Answer one"),
            sample("Question two", "Answer two"),
        ]
    )

    assert result.metrics == {"answer_relevancy": 0.7}
    assert result.evaluated_samples == 2


@pytest.mark.asyncio
async def test_answer_relevancy_rejects_empty_samples() -> None:
    adapter = RagasAnswerRelevancyAdapter(metric=FakeAnswerRelevancyMetric([]))

    with pytest.raises(ValueError, match="samples must not be empty"):
        await adapter.evaluate([])


@pytest.mark.asyncio
async def test_answer_relevancy_rejects_nonnumeric_score() -> None:
    metric = FakeAnswerRelevancyMetric(["not-a-score"])
    adapter = RagasAnswerRelevancyAdapter(metric=metric)

    with pytest.raises(
        TypeError,
        match="RAGAS answer relevancy metric returned a non-numeric score",
    ):
        await adapter.evaluate([sample()])


def test_answer_relevancy_rejects_metric_with_llm() -> None:
    metric = FakeAnswerRelevancyMetric([0.8])

    with pytest.raises(
        ValueError,
        match="metric must not be provided with llm or embeddings",
    ):
        RagasAnswerRelevancyAdapter(
            metric=metric,
            llm=object(),
        )


def test_answer_relevancy_requires_llm_and_embeddings_together() -> None:
    with pytest.raises(
        ValueError,
        match="llm and embeddings must be provided together",
    ):
        RagasAnswerRelevancyAdapter(llm=object())


def test_answer_relevancy_requires_embeddings_with_llm() -> None:
    with pytest.raises(
        ValueError,
        match="llm and embeddings must be provided together",
    ):
        RagasAnswerRelevancyAdapter(embeddings=object())


@pytest.mark.asyncio
async def test_answer_relevancy_lazily_constructs_and_caches_metric(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    created: list[tuple[object, object]] = []

    class FakeRealMetric(FakeAnswerRelevancyMetric):
        def __init__(self, llm: object, embeddings: object) -> None:
            super().__init__([0.9])
            created.append((llm, embeddings))

    def fake_answer_relevancy(
        *,
        llm: object,
        embeddings: object,
    ) -> FakeRealMetric:
        return FakeRealMetric(llm, embeddings)

    monkeypatch.setattr(
        "ragas.metrics.collections.AnswerRelevancy",
        fake_answer_relevancy,
    )

    llm = object()
    embeddings = object()

    adapter = RagasAnswerRelevancyAdapter(
        llm=llm,
        embeddings=embeddings,
    )

    first = adapter._get_metric()
    second = adapter._get_metric()

    assert first is second
    assert created == [(llm, embeddings)]
