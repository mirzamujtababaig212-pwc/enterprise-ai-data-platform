from __future__ import annotations

import pytest

from rag.evaluation.external import ExternalEvaluationSample
from rag.evaluation.external.ragas import RagasFaithfulnessAdapter


class FakeMetricResult:
    def __init__(self, value: float) -> None:
        self.value = value


class FakeFaithfulnessMetric:
    name = "faithfulness"

    def __init__(self, scores: list[float]) -> None:
        self._scores = iter(scores)
        self.calls: list[dict[str, object]] = []

    async def ascore(
        self,
        *,
        user_input: str,
        response: str,
        retrieved_contexts: list[str],
    ) -> FakeMetricResult:
        self.calls.append(
            {
                "user_input": user_input,
                "response": response,
                "retrieved_contexts": retrieved_contexts,
            }
        )
        return FakeMetricResult(next(self._scores))


def make_sample(
    *,
    query: str = "What powers an electric vehicle?",
    response: str = ("An electric vehicle is powered by electricity stored in a battery."),
) -> ExternalEvaluationSample:
    return ExternalEvaluationSample(
        query=query,
        retrieved_contexts=("Electric vehicles are powered by electricity stored in a battery.",),
        response=response,
    )


def test_external_evaluation_sample_validation() -> None:
    sample = make_sample()

    assert sample.query == "What powers an electric vehicle?"
    assert len(sample.retrieved_contexts) == 1
    assert sample.reference is None


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        (
            {"query": ""},
            "query must not be empty",
        ),
        (
            {"retrieved_contexts": ()},
            "retrieved_contexts must not be empty",
        ),
        (
            {"retrieved_contexts": ("",)},
            "retrieved_contexts must not contain empty values",
        ),
        (
            {"response": ""},
            "response must not be empty",
        ),
        (
            {"reference": ""},
            "reference must not be empty when provided",
        ),
    ],
)
def test_external_evaluation_sample_rejects_invalid_values(
    kwargs: dict[str, object],
    message: str,
) -> None:
    values: dict[str, object] = {
        "query": "question",
        "retrieved_contexts": ("context",),
        "response": "answer",
    }
    values.update(kwargs)

    with pytest.raises(ValueError, match=message):
        ExternalEvaluationSample(**values)


@pytest.mark.asyncio
async def test_ragas_adapter_aggregates_faithfulness_scores() -> None:
    metric = FakeFaithfulnessMetric([1.0, 0.5])
    adapter = RagasFaithfulnessAdapter(metric=metric)

    result = await adapter.evaluate(
        [
            make_sample(),
            make_sample(query="What stores the vehicle's energy?"),
        ]
    )

    assert result.provider == "ragas"
    assert result.evaluator == "faithfulness"
    assert result.evaluated_samples == 2
    assert result.metrics == {"faithfulness": 0.75}
    assert len(metric.calls) == 2


@pytest.mark.asyncio
async def test_ragas_adapter_maps_deldai_sample_to_metric_arguments() -> None:
    metric = FakeFaithfulnessMetric([1.0])
    adapter = RagasFaithfulnessAdapter(metric=metric)

    sample = ExternalEvaluationSample(
        query="What powers an electric vehicle?",
        retrieved_contexts=("Battery packs store electrical energy.",),
        response="A battery pack stores electrical energy.",
        reference="Electric vehicles use battery packs.",
    )

    await adapter.evaluate([sample])

    assert metric.calls == [
        {
            "user_input": sample.query,
            "response": sample.response,
            "retrieved_contexts": list(sample.retrieved_contexts),
        }
    ]


@pytest.mark.asyncio
async def test_ragas_adapter_requires_samples() -> None:
    adapter = RagasFaithfulnessAdapter(metric=FakeFaithfulnessMetric([]))

    with pytest.raises(ValueError, match="samples must not be empty"):
        await adapter.evaluate([])


@pytest.mark.asyncio
async def test_ragas_adapter_rejects_non_numeric_metric_result() -> None:
    class InvalidMetric:
        name = "faithfulness"

        async def ascore(
            self,
            *,
            user_input: str,
            response: str,
            retrieved_contexts: list[str],
        ) -> str:
            return "invalid"

    adapter = RagasFaithfulnessAdapter(metric=InvalidMetric())

    with pytest.raises(
        TypeError,
        match="RAGAS faithfulness metric returned a non-numeric score",
    ):
        await adapter.evaluate([make_sample()])


def test_ragas_adapter_rejects_metric_and_llm_together() -> None:
    metric = FakeFaithfulnessMetric([1.0])

    with pytest.raises(
        ValueError,
        match="metric and llm must not be provided together",
    ):
        RagasFaithfulnessAdapter(metric=metric, llm=object())


def test_ragas_adapter_requires_llm_for_real_metric() -> None:
    adapter = RagasFaithfulnessAdapter()

    with pytest.raises(
        ValueError,
        match="An LLM is required for RAGAS Faithfulness evaluation",
    ):
        adapter._get_metric()


@pytest.mark.asyncio
async def test_ragas_adapter_reuses_real_metric_instance(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    created: list[object] = []

    class FakeRealMetric(FakeFaithfulnessMetric):
        def __init__(self) -> None:
            super().__init__([1.0, 1.0])

    fake_metric = FakeRealMetric()

    def fake_faithfulness(*, llm: object) -> FakeRealMetric:
        created.append(llm)
        return fake_metric

    monkeypatch.setattr(
        "ragas.metrics.collections.Faithfulness",
        fake_faithfulness,
    )

    adapter = RagasFaithfulnessAdapter(llm=object())

    first = adapter._get_metric()
    second = adapter._get_metric()

    assert first is second
    assert len(created) == 1
