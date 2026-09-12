from __future__ import annotations

import pytest
from pydantic import BaseModel

from rag.evaluation.external.ragas.gateway_llm import GatewayRagasLLM


class OutputModel(BaseModel):
    statements: list[str]


class FakeGateway:
    def __init__(self, response: dict[str, object]) -> None:
        self.response = response
        self.calls: list[dict[str, object]] = []

    async def generate(
        self,
        prompt: str,
        *,
        temperature: float,
        max_tokens: int,
        user_id: str | None,
    ) -> dict[str, object]:
        self.calls.append(
            {
                "prompt": prompt,
                "temperature": temperature,
                "max_tokens": max_tokens,
                "user_id": user_id,
            }
        )
        return self.response


@pytest.mark.asyncio
async def test_gateway_ragas_llm_maps_gateway_reply_to_response_model() -> None:
    gateway = FakeGateway(
        {
            "reply": '{"statements":["Electric vehicles use battery power."]}',
            "metrics": {},
        }
    )

    llm = GatewayRagasLLM(
        gateway,  # type: ignore[arg-type]
        temperature=0.0,
        max_tokens=4096,
        user_id="evaluation-user",
    )

    result = await llm.agenerate(
        "Return the statements.",
        OutputModel,
    )

    assert result.statements == ["Electric vehicles use battery power."]

    assert gateway.calls == [
        {
            "prompt": "Return the statements.",
            "temperature": 0.0,
            "max_tokens": 4096,
            "user_id": "evaluation-user",
        }
    ]


@pytest.mark.asyncio
async def test_gateway_ragas_llm_rejects_empty_reply() -> None:
    gateway = FakeGateway(
        {
            "reply": "",
            "metrics": {},
        }
    )

    llm = GatewayRagasLLM(gateway)  # type: ignore[arg-type]

    with pytest.raises(ValueError, match="non-empty string 'reply'"):
        await llm.agenerate("Return JSON.", OutputModel)


@pytest.mark.asyncio
async def test_gateway_ragas_llm_rejects_invalid_structured_output() -> None:
    gateway = FakeGateway(
        {
            "reply": '{"invalid": "schema"}',
            "metrics": {},
        }
    )

    llm = GatewayRagasLLM(gateway)  # type: ignore[arg-type]

    with pytest.raises(ValueError):
        await llm.agenerate("Return JSON.", OutputModel)


@pytest.mark.asyncio
async def test_gateway_ragas_llm_rejects_empty_prompt() -> None:
    gateway = FakeGateway(
        {
            "reply": '{"statements":["valid"]}',
            "metrics": {},
        }
    )

    llm = GatewayRagasLLM(gateway)  # type: ignore[arg-type]

    with pytest.raises(ValueError, match="prompt must not be empty"):
        await llm.agenerate("", OutputModel)


def test_gateway_ragas_llm_is_async() -> None:
    gateway = FakeGateway({"reply": "{}", "metrics": {}})

    llm = GatewayRagasLLM(gateway)  # type: ignore[arg-type]

    assert llm.is_async is True


def test_gateway_ragas_llm_sync_path_is_rejected() -> None:
    gateway = FakeGateway({"reply": "{}", "metrics": {}})

    llm = GatewayRagasLLM(gateway)  # type: ignore[arg-type]

    with pytest.raises(TypeError, match="asynchronous"):
        llm.generate("Return JSON.", OutputModel)


class FaithfulnessGateway:
    def __init__(self) -> None:
        self.calls: list[str] = []

    async def generate(
        self,
        prompt: str,
        *,
        temperature: float,
        max_tokens: int,
        user_id: str | None,
    ) -> dict[str, object]:
        self.calls.append(prompt)

        if "Break down each sentence" in prompt:
            return {
                "reply": (
                    '{"statements":'
                    '["An electric vehicle is powered by electricity stored in a battery pack."]}'
                ),
                "metrics": {},
            }

        if "judge the faithfulness" in prompt:
            return {
                "reply": (
                    '{"statements":['
                    '{"statement":"An electric vehicle is powered by electricity stored in a battery pack.",'
                    '"reason":"The context directly states this.",'
                    '"verdict":1'
                    "}]}"
                ),
                "metrics": {},
            }

        raise AssertionError("Unexpected RAGAS prompt")


@pytest.mark.asyncio
async def test_ragas_faithfulness_uses_gateway_llm() -> None:
    from rag.evaluation.external.models import ExternalEvaluationSample
    from rag.evaluation.external.ragas.adapter import RagasFaithfulnessAdapter
    from rag.evaluation.external.ragas.gateway_llm import GatewayRagasLLM

    gateway = FaithfulnessGateway()
    llm = GatewayRagasLLM(gateway)  # type: ignore[arg-type]

    adapter = RagasFaithfulnessAdapter(llm=llm)

    result = await adapter.evaluate(
        [
            ExternalEvaluationSample(
                query="What powers an electric vehicle?",
                retrieved_contexts=(
                    "Electric vehicles are powered by electricity stored in a battery pack.",
                ),
                response=(
                    "An electric vehicle is powered by electricity stored " "in a battery pack."
                ),
            )
        ]
    )

    assert result.provider == "ragas"
    assert result.evaluator == "faithfulness"
    assert result.evaluated_samples == 1
    assert result.metrics["faithfulness"] == 1.0

    assert len(gateway.calls) == 2


@pytest.mark.asyncio
async def test_rag_generation_workflow_uses_gateway_backed_ragas_faithfulness(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from unittest.mock import AsyncMock, MagicMock

    from rag.evaluation.external.ragas.adapter import RagasFaithfulnessAdapter
    from rag.evaluation.external.ragas.gateway_llm import GatewayRagasLLM
    from rag.evaluation.external.workflow import RAGGenerationEvaluationWorkflow
    from rag.evaluation.models import RetrievalEvaluationCase

    gateway = FaithfulnessGateway()
    llm = GatewayRagasLLM(gateway)  # type: ignore[arg-type]
    evaluator = RagasFaithfulnessAdapter(llm=llm)

    retriever = MagicMock()

    rag_query_service = MagicMock()
    rag_query_service.query = AsyncMock(
        return_value=MagicMock(
            answer=("An electric vehicle is powered by electricity stored " "in a battery pack."),
            sources=(
                MagicMock(
                    content=(
                        "Electric vehicles are powered by electricity " "stored in a battery pack."
                    )
                ),
            ),
        )
    )

    monkeypatch.setattr(
        "rag.evaluation.external.workflow.RAGQueryService",
        MagicMock(return_value=rag_query_service),
    )

    workflow = RAGGenerationEvaluationWorkflow(
        chat_service=MagicMock(),
        evaluator=evaluator,
    )

    result = await workflow.evaluate(
        (
            RetrievalEvaluationCase(
                query="What powers an electric vehicle?",
                relevant_chunk_ids=("vehicle-electric-powertrain",),
            ),
        ),
        retriever=retriever,
    )

    assert result.provider == "ragas"
    assert result.evaluator == "faithfulness"
    assert result.evaluated_samples == 1
    assert result.metrics["faithfulness"] == 1.0

    rag_query_service.query.assert_awaited_once_with(
        "What powers an electric vehicle?",
        top_k=5,
        min_score=None,
    )

    assert len(gateway.calls) == 2
