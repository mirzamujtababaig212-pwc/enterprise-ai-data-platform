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
        structured_output: dict[str, object] | None,
    ) -> dict[str, object]:
        self.calls.append(
            {
                "prompt": prompt,
                "temperature": temperature,
                "max_tokens": max_tokens,
                "user_id": user_id,
                "structured_output": structured_output,
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
            "structured_output": {
                "name": "OutputModel",
                "schema": OutputModel.model_json_schema(),
                "strict": True,
            },
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
        structured_output: dict[str, object] | None,
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
    from rag.evaluation.lineage import RetrievalEvaluationArtifact
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

    monkeypatch.setattr(
        "rag.retrieval.factory.RAGRetrieverFactory.build_retrieval_artifact",
        MagicMock(
            return_value=RetrievalEvaluationArtifact(
                retriever_type="TestRetriever",
                vector_store_type="TestVectorStore",
            )
        ),
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


class FakeEmbeddingService:
    def __init__(self) -> None:
        self.calls: list[str] = []

    async def embed(self, text: str) -> list[float]:
        self.calls.append(text)
        return [float(len(text)), 1.0]


@pytest.mark.asyncio
async def test_gateway_ragas_embedding_delegates_single_text() -> None:
    from rag.evaluation.external.ragas import GatewayRagasEmbedding

    service = FakeEmbeddingService()
    embedding = GatewayRagasEmbedding(service)

    result = await embedding.aembed_text("hello")

    assert result == [5.0, 1.0]
    assert service.calls == ["hello"]


@pytest.mark.asyncio
async def test_gateway_ragas_embedding_delegates_batch() -> None:
    from rag.evaluation.external.ragas import GatewayRagasEmbedding

    service = FakeEmbeddingService()
    embedding = GatewayRagasEmbedding(service)

    result = await embedding.aembed_texts(["one", "two"])

    assert result == [[3.0, 1.0], [3.0, 1.0]]
    assert service.calls == ["one", "two"]


def test_gateway_ragas_embedding_rejects_sync_single_text() -> None:
    from rag.evaluation.external.ragas import GatewayRagasEmbedding

    embedding = GatewayRagasEmbedding(FakeEmbeddingService())

    with pytest.raises(
        TypeError,
        match="GatewayRagasEmbedding is asynchronous",
    ):
        embedding.embed_text("hello")


def test_gateway_ragas_embedding_rejects_sync_batch() -> None:
    from rag.evaluation.external.ragas import GatewayRagasEmbedding

    embedding = GatewayRagasEmbedding(FakeEmbeddingService())

    with pytest.raises(
        TypeError,
        match="GatewayRagasEmbedding is asynchronous",
    ):
        embedding.embed_texts(["hello"])


class AnswerRelevancyGateway:
    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    async def generate(
        self,
        prompt: str,
        *,
        temperature: float,
        max_tokens: int,
        user_id: str | None,
        structured_output: dict[str, object] | None,
    ) -> dict[str, object]:
        self.calls.append(
            {
                "prompt": prompt,
                "temperature": temperature,
                "max_tokens": max_tokens,
                "user_id": user_id,
                "structured_output": structured_output,
            }
        )

        assert structured_output is not None
        assert structured_output["name"] == "AnswerRelevanceOutput"
        assert structured_output["strict"] is True

        return {
            "reply": '{"question":"What is DELDAI?","noncommittal":0}',
            "metrics": {},
        }


class AnswerRelevancyEmbeddingService:
    def __init__(self) -> None:
        self.calls: list[str] = []

    async def embed(self, text: str) -> list[float]:
        self.calls.append(text)
        return [1.0, 0.0]


@pytest.mark.asyncio
async def test_ragas_answer_relevancy_uses_gateway_llm_and_embeddings() -> None:
    from rag.evaluation.external.models import ExternalEvaluationSample
    from rag.evaluation.external.ragas import (
        GatewayRagasEmbedding,
        GatewayRagasLLM,
        RagasAnswerRelevancyAdapter,
    )

    gateway = AnswerRelevancyGateway()
    embedding_service = AnswerRelevancyEmbeddingService()

    llm = GatewayRagasLLM(gateway)  # type: ignore[arg-type]
    embeddings = GatewayRagasEmbedding(embedding_service)

    adapter = RagasAnswerRelevancyAdapter(
        llm=llm,
        embeddings=embeddings,
    )

    result = await adapter.evaluate(
        [
            ExternalEvaluationSample(
                query="What is DELDAI?",
                retrieved_contexts=("DELDAI is an enterprise AI platform.",),
                response="DELDAI is an enterprise AI platform.",
            )
        ]
    )

    assert result.provider == "ragas"
    assert result.evaluator == "answer_relevancy"
    assert result.evaluated_samples == 1
    assert result.metrics["answer_relevancy"] == pytest.approx(1.0)

    assert len(gateway.calls) == 3

    for call in gateway.calls:
        structured_output = call["structured_output"]
        assert isinstance(structured_output, dict)
        assert structured_output["name"] == "AnswerRelevanceOutput"
        assert structured_output["strict"] is True
        assert structured_output["schema"] == {
            "description": "Structured output for answer relevance question generation.",
            "properties": {
                "question": {
                    "description": "Question that can be answered from the response",
                    "title": "Question",
                    "type": "string",
                },
                "noncommittal": {
                    "description": ("1 if the response is evasive/vague, 0 if it is substantive"),
                    "title": "Noncommittal",
                    "type": "integer",
                },
            },
            "required": ["question", "noncommittal"],
            "title": "AnswerRelevanceOutput",
            "type": "object",
        }

    assert embedding_service.calls == [
        "What is DELDAI?",
        "What is DELDAI?",
        "What is DELDAI?",
        "What is DELDAI?",
    ]
