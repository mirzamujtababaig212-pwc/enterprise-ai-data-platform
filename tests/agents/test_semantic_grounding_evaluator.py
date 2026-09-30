from __future__ import annotations

import pytest
from pydantic import ValidationError

from ai_platform.agents.evaluation.semantic_grounding_evaluator import (
    SEMANTIC_GROUNDING_EVALUATION_METHOD,
    SemanticGroundingEvaluator,
)


class FakeRouterResult:
    def __init__(
        self,
        *,
        reply: str,
        provider_name: str = "openai",
        model_name: str = "gpt-4.1-mini",
    ) -> None:
        self.response = {
            "reply": reply,
        }
        self.provider_name = provider_name
        self.model_name = model_name


class FakeRouter:
    def __init__(self, result: FakeRouterResult) -> None:
        self.result = result
        self.calls: list[dict[str, object]] = []

    async def route_chat_with_metadata(
        self,
        request: dict[str, object],
    ) -> FakeRouterResult:
        self.calls.append(request)
        return self.result


@pytest.mark.asyncio
async def test_semantic_grounding_evaluator_uses_structured_gateway_output() -> None:
    router = FakeRouter(
        FakeRouterResult(
            reply='{"score":0.92,"passed":true}',
        )
    )

    evaluator = SemanticGroundingEvaluator(
        router=router,  # type: ignore[arg-type]
        model="gpt-4.1-mini",
    )

    result = await evaluator.evaluate(
        answer_text="The vehicle battery stores electrical energy.",
        source_texts=(
            "The vehicle battery stores electrical energy for later use.",
            "The electric motor converts electrical energy into mechanical motion.",
        ),
    )

    assert result.score == 0.92
    assert result.passed is True
    assert result.method == SEMANTIC_GROUNDING_EVALUATION_METHOD
    assert result.evaluator_model == "gpt-4.1-mini"
    assert result.evaluator_provider == "openai"

    assert len(router.calls) == 1

    request = router.calls[0]

    assert request["model"] == "gpt-4.1-mini"
    assert request["temperature"] == 0.0
    assert request["stream"] is False

    structured_output = request["structured_output"]
    assert isinstance(structured_output, dict)
    assert structured_output["strict"] is True

    prompt = request["prompt"]
    assert isinstance(prompt, str)
    assert "The vehicle battery stores electrical energy." in prompt
    assert "The vehicle battery stores electrical energy for later use." in prompt
    assert "The electric motor converts electrical energy" in prompt


@pytest.mark.asyncio
async def test_semantic_grounding_application_threshold_is_authoritative() -> None:
    router = FakeRouter(
        FakeRouterResult(
            reply='{"score":0.79,"passed":true}',
        )
    )

    evaluator = SemanticGroundingEvaluator(
        router=router,  # type: ignore[arg-type]
        model="gpt-4.1-mini",
        threshold=0.80,
    )

    result = await evaluator.evaluate(
        answer_text="The battery stores electrical energy.",
        source_texts=("The battery stores electrical energy.",),
    )

    assert result.score == 0.79
    assert result.passed is False


@pytest.mark.asyncio
async def test_semantic_grounding_rejects_empty_answer() -> None:
    router = FakeRouter(
        FakeRouterResult(
            reply='{"score":0.92,"passed":true}',
        )
    )

    evaluator = SemanticGroundingEvaluator(
        router=router,  # type: ignore[arg-type]
        model="gpt-4.1-mini",
    )

    with pytest.raises(ValueError, match="answer_text must not be empty"):
        await evaluator.evaluate(
            answer_text="   ",
            source_texts=("source",),
        )

    assert router.calls == []


@pytest.mark.asyncio
async def test_semantic_grounding_rejects_missing_sources() -> None:
    router = FakeRouter(
        FakeRouterResult(
            reply='{"score":0.92,"passed":true}',
        )
    )

    evaluator = SemanticGroundingEvaluator(
        router=router,  # type: ignore[arg-type]
        model="gpt-4.1-mini",
    )

    with pytest.raises(
        ValueError,
        match="source_texts must contain at least one non-empty source",
    ):
        await evaluator.evaluate(
            answer_text="The answer.",
            source_texts=(" ", "", "\t"),
        )

    assert router.calls == []


@pytest.mark.asyncio
async def test_semantic_grounding_rejects_empty_gateway_reply() -> None:
    router = FakeRouter(
        FakeRouterResult(
            reply=" ",
        )
    )

    evaluator = SemanticGroundingEvaluator(
        router=router,  # type: ignore[arg-type]
        model="gpt-4.1-mini",
    )

    with pytest.raises(
        ValueError,
        match="non-empty string 'reply'",
    ):
        await evaluator.evaluate(
            answer_text="The answer.",
            source_texts=("The source.",),
        )


@pytest.mark.asyncio
async def test_semantic_grounding_rejects_invalid_gateway_output() -> None:
    router = FakeRouter(
        FakeRouterResult(
            reply='{"score":"not-a-number","passed":true}',
        )
    )

    evaluator = SemanticGroundingEvaluator(
        router=router,  # type: ignore[arg-type]
        model="gpt-4.1-mini",
    )

    with pytest.raises(ValidationError):
        await evaluator.evaluate(
            answer_text="The answer.",
            source_texts=("The source.",),
        )


@pytest.mark.asyncio
async def test_semantic_grounding_rejects_non_dictionary_gateway_response() -> None:
    router = FakeRouter(
        FakeRouterResult(
            reply='{"score":0.92,"passed":true}',
        )
    )
    router.result.response = "not-a-dictionary"

    evaluator = SemanticGroundingEvaluator(
        router=router,  # type: ignore[arg-type]
        model="gpt-4.1-mini",
    )

    with pytest.raises(
        ValueError,
        match="response must be a dictionary",
    ):
        await evaluator.evaluate(
            answer_text="The answer.",
            source_texts=("The source.",),
        )


@pytest.mark.asyncio
async def test_semantic_grounding_rejects_invalid_configuration() -> None:
    router = FakeRouter(
        FakeRouterResult(
            reply='{"score":0.92,"passed":true}',
        )
    )

    with pytest.raises(ValueError, match="model must not be empty"):
        SemanticGroundingEvaluator(
            router=router,  # type: ignore[arg-type]
            model=" ",
        )

    with pytest.raises(ValueError, match="threshold must be between"):
        SemanticGroundingEvaluator(
            router=router,  # type: ignore[arg-type]
            model="gpt-4.1-mini",
            threshold=1.1,
        )
