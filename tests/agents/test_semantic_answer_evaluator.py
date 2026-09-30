from __future__ import annotations

import pytest

from ai_platform.agents.evaluation.semantic_answer_evaluator import (
    SEMANTIC_ANSWER_EVALUATION_METHOD,
    SEMANTIC_ANSWER_PASS_THRESHOLD,
    SemanticAnswerEvaluator,
    SemanticAnswerJudgeOutput,
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
async def test_semantic_answer_evaluator_uses_structured_gateway_output() -> None:
    router = FakeRouter(
        FakeRouterResult(
            reply='{"score":0.92,"passed":true}',
        )
    )

    evaluator = SemanticAnswerEvaluator(
        router=router,  # type: ignore[arg-type]
        model="gpt-4.1-mini",
    )

    result = await evaluator.evaluate(
        actual_answer="An electric vehicle uses electricity stored in a battery.",
        expected_answer="Electric vehicles are powered by energy stored in batteries.",
    )

    assert result.score == 0.92
    assert result.passed is True
    assert result.method == SEMANTIC_ANSWER_EVALUATION_METHOD
    assert result.evaluator_model == "gpt-4.1-mini"
    assert result.evaluator_provider == "openai"

    assert len(router.calls) == 1

    request = router.calls[0]

    assert request["model"] == "gpt-4.1-mini"
    assert request["temperature"] == 0.0
    assert request["max_tokens"] == 256
    assert request["stream"] is False

    structured_output = request["structured_output"]

    assert isinstance(structured_output, dict)
    assert structured_output["name"] == "SemanticAnswerJudgeOutput"
    assert structured_output["schema"] == SemanticAnswerJudgeOutput.model_json_schema()
    assert structured_output["strict"] is True

    prompt = request["prompt"]
    assert isinstance(prompt, str)
    assert "EXPECTED ANSWER:" in prompt
    assert "ACTUAL ANSWER:" in prompt
    assert "energy stored in batteries" in prompt
    assert "electricity stored in a battery" in prompt


@pytest.mark.asyncio
async def test_semantic_answer_evaluator_uses_application_threshold() -> None:
    router = FakeRouter(
        FakeRouterResult(
            reply='{"score":0.79,"passed":true}',
        )
    )

    evaluator = SemanticAnswerEvaluator(
        router=router,  # type: ignore[arg-type]
        model="gpt-4.1-mini",
    )

    result = await evaluator.evaluate(
        actual_answer="The vehicle uses battery power.",
        expected_answer="The vehicle is powered by a battery.",
    )

    assert result.score == 0.79
    assert result.passed is False
    assert SEMANTIC_ANSWER_PASS_THRESHOLD == 0.80


@pytest.mark.asyncio
async def test_semantic_answer_evaluator_rejects_empty_actual_answer() -> None:
    router = FakeRouter(
        FakeRouterResult(
            reply='{"score":1.0,"passed":true}',
        )
    )

    evaluator = SemanticAnswerEvaluator(
        router=router,  # type: ignore[arg-type]
        model="gpt-4.1-mini",
    )

    with pytest.raises(ValueError, match="actual_answer must not be empty"):
        await evaluator.evaluate(
            actual_answer=" ",
            expected_answer="Expected answer.",
        )

    assert router.calls == []


@pytest.mark.asyncio
async def test_semantic_answer_evaluator_rejects_empty_expected_answer() -> None:
    router = FakeRouter(
        FakeRouterResult(
            reply='{"score":1.0,"passed":true}',
        )
    )

    evaluator = SemanticAnswerEvaluator(
        router=router,  # type: ignore[arg-type]
        model="gpt-4.1-mini",
    )

    with pytest.raises(ValueError, match="expected_answer must not be empty"):
        await evaluator.evaluate(
            actual_answer="Actual answer.",
            expected_answer=" ",
        )

    assert router.calls == []


@pytest.mark.asyncio
async def test_semantic_answer_evaluator_rejects_empty_gateway_reply() -> None:
    router = FakeRouter(
        FakeRouterResult(
            reply="",
        )
    )

    evaluator = SemanticAnswerEvaluator(
        router=router,  # type: ignore[arg-type]
        model="gpt-4.1-mini",
    )

    with pytest.raises(ValueError, match="non-empty string 'reply'"):
        await evaluator.evaluate(
            actual_answer="Actual answer.",
            expected_answer="Expected answer.",
        )


@pytest.mark.asyncio
async def test_semantic_answer_evaluator_rejects_invalid_gateway_output() -> None:
    router = FakeRouter(
        FakeRouterResult(
            reply='{"score":"not-a-number","passed":true}',
        )
    )

    evaluator = SemanticAnswerEvaluator(
        router=router,  # type: ignore[arg-type]
        model="gpt-4.1-mini",
    )

    with pytest.raises(ValueError):
        await evaluator.evaluate(
            actual_answer="Actual answer.",
            expected_answer="Expected answer.",
        )


@pytest.mark.asyncio
async def test_semantic_answer_evaluator_rejects_non_dictionary_gateway_response() -> None:
    router = FakeRouter(
        FakeRouterResult(
            reply='{"score":0.9,"passed":true}',
        )
    )
    router.result.response = "not-a-dictionary"

    evaluator = SemanticAnswerEvaluator(
        router=router,  # type: ignore[arg-type]
        model="gpt-4.1-mini",
    )

    with pytest.raises(ValueError, match="response must be a dictionary"):
        await evaluator.evaluate(
            actual_answer="Actual answer.",
            expected_answer="Expected answer.",
        )


def test_semantic_answer_evaluator_rejects_invalid_configuration() -> None:
    router = FakeRouter(
        FakeRouterResult(
            reply='{"score":0.9,"passed":true}',
        )
    )

    with pytest.raises(ValueError, match="model must not be empty"):
        SemanticAnswerEvaluator(
            router=router,  # type: ignore[arg-type]
            model=" ",
        )

    with pytest.raises(ValueError, match="threshold must be between"):
        SemanticAnswerEvaluator(
            router=router,  # type: ignore[arg-type]
            model="gpt-4.1-mini",
            threshold=1.1,
        )
