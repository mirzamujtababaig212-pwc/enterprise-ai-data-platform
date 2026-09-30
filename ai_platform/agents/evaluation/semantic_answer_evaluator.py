"""LLM-backed semantic answer evaluation."""

from __future__ import annotations

from dataclasses import dataclass
from pydantic import BaseModel, Field

from ai_platform.llm_gateway.routing.router import Router

SEMANTIC_ANSWER_EVALUATION_METHOD = "llm_judge_v1"
SEMANTIC_ANSWER_PASS_THRESHOLD = 0.80


class SemanticAnswerJudgeOutput(BaseModel):
    """Structured response expected from the semantic answer judge."""

    score: float = Field(ge=0.0, le=1.0)
    passed: bool


@dataclass(frozen=True)
class SemanticAnswerEvaluation:
    """Transient semantic evaluation result."""

    score: float
    passed: bool
    method: str
    evaluator_model: str
    evaluator_provider: str


class SemanticAnswerEvaluator:
    """Evaluate whether an actual answer is semantically equivalent to an expected answer."""

    def __init__(
        self,
        *,
        router: Router,
        model: str,
        threshold: float = SEMANTIC_ANSWER_PASS_THRESHOLD,
    ) -> None:
        if not model.strip():
            raise ValueError("model must not be empty.")

        if not 0.0 <= threshold <= 1.0:
            raise ValueError("threshold must be between 0.0 and 1.0.")

        self._router = router
        self._model = model
        self._threshold = threshold

    async def evaluate(
        self,
        *,
        actual_answer: str,
        expected_answer: str,
    ) -> SemanticAnswerEvaluation:
        """Evaluate semantic equivalence using the central LLM gateway."""

        if not actual_answer.strip():
            raise ValueError("actual_answer must not be empty.")

        if not expected_answer.strip():
            raise ValueError("expected_answer must not be empty.")

        prompt = self._build_prompt(
            actual_answer=actual_answer,
            expected_answer=expected_answer,
        )

        structured_output = {
            "name": SemanticAnswerJudgeOutput.__name__,
            "schema": SemanticAnswerJudgeOutput.model_json_schema(),
            "strict": True,
        }

        result = await self._router.route_chat_with_metadata(
            {
                "model": self._model,
                "prompt": prompt,
                "temperature": 0.0,
                "max_tokens": 256,
                "stream": False,
                "structured_output": structured_output,
            }
        )

        response = result.response

        if not isinstance(response, dict):
            raise ValueError("LLM Gateway response must be a dictionary.")

        reply = response.get("reply")

        if not isinstance(reply, str) or not reply.strip():
            raise ValueError("LLM Gateway response must contain a non-empty string 'reply'.")

        judge_output = SemanticAnswerJudgeOutput.model_validate_json(reply)

        # The model is asked to return a pass/fail value, but the threshold remains
        # authoritative in application code so the evaluation policy is deterministic.
        passed = judge_output.score >= self._threshold

        provider = str(result.provider_name).strip()
        model = str(result.model_name or self._model).strip()

        if not provider:
            raise ValueError("LLM Gateway response must contain a provider name.")

        if not model:
            raise ValueError("LLM Gateway response must contain a model name.")

        return SemanticAnswerEvaluation(
            score=judge_output.score,
            passed=passed,
            method=SEMANTIC_ANSWER_EVALUATION_METHOD,
            evaluator_model=model,
            evaluator_provider=provider,
        )

    @staticmethod
    def _build_prompt(
        *,
        actual_answer: str,
        expected_answer: str,
    ) -> str:
        return f"""
You are evaluating the semantic correctness of an answer.

Determine whether the ACTUAL ANSWER conveys the same substantive meaning as the EXPECTED ANSWER.

Evaluation rules:
- Focus on factual and semantic equivalence.
- Do not require identical wording.
- Treat harmless differences in phrasing, formatting, ordering, or verbosity as acceptable.
- Do not reward an answer merely because it is fluent or well-written.
- Penalize missing required facts, contradictions, materially incorrect claims, or unsupported changes in meaning.
- Score from 0.0 to 1.0, where 1.0 means fully semantically equivalent.
- Return only the requested structured output.
- The application will determine pass/fail from the score threshold.

EXPECTED ANSWER:
{expected_answer}

ACTUAL ANSWER:
{actual_answer}
""".strip()
