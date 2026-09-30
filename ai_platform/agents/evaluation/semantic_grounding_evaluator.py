"""LLM-backed semantic grounding evaluation."""

from __future__ import annotations

from dataclasses import dataclass

from pydantic import BaseModel, Field

from ai_platform.llm_gateway.routing.router import Router

SEMANTIC_GROUNDING_EVALUATION_METHOD = "llm_grounding_judge_v1"
SEMANTIC_GROUNDING_PASS_THRESHOLD = 0.80


class SemanticGroundingJudgeOutput(BaseModel):
    """Structured response expected from the semantic grounding judge."""

    score: float = Field(ge=0.0, le=1.0)
    passed: bool


@dataclass(frozen=True)
class SemanticGroundingEvaluation:
    """Transient semantic grounding evaluation result."""

    score: float
    passed: bool
    method: str
    evaluator_model: str
    evaluator_provider: str


class SemanticGroundingEvaluator:
    """Evaluate whether an answer is supported by retrieved source material."""

    def __init__(
        self,
        *,
        router: Router,
        model: str,
        threshold: float = SEMANTIC_GROUNDING_PASS_THRESHOLD,
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
        answer_text: str,
        source_texts: list[str] | tuple[str, ...],
    ) -> SemanticGroundingEvaluation:
        """Evaluate whether the answer's claims are supported by retrieved sources."""

        if not answer_text.strip():
            raise ValueError("answer_text must not be empty.")

        normalized_sources = tuple(
            source.strip() for source in source_texts if isinstance(source, str) and source.strip()
        )

        if not normalized_sources:
            raise ValueError("source_texts must contain at least one non-empty source.")

        prompt = self._build_prompt(
            answer_text=answer_text,
            source_texts=normalized_sources,
        )

        structured_output = {
            "name": SemanticGroundingJudgeOutput.__name__,
            "schema": SemanticGroundingJudgeOutput.model_json_schema(),
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

        judge_output = SemanticGroundingJudgeOutput.model_validate_json(reply)

        # The model is asked to return pass/fail, but the application threshold
        # remains authoritative so pass/fail semantics are deterministic.
        passed = judge_output.score >= self._threshold

        provider = str(result.provider_name).strip()
        model = str(result.model_name or self._model).strip()

        if not provider:
            raise ValueError("LLM Gateway response must contain a provider name.")

        if not model:
            raise ValueError("LLM Gateway response must contain a model name.")

        return SemanticGroundingEvaluation(
            score=judge_output.score,
            passed=passed,
            method=SEMANTIC_GROUNDING_EVALUATION_METHOD,
            evaluator_model=model,
            evaluator_provider=provider,
        )

    @staticmethod
    def _build_prompt(
        *,
        answer_text: str,
        source_texts: tuple[str, ...],
    ) -> str:
        sources = "\n\n".join(
            f"SOURCE {index}:\n{source}" for index, source in enumerate(source_texts, start=1)
        )

        return f"""
You are evaluating whether an answer is grounded in retrieved source material.

Determine whether the substantive claims in the ANSWER are supported by the
provided RETRIEVED SOURCES.

Evaluation rules:
- Evaluate source support, not general answer quality.
- Judge whether the answer's factual claims can be supported by the sources.
- Do not require identical wording between the answer and sources.
- Paraphrases and reasonable synthesis are acceptable when supported by the sources.
- Penalize claims that contradict the sources.
- Penalize material factual claims that are not supported by the sources.
- Do not reward fluency, style, completeness, or relevance by themselves.
- A concise answer can receive a high score if its claims are well supported.
- If the answer contains multiple substantive claims, consider support across
  the answer rather than relying on a single matching phrase.
- Score from 0.0 to 1.0, where 1.0 means the answer is fully supported by
  the retrieved sources.
- Return only the requested structured output.
- The application will determine pass/fail from the score threshold.

ANSWER:
{answer_text}

RETRIEVED SOURCES:
{sources}
""".strip()
