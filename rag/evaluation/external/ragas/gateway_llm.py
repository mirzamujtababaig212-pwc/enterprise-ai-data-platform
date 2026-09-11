from __future__ import annotations

from typing import TypeVar

from ragas.llms.base import InstructorBaseRagasLLM

from rag.generation.gateway import GatewayChatService

ResponseModelT = TypeVar("ResponseModelT")


class GatewayRagasLLM(InstructorBaseRagasLLM):
    """
    RAGAS Instructor-compatible LLM backed by the Deldai LLM Gateway.

    RAGAS remains responsible for evaluation prompts and structured response
    models. The Deldai Gateway remains responsible for provider routing,
    model resolution, fallback, authentication, observability, and execution.
    """

    is_async = True

    def __init__(
        self,
        gateway: GatewayChatService,
        *,
        temperature: float = 0.0,
        max_tokens: int = 2048,
        user_id: str | None = None,
    ) -> None:
        if temperature < 0:
            raise ValueError("temperature must be >= 0")

        if max_tokens <= 0:
            raise ValueError("max_tokens must be > 0")

        self.gateway = gateway
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.user_id = user_id

    async def agenerate(
        self,
        prompt: str,
        response_model: type[ResponseModelT],
    ) -> ResponseModelT:
        if not prompt.strip():
            raise ValueError("prompt must not be empty")

        response = await self.gateway.generate(
            prompt,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            user_id=self.user_id,
        )

        reply = response.get("reply")

        if not isinstance(reply, str) or not reply.strip():
            raise ValueError("LLM Gateway response must contain a non-empty string 'reply'")

        return response_model.model_validate_json(reply)

    def generate(
        self,
        prompt: str,
        response_model: type[ResponseModelT],
    ) -> ResponseModelT:
        """
        Synchronous InstructorBaseRagasLLM compatibility.

        RAGAS uses the asynchronous path when is_async=True, so this method
        exists primarily to satisfy the InstructorBaseRagasLLM contract.
        """
        raise TypeError("GatewayRagasLLM is asynchronous; use agenerate() instead.")
