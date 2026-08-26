from __future__ import annotations

from typing import Any

from openai import AsyncOpenAI

from app.providers.base import LLMProvider, LLMRequest, LLMResponse


class OpenAIProvider(LLMProvider):

    def __init__(
        self,
        api_key: str,
        default_model: str = "gpt-4o-mini",
    ) -> None:
        self.client = AsyncOpenAI(api_key=api_key)
        self.default_model = default_model

    @property
    def name(self) -> str:
        return "openai"

    async def generate(self, request: LLMRequest) -> LLMResponse:

        model = request.model or self.default_model

        kwargs: dict[str, Any] = {
            "model": model,
            "messages": request.messages,
            "temperature": request.temperature,
        }

        if request.max_tokens is not None:
            kwargs["max_tokens"] = request.max_tokens

        response = await self.client.chat.completions.create(**kwargs)

        usage = response.usage

        input_tokens = usage.prompt_tokens if usage is not None else 0

        output_tokens = usage.completion_tokens if usage is not None else 0

        total_tokens = usage.total_tokens if usage is not None else input_tokens + output_tokens

        content = response.choices[0].message.content or ""

        return LLMResponse(
            provider=self.name,
            model=response.model,
            content=content,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_tokens=total_tokens,
            request_id=response.id,
        )
