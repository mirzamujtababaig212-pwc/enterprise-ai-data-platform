from typing import Any

from ai_platform.llm_gateway.providers.base_provider import BaseProvider

SUPPORTED_CHAT_MODELS = {
    "openai-gpt",
}

SUPPORTED_EMBEDDING_MODELS = {
    "openai-embedding",
}


class OpenAIProvider(BaseProvider):
    async def chat(
        self,
        request: dict[str, Any],
    ) -> dict[str, Any]:
        model = request["model"]
        if model not in SUPPORTED_CHAT_MODELS:
            raise ValueError(f"Unsupported OpenAI model: {model}")

            return {"reply": f"OpenAI echo: {request['prompt']}"}

    async def stream(self, request: dict[str, Any]) -> dict[str, Any]:
        return {"stream": ["openai-chunk1", "openai-chunk2"]}

    async def embeddings(self, request: dict[str, Any]) -> list[float]:
        model = request["model"]

        if model not in SUPPORTED_EMBEDDING_MODELS:
            raise ValueError(f"Unsupported OpenAI embedding model: {model}")

        return [0.1, 0.2, 0.3]

    async def health_check(self) -> dict[str, Any]:
        return {"status": "ok"}

    async def list_models(self) -> list[str]:
        return ["openai-gpt", "openai-embedding"]

    def supported_chat_models(self) -> list[str]:
        return list(SUPPORTED_CHAT_MODELS)

    def supported_embedding_models(self) -> list[str]:
        return list(SUPPORTED_EMBEDDING_MODELS)
