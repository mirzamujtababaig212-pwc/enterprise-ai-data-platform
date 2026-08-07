from typing import Any

from ai_platform.llm_gateway.providers.base_provider import BaseProvider

SUPPORTED_CHAT_MODELS = {
    "anthropic-chat",
}

SUPPORTED_EMBEDDING_MODELS = {
    "anthropic-embedding",
}


class AnthropicProvider(BaseProvider):
    async def chat(self, request: dict[str, Any]) -> dict[str, Any]:
        return {"reply": f"Anthropic echo: {request['prompt']}"}

    async def stream(self, request: dict[str, Any]) -> dict[str, Any]:
        return {"stream": ["anthropic-chunk1", "anthropic-chunk2"]}

    async def embeddings(self, request: dict[str, Any]) -> list[float]:
        model = request["model"]

        if model not in SUPPORTED_EMBEDDING_MODELS:
            raise ValueError(f"Unsupported Anthropic embedding model: {model}")

        return [
            0.7,
            0.8,
            0.9,
        ]

    async def health_check(self) -> dict[str, Any]:
        return {"status": "ok"}

    async def list_models(self) -> list[str]:
        return ["anthropic-chat", "anthropic-embedding"]

    def supported_chat_models(self):
        return list(SUPPORTED_CHAT_MODELS)

    def supported_embedding_models(self):
        return list(SUPPORTED_EMBEDDING_MODELS)
