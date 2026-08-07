from typing import Any

from ai_platform.llm_gateway.providers.base_provider import BaseProvider

SUPPORTED_CHAT_MODELS = {
    "gemini-chat",
}

SUPPORTED_EMBEDDING_MODELS = {
    "gemini-embedding",
}


class GeminiProvider(BaseProvider):
    async def chat(self, request: dict[str, Any]) -> dict[str, Any]:
        model = request["model"]

        if model not in SUPPORTED_CHAT_MODELS:
            raise ValueError(f"Unsupported Gemini model: {model}")

        return {"reply": f"Gemini echo: {request['prompt']}"}

    async def stream(self, request: dict[str, Any]) -> dict[str, Any]:
        return {"stream": ["gemini-chunk1", "gemini-chunk2"]}

    async def embeddings(self, request: dict[str, Any]) -> list[float]:
        model = request["model"]

        if model not in SUPPORTED_EMBEDDING_MODELS:
            raise ValueError(f"Unsupported Gemini embedding model: {model}")

        return [
            0.4,
            0.5,
            0.6,
        ]

    async def health_check(self) -> dict[str, Any]:
        return {"status": "ok"}

    async def list_models(self) -> list[str]:
        return ["gemini-chat", "gemini-embedding"]

    def supported_chat_models(self):
        return list(SUPPORTED_CHAT_MODELS)

    def supported_embedding_models(self):
        return list(SUPPORTED_EMBEDDING_MODELS)
