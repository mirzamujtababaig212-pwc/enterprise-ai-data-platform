from typing import Any
from typing import Iterable
from ai_platform.llm_gateway.providers.base_provider import BaseProvider

SUPPORTED_CHAT_MODELS = {
    "ollama-chat",
}

SUPPORTED_EMBEDDING_MODELS = {
    "ollama-embedding",
}


class OllamaProvider(BaseProvider):
    async def chat(self, request: dict[str, Any]) -> dict[str, Any]:
        return {"reply": f"Ollama echo: {request['prompt']}"}

    async def stream(
        self,
        request: dict[str, Any],
    ):

        yield "ollama-chunk1"
        yield "ollama-chunk2"

    async def embeddings(self, request: dict[str, Any]) -> list[float]:
        model = request["model"]

        if model not in SUPPORTED_EMBEDDING_MODELS:
            raise ValueError(f"Unsupported Ollama embedding model: {model}")

        return [
            1.3,
            1.4,
            1.5,
        ]

    async def health_check(self) -> dict[str, Any]:
        return {"status": "ok"}

    async def list_models(self) -> list[str]:
        return ["ollama-chat", "ollama-embedding"]

    def supported_chat_models(self):
        return list(SUPPORTED_CHAT_MODELS)

    def supported_embedding_models(self):
        return list(SUPPORTED_EMBEDDING_MODELS)

    def supported_stream_models(self) -> Iterable[str]:
        return self.supported_chat_models()
