from typing import Any

from ai_platform.llm_gateway.providers.base_provider import BaseProvider

SUPPORTED_CHAT_MODELS = {
    "azure-openai-chat",
}

SUPPORTED_EMBEDDING_MODELS = {
    "azure-openai-embedding",
}


class AzureOpenAIProvider(BaseProvider):
    async def chat(self, request: dict[str, Any]) -> dict[str, Any]:
        return {"reply": f"Azure OpenAI echo: {request['prompt']}"}

    async def stream(self, request: dict[str, Any]) -> dict[str, Any]:
        return {"stream": ["azure-chunk1", "azure-chunk2"]}

    async def embeddings(self, request: dict[str, Any]) -> list[float]:
        model = request["model"]

        if model not in SUPPORTED_EMBEDDING_MODELS:
            raise ValueError(f"Unsupported Azure OpenAI embedding model: {model}")

        return [
            1.6,
            1.7,
            1.8,
        ]

    async def health_check(self) -> dict[str, Any]:
        return {"status": "ok"}

    async def list_models(self) -> list[str]:
        return ["azure-openai-chat", "azure-openai-embedding"]

    def supported_chat_models(self):
        return list(SUPPORTED_CHAT_MODELS)

    def supported_embedding_models(self):
        return list(SUPPORTED_EMBEDDING_MODELS)
