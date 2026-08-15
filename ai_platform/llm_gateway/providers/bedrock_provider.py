from typing import Any
from typing import Iterable
from ai_platform.llm_gateway.providers.base_provider import BaseProvider

SUPPORTED_CHAT_MODELS = {
    "bedrock-chat",
}

SUPPORTED_EMBEDDING_MODELS = {
    "bedrock-embedding",
}


class BedrockProvider(BaseProvider):
    async def chat(self, request: dict[str, Any]) -> dict[str, Any]:
        return {"reply": f"Bedrock echo: {request['prompt']}"}

    async def stream(
        self,
        request: dict[str, Any],
    ):

        yield "bedrock-chunk1"
        yield "bedrock-chunk2"

    async def embeddings(self, request: dict[str, Any]) -> list[float]:
        model = request["model"]

        if model not in SUPPORTED_EMBEDDING_MODELS:
            raise ValueError(f"Unsupported Bedrock embedding model: {model}")

        return [
            1.0,
            1.1,
            1.2,
        ]

    async def health_check(self) -> dict[str, Any]:
        return {"status": "ok"}

    async def list_models(self) -> list[str]:
        return ["bedrock-chat", "bedrock-embedding"]

    def supported_chat_models(self):
        return list(SUPPORTED_CHAT_MODELS)

    def supported_embedding_models(self):
        return list(SUPPORTED_EMBEDDING_MODELS)

    def supported_stream_models(self) -> Iterable[str]:
        return self.supported_chat_models()
