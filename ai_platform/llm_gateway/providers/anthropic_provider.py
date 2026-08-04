from ai_platform.llm_gateway.providers.base_provider import BaseProvider


class AnthropicProvider(BaseProvider):
    def chat(self, request: dict) -> dict:
        return {"reply": f"Anthropic echo: {request.get('message', '')}"}

    def stream(self, request: dict) -> dict:
        return {"stream": ["anthropic-chunk1", "anthropic-chunk2"]}

    def embeddings(self, request: dict) -> list[float]:
        return [0.7, 0.8, 0.9]

    def health_check(self) -> dict:
        return {"status": "ok"}

    def list_models(self) -> list[str]:
        return ["anthropic-chat", "anthropic-embedding"]
