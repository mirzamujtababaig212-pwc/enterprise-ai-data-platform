from ai_platform.llm_gateway.providers.base_provider import BaseProvider


class OllamaProvider(BaseProvider):
    def chat(self, request: dict) -> dict:
        return {"reply": f"Ollama echo: {request.get('message', '')}"}

    def stream(self, request: dict) -> dict:
        return {"stream": ["ollama-chunk1", "ollama-chunk2"]}

    def embeddings(self, request: dict) -> list[float]:
        return [1.3, 1.4, 1.5]

    def health_check(self) -> dict:
        return {"status": "ok"}

    def list_models(self) -> list[str]:
        return ["ollama-chat", "ollama-embedding"]
