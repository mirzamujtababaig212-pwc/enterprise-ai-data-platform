from ai_platform.llm_gateway.providers.base_provider import BaseProvider


class GeminiProvider(BaseProvider):
    def chat(self, request: dict) -> dict:
        return {"reply": f"Gemini echo: {request.get('message', '')}"}

    def stream(self, request: dict) -> dict:
        return {"stream": ["gemini-chunk1", "gemini-chunk2"]}

    def embeddings(self, request: dict) -> list[float]:
        return [0.4, 0.5, 0.6]

    def health_check(self) -> dict:
        return {"status": "ok"}

    def list_models(self) -> list[str]:
        return ["gemini-chat", "gemini-embedding"]
