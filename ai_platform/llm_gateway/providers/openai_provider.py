from ai_platform.llm_gateway.providers.base_provider import BaseProvider


class OpenAIProvider(BaseProvider):
    def chat(self, request: dict) -> dict:
        return {"reply": f"OpenAI echo: {request.get('message', '')}"}

    def stream(self, request: dict) -> dict:
        return {"stream": ["openai-chunk1", "openai-chunk2"]}

    def embeddings(self, request: dict) -> list[float]:
        return [0.1, 0.2, 0.3]

    def health_check(self) -> dict:
        return {"status": "ok"}

    def list_models(self) -> list[str]:
        return ["openai-gpt", "openai-embedding"]
