from ai_platform.llm_gateway.providers.base_provider import BaseProvider


class AzureOpenAIProvider(BaseProvider):
    def chat(self, request: dict) -> dict:
        return {"reply": f"Azure OpenAI echo: {request.get('message', '')}"}

    def stream(self, request: dict) -> dict:
        return {"stream": ["azure-chunk1", "azure-chunk2"]}

    def embeddings(self, request: dict) -> list[float]:
        return [1.6, 1.7, 1.8]

    def health_check(self) -> dict:
        return {"status": "ok"}

    def list_models(self) -> list[str]:
        return ["azure-openai-chat", "azure-openai-embedding"]
