from ai_platform.llm_gateway.providers.base_provider import BaseProvider


class BedrockProvider(BaseProvider):
    def chat(self, request: dict) -> dict:
        return {"reply": f"Bedrock echo: {request.get('message', '')}"}

    def stream(self, request: dict) -> dict:
        return {"stream": ["bedrock-chunk1", "bedrock-chunk2"]}

    def embeddings(self, request: dict) -> list[float]:
        return [1.0, 1.1, 1.2]

    def health_check(self) -> dict:
        return {"status": "ok"}

    def list_models(self) -> list[str]:
        return ["bedrock-chat", "bedrock-embedding"]
