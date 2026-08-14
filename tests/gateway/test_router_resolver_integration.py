class FakeProvider:

    def supported_chat_models(self):
        return [
            "gpt-4.1",
        ]

    def supported_embedding_models(self):
        return []

    async def chat(self, request):
        return {
            "provider": "openai",
            "response": "hello",
        }

    async def stream(self, request):
        yield "hello"


class FakeRoutingResolver:

    def resolve(
        self,
        capability,
        model,
        requested_provider=None,
    ):
        return ["openai"]


class FakeProviderResolver:

    def resolve(
        self,
        provider_name,
    ):
        return FakeProvider()
