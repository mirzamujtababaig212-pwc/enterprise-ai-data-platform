import asyncio

from ai_platform.llm_gateway.providers.openai_provider import (
    OpenAIProvider,
)


async def main():

    provider = OpenAIProvider()

    request = {
        "prompt": "Explain RAG in one short sentence.",
        "provider": "openai",
        "model": "gpt-4.1-mini",
        "temperature": 0.7,
        "max_tokens": 100,
        "stream": True,
    }

    chunks = []

    async for chunk in provider.stream(request):
        print(f"CHUNK: {chunk!r}")

        chunks.append(chunk)

    print()
    print("FULL RESPONSE:")

    print("".join(chunks))

    assert chunks


if __name__ == "__main__":
    asyncio.run(main())
