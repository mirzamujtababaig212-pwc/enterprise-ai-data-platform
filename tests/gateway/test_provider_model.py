from ai_platform.llm_gateway.models.provider import ProviderInfo


def test_provider_info():

    provider = ProviderInfo(
        name="openai",
        healthy=True,
        available_models=[
            "gpt-4",
            "gpt-4o",
        ],
    )

    assert provider.name == "openai"
    assert provider.healthy is True
    assert provider.available_models == [
        "gpt-4",
        "gpt-4o",
    ]


def test_provider_info_dump():

    provider = ProviderInfo(
        name="gemini",
        healthy=False,
        available_models=["gemini-pro"],
    )

    data = provider.model_dump()

    assert data == {
        "name": "gemini",
        "healthy": False,
        "available_models": [
            "gemini-pro",
        ],
    }
