from ai_platform.llm_gateway.models.chat import ChatRequest


def test_provider_defaults_to_none_for_automatic_routing():
    request = ChatRequest(
        model="enterprise-chat",
        prompt="test",
    )

    assert request.provider is None
    assert request.model_dump()["provider"] is None


def test_explicit_provider_is_preserved():
    request = ChatRequest(
        model="enterprise-chat",
        prompt="test",
        provider="ollama",
    )

    assert request.provider == "ollama"
    assert request.model_dump()["provider"] == "ollama"
