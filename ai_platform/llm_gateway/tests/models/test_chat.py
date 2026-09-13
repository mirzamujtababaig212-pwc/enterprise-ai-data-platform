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


def test_structured_output_is_preserved_with_schema_alias():
    request = ChatRequest(
        model="enterprise-chat",
        prompt="Return a result",
        structured_output={
            "name": "test_result",
            "schema": {
                "type": "object",
                "properties": {
                    "answer": {"type": "string"},
                },
                "required": ["answer"],
                "additionalProperties": False,
            },
            "strict": True,
        },
    )

    assert request.structured_output is not None
    assert request.structured_output.name == "test_result"
    assert request.structured_output.schema_ == {
        "type": "object",
        "properties": {
            "answer": {"type": "string"},
        },
        "required": ["answer"],
        "additionalProperties": False,
    }
    assert request.structured_output.strict is True
    assert request.model_dump()["structured_output"]["schema_"] == {
        "type": "object",
        "properties": {
            "answer": {"type": "string"},
        },
        "required": ["answer"],
        "additionalProperties": False,
    }
    assert (
        request.model_dump(by_alias=True)["structured_output"]["schema"]
        == request.structured_output.schema_
    )
