from tools.models import ToolDefinition, ToolProvider


def test_tool_provider_is_immutable_and_normalized():
    provider = ToolProvider(
        kind="mcp",
        name="document-server",
    )

    assert provider.kind == "mcp"
    assert provider.name == "document-server"

    definition = ToolDefinition(
        name="documents.search",
        description="Search documents.",
        provider=provider,
    )

    assert definition.provider == provider


def test_tool_provider_rejects_empty_kind():
    try:
        ToolProvider(kind="", name="internal")
    except ValueError as exc:
        assert str(exc) == "Tool provider kind must not be empty."
    else:
        raise AssertionError("Expected empty provider kind to be rejected.")


def test_tool_provider_rejects_empty_name():
    try:
        ToolProvider(kind="native", name="")
    except ValueError as exc:
        assert str(exc) == "Tool provider name must not be empty."
    else:
        raise AssertionError("Expected empty provider name to be rejected.")
