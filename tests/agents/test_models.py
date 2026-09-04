from __future__ import annotations

import pytest

from ai_platform.agents.models import (
    AgentDefinition,
    AgentRequest,
    AgentResponse,
)


def test_agent_definition_accepts_valid_configuration():
    definition = AgentDefinition(
        name="data-engineering-agent",
        description="Enterprise data engineering assistant.",
        system_prompt="You are an enterprise data engineering assistant.",
        model="test-model",
        temperature=0.2,
        max_tokens=512,
        tool_names=(
            "search_documents",
            "execute_sql",
        ),
        metadata={
            "domain": "data-engineering",
        },
    )

    assert definition.name == "data-engineering-agent"
    assert definition.description == ("Enterprise data engineering assistant.")
    assert definition.system_prompt == ("You are an enterprise data engineering assistant.")
    assert definition.model == "test-model"
    assert definition.temperature == 0.2
    assert definition.max_tokens == 512
    assert definition.tool_names == (
        "search_documents",
        "execute_sql",
    )
    assert definition.metadata == {
        "domain": "data-engineering",
    }
    assert definition.enabled is True


def test_agent_definition_rejects_empty_name():
    with pytest.raises(
        ValueError,
        match="Agent name must not be empty",
    ):
        AgentDefinition(
            name="",
            description="Test agent.",
            system_prompt="You are a test agent.",
        )


def test_agent_definition_rejects_empty_description():
    with pytest.raises(
        ValueError,
        match="Agent description must not be empty",
    ):
        AgentDefinition(
            name="test-agent",
            description="",
            system_prompt="You are a test agent.",
        )


def test_agent_definition_rejects_empty_system_prompt():
    with pytest.raises(
        ValueError,
        match="Agent system prompt must not be empty",
    ):
        AgentDefinition(
            name="test-agent",
            description="Test agent.",
            system_prompt="",
        )


def test_agent_definition_rejects_empty_model():
    with pytest.raises(
        ValueError,
        match="Agent model must not be empty",
    ):
        AgentDefinition(
            name="test-agent",
            description="Test agent.",
            system_prompt="You are a test agent.",
            model="",
        )


def test_agent_definition_rejects_empty_tool_name():
    with pytest.raises(
        ValueError,
        match="Agent tool names must not contain empty values",
    ):
        AgentDefinition(
            name="test-agent",
            description="Test agent.",
            system_prompt="You are a test agent.",
            tool_names=(
                "search_documents",
                "",
            ),
        )


def test_agent_definition_rejects_duplicate_tool_names():
    with pytest.raises(
        ValueError,
        match="Agent tool names must not contain duplicates",
    ):
        AgentDefinition(
            name="test-agent",
            description="Test agent.",
            system_prompt="You are a test agent.",
            tool_names=(
                "search_documents",
                "search_documents",
            ),
        )


def test_agent_definition_copies_metadata():
    metadata = {
        "domain": "data-engineering",
    }

    definition = AgentDefinition(
        name="test-agent",
        description="Test agent.",
        system_prompt="You are a test agent.",
        metadata=metadata,
    )

    metadata["domain"] = "changed"

    assert definition.metadata == {
        "domain": "data-engineering",
    }


def test_agent_request_accepts_valid_input():
    request = AgentRequest(
        input="Find the latest ingestion failures.",
        session_id="session-123",
        user_id="user-456",
        metadata={
            "source": "api",
        },
    )

    assert request.input == ("Find the latest ingestion failures.")
    assert request.session_id == "session-123"
    assert request.user_id == "user-456"
    assert request.metadata == {
        "source": "api",
    }


def test_agent_request_allows_optional_context():
    request = AgentRequest(
        input="Hello agent.",
    )

    assert request.input == "Hello agent."
    assert request.session_id is None
    assert request.user_id is None
    assert request.metadata == {}


def test_agent_request_rejects_empty_input():
    with pytest.raises(
        ValueError,
        match="Agent request input must not be empty",
    ):
        AgentRequest(
            input="",
        )


def test_agent_request_rejects_empty_session_id():
    with pytest.raises(
        ValueError,
        match="session_id must not be empty",
    ):
        AgentRequest(
            input="Hello agent.",
            session_id="",
        )


def test_agent_request_rejects_empty_user_id():
    with pytest.raises(
        ValueError,
        match="user_id must not be empty",
    ):
        AgentRequest(
            input="Hello agent.",
            user_id="",
        )


def test_agent_request_copies_metadata():
    metadata = {
        "source": "api",
    }

    request = AgentRequest(
        input="Hello agent.",
        metadata=metadata,
    )

    metadata["source"] = "changed"

    assert request.metadata == {
        "source": "api",
    }


def test_agent_response_accepts_valid_output():
    response = AgentResponse(
        agent_name="data-engineering-agent",
        output={
            "answer": "No ingestion failures found.",
        },
        session_id="session-123",
        metadata={
            "model": "test-model",
        },
    )

    assert response.agent_name == "data-engineering-agent"
    assert response.output == {
        "answer": "No ingestion failures found.",
    }
    assert response.session_id == "session-123"
    assert response.metadata == {
        "model": "test-model",
    }


def test_agent_response_supports_string_output():
    response = AgentResponse(
        agent_name="test-agent",
        output="Hello from the agent.",
    )

    assert response.agent_name == "test-agent"
    assert response.output == "Hello from the agent."
    assert response.session_id is None
    assert response.metadata == {}


def test_agent_response_rejects_empty_agent_name():
    with pytest.raises(
        ValueError,
        match="agent_name must not be empty",
    ):
        AgentResponse(
            agent_name="",
            output="Hello.",
        )


def test_agent_models_are_immutable():
    definition = AgentDefinition(
        name="test-agent",
        description="Test agent.",
        system_prompt="You are a test agent.",
    )

    request = AgentRequest(
        input="Hello.",
    )

    response = AgentResponse(
        agent_name="test-agent",
        output="Hello.",
    )

    with pytest.raises(AttributeError):
        definition.name = "changed"  # type: ignore[misc]

    with pytest.raises(AttributeError):
        request.input = "changed"  # type: ignore[misc]

    with pytest.raises(AttributeError):
        response.output = "changed"  # type: ignore[misc]


def test_agent_definition_rejects_invalid_temperature():
    with pytest.raises(
        ValueError,
        match="Agent temperature must be between 0 and 2",
    ):
        AgentDefinition(
            name="test-agent",
            description="Test agent.",
            system_prompt="You are a test agent.",
            temperature=2.1,
        )


def test_agent_definition_rejects_invalid_max_tokens():
    with pytest.raises(
        ValueError,
        match="Agent max_tokens must be greater than zero",
    ):
        AgentDefinition(
            name="test-agent",
            description="Test agent.",
            system_prompt="You are a test agent.",
            max_tokens=0,
        )


def test_agent_definition_builds_llm_config():
    definition = AgentDefinition(
        name="test-agent",
        description="Test agent.",
        system_prompt="You are a test agent.",
        model="custom-model",
        temperature=0.25,
        max_tokens=768,
    )

    config = definition.llm_config

    assert config.model == "custom-model"
    assert config.system_prompt == "You are a test agent."
    assert config.temperature == 0.25
    assert config.max_tokens == 768
