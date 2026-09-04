from __future__ import annotations

import pytest

from ai_platform.agents.llm_config import AgentLLMConfig


def test_agent_llm_config_accepts_valid_configuration() -> None:
    config = AgentLLMConfig(
        model="mock-gpt",
        system_prompt="You are an enterprise AI agent.",
        temperature=0.2,
        max_tokens=512,
    )

    assert config.model == "mock-gpt"
    assert config.system_prompt == ("You are an enterprise AI agent.")
    assert config.temperature == 0.2
    assert config.max_tokens == 512


def test_agent_llm_config_rejects_empty_system_prompt() -> None:
    with pytest.raises(
        ValueError,
        match="system prompt must not be empty",
    ):
        AgentLLMConfig(
            model="mock-gpt",
            system_prompt="",
        )


def test_agent_llm_config_rejects_empty_model() -> None:
    with pytest.raises(
        ValueError,
        match="model must not be empty",
    ):
        AgentLLMConfig(
            model="   ",
            system_prompt="You are an agent.",
        )


def test_agent_llm_config_rejects_invalid_temperature() -> None:
    with pytest.raises(
        ValueError,
        match="temperature must be between 0 and 2",
    ):
        AgentLLMConfig(
            model="mock-gpt",
            system_prompt="You are an agent.",
            temperature=2.1,
        )


def test_agent_llm_config_rejects_invalid_max_tokens() -> None:
    with pytest.raises(
        ValueError,
        match="max_tokens must be greater than zero",
    ):
        AgentLLMConfig(
            model="mock-gpt",
            system_prompt="You are an agent.",
            max_tokens=0,
        )
