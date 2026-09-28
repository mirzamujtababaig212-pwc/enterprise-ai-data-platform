from __future__ import annotations

from ai_platform.agents.decision_provider import (
    DeterministicAgentRuntimeDecisionProvider,
)
from ai_platform.agents.execution import AgentExecutionContext
from ai_platform.agents.llm_context import AgentLLMContext
from ai_platform.agents.models import AgentDefinition, AgentRequest
from ai_platform.agents.models import AgentResponse
from ai_platform.agents.orchestration import AgentRuntimeDecision
from ai_platform.agents.runtime_evaluation import AgentRuntimeEvaluationSnapshot
from ai_platform.agents.tool_context import AgentToolContext
from tools.registry.in_memory import InMemoryToolRegistry


def make_context() -> AgentExecutionContext:
    definition = AgentDefinition(
        name="enterprise-rag-analyst",
        description="Test agent",
        system_prompt="You are a test agent.",
    )

    tools = AgentToolContext(
        InMemoryToolRegistry(),
        definition,
    )

    llm = AgentLLMContext(
        type(
            "FakeGateway",
            (),
            {
                "route_chat": lambda self, request: None,
            },
        )(),
        definition.llm_config,
    )

    return AgentExecutionContext(
        AgentRequest(input="Hello."),
        tools=tools,
        llm=llm,
    )


def test_deterministic_decision_provider_returns_stop() -> None:
    provider = DeterministicAgentRuntimeDecisionProvider()

    context = make_context()
    evaluation = AgentRuntimeEvaluationSnapshot(
        iteration=1,
        step_index=None,
        response=AgentResponse(
            agent_name="test-agent",
            output="test",
        ),
        tool_rounds=0,
    )

    assert provider.evaluate(context, evaluation) is AgentRuntimeDecision.STOP


def test_execution_context_defaults_to_deterministic_decision_provider() -> None:
    context = make_context()

    assert isinstance(
        context.decision_provider,
        DeterministicAgentRuntimeDecisionProvider,
    )


def test_execution_context_accepts_custom_decision_provider() -> None:
    class StubDecisionProvider:
        def evaluate(self, context):
            return AgentRuntimeDecision.CONTINUE

    definition = AgentDefinition(
        name="enterprise-rag-analyst",
        description="Test agent",
        system_prompt="You are a test agent.",
    )

    tools = AgentToolContext(
        InMemoryToolRegistry(),
        definition,
    )

    llm = AgentLLMContext(
        type(
            "FakeGateway",
            (),
            {
                "route_chat": lambda self, request: None,
            },
        )(),
        definition.llm_config,
    )

    provider = StubDecisionProvider()
    context = AgentExecutionContext(
        AgentRequest(input="Hello."),
        tools=tools,
        llm=llm,
        decision_provider=provider,
    )

    assert context.decision_provider is provider


def make_evaluation(
    *,
    iteration: int = 1,
    output="test",
    step_index: int | None = None,
    tool_rounds: int = 0,
) -> AgentRuntimeEvaluationSnapshot:
    return AgentRuntimeEvaluationSnapshot(
        iteration=iteration,
        step_index=step_index,
        response=AgentResponse(
            agent_name="test-agent",
            output=output,
        ),
        tool_rounds=tool_rounds,
    )


def test_deterministic_decision_provider_stops_on_none_output() -> None:
    provider = DeterministicAgentRuntimeDecisionProvider()

    context = make_context()
    evaluation = make_evaluation(output=None)

    assert provider.evaluate(context, evaluation) is AgentRuntimeDecision.STOP


def test_deterministic_decision_provider_stops_on_blank_string_output() -> None:
    provider = DeterministicAgentRuntimeDecisionProvider()

    context = make_context()
    evaluation = make_evaluation(output="   ")

    assert provider.evaluate(context, evaluation) is AgentRuntimeDecision.STOP


def test_deterministic_decision_provider_continues_before_iteration_limit() -> None:
    provider = DeterministicAgentRuntimeDecisionProvider()

    context = make_context()
    context.execution_budget = context.execution_budget.__class__(max_iterations=3)

    evaluation = make_evaluation(iteration=1)

    assert provider.evaluate(context, evaluation) is AgentRuntimeDecision.CONTINUE


def test_deterministic_decision_provider_stops_at_iteration_limit() -> None:
    provider = DeterministicAgentRuntimeDecisionProvider()

    context = make_context()
    context.execution_budget = context.execution_budget.__class__(max_iterations=3)

    evaluation = make_evaluation(iteration=3)

    assert provider.evaluate(context, evaluation) is AgentRuntimeDecision.STOP


def test_deterministic_decision_provider_accepts_non_string_output() -> None:
    provider = DeterministicAgentRuntimeDecisionProvider()

    context = make_context()
    evaluation = make_evaluation(output={"answer": "structured"})

    assert provider.evaluate(context, evaluation) is AgentRuntimeDecision.STOP
