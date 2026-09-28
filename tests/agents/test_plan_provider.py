from __future__ import annotations

from ai_platform.agents.execution import AgentExecutionContext
from ai_platform.agents.llm_context import AgentLLMContext
from ai_platform.agents.models import AgentDefinition, AgentRequest
from ai_platform.agents.orchestration import OrchestrationStepStatus
from ai_platform.agents.plan_provider import (
    DeterministicAgentPlanProvider,
)
from ai_platform.agents.plans import build_agent_orchestration_plan
from ai_platform.agents.tool_context import AgentToolContext
from tools.registry.in_memory import InMemoryToolRegistry


def make_context(
    *,
    plan_provider=None,
) -> AgentExecutionContext:
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
        plan_provider=plan_provider,
    )


def test_deterministic_plan_provider_builds_registered_plan() -> None:
    context = make_context()
    provider = DeterministicAgentPlanProvider()

    plan = provider.build_plan(context)

    expected = build_agent_orchestration_plan("enterprise-rag-analyst")

    assert plan is not expected
    assert plan.steps == expected.steps


def test_deterministic_plan_provider_returns_fresh_plan_each_time() -> None:
    context = make_context()
    provider = DeterministicAgentPlanProvider()

    first = provider.build_plan(context)
    second = provider.build_plan(context)

    assert first is not second
    assert first.steps == second.steps


def test_execution_context_install_plan_materializes_fresh_state() -> None:
    context = make_context()
    provider = DeterministicAgentPlanProvider()

    first_plan = provider.build_plan(context)
    context.install_orchestration_plan(first_plan)

    first_state = context.orchestration_state
    first_state.start_step(0)

    second_plan = provider.build_plan(context)
    context.install_orchestration_plan(second_plan)

    assert context.orchestration_plan is second_plan
    assert context.orchestration_state is not first_state
    assert context.orchestration_state.current_step_index is None
    assert all(
        step.status is OrchestrationStepStatus.PENDING for step in context.orchestration_state.steps
    )

    assert first_state.current_step_index == 0
    assert first_state.steps[0].status is OrchestrationStepStatus.RUNNING


def test_execution_context_install_none_clears_plan_and_state() -> None:
    context = make_context()
    provider = DeterministicAgentPlanProvider()

    context.install_orchestration_plan(provider.build_plan(context))
    context.install_orchestration_plan(None)

    assert context.orchestration_plan is None
    assert context.orchestration_state.steps == []
    assert context.orchestration_state.current_step_index is None


def test_execution_context_defaults_to_deterministic_plan_provider() -> None:
    context = make_context()

    assert isinstance(context.plan_provider, DeterministicAgentPlanProvider)


def test_execution_context_accepts_custom_plan_provider() -> None:
    class StubPlanProvider:
        def build_plan(self, context):
            return build_agent_orchestration_plan(context.agent_name)

    provider = StubPlanProvider()
    context = make_context(plan_provider=provider)

    assert context.plan_provider is provider
