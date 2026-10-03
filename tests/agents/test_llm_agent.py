from __future__ import annotations

from app.control_plane.agent_run_steps.in_memory import InMemoryAgentRunStepsRepository
from app.control_plane.agent_run_steps.models import AgentRunStep, AgentRunStepStatus
from app.control_plane.approvals.coordinator import ControlPlaneApprovalCoordinator
from app.control_plane.approvals.in_memory import InMemoryApprovalRequestRepository
from app.control_plane.approvals.models import ApprovalStatus
from app.control_plane.approvals.policy import SideEffectApprovalPolicy

import asyncio
from datetime import datetime
from time import monotonic

from typing import Any
from unittest.mock import patch

import pytest

from tests.tools.execution.test_service import (
    FakeTool,
    FailingTool,
    OwnershipLosingTool,
    ProvenanceToolAuthorizer,
)
from ai_platform.agents.contracts import Agent
from ai_platform.agents.checkpoint import (
    AgentCheckpointPosition,
    AgentExecutionCheckpoint,
)
from ai_platform.agents.budget import ExecutionBudget, ExecutionBudgetState
from ai_platform.agents.exceptions import (
    AgentExecutionDurationLimitError,
    AgentExecutionOwnershipLostError,
    AgentExecutionWaitingForApprovalError,
    AgentLLMCallLimitError,
    AgentOutputPolicyError,
    AgentTokenLimitError,
    AgentToolLoopLimitError,
)
from ai_platform.agents.execution import AgentExecutionContext
from ai_platform.agents.orchestration import AgentRuntimeDecision, AgentRuntimePhase
from ai_platform.agents.runtime_evaluation import AgentRuntimeEvaluationSnapshot
from ai_platform.agents.llm_context import AgentLLMContext
from ai_platform.agents.policy import (
    ModelGovernanceDecision,
    OutputGovernanceDecision,
)
from ai_platform.agents.llm_config import AgentLLMConfig
from ai_platform.agents.llm_messages import (
    AgentMessageRole,
    assistant_message,
    assistant_tool_call_message,
    system_message,
    tool_message,
    tool_result_message,
    user_message,
)
from ai_platform.agents.orchestration import (
    OrchestrationPlan,
    OrchestrationStep,
    OrchestrationStepCompletionPolicy,
    OrchestrationStepStatus,
)
from ai_platform.agents.models import (
    AgentDefinition,
    AgentRequest,
    AgentResponse,
)
from ai_platform.agents.tool_context import AgentToolContext
from tools.registry.in_memory import InMemoryToolRegistry
from tools.models import (
    ToolDefinition,
    ToolExecutionFailureCategory,
    ToolExecutionPolicy,
    ToolProvider,
)
from tools.rag.search import RAGSearchTool
from tools.execution.idempotency import InMemoryToolExecutionIdempotencyStore
from tools.execution.service import ToolExecutionService
from tools.execution.context import ToolExecutionContext
from tools.authorization.service import ToolAuthorizationService
from rag.governance import GovernancePolicy
from rag.models import DocumentChunk, RetrievalResult
from app.control_plane.retries.policy import RetryPolicy
from ai_platform.agents.llm_agent import LLMAgent
from ai_platform.agents.plans import build_enterprise_rag_analyst_plan
from memory.context.builder import MemoryContext
from memory.models import MemoryItem
from ai_platform.agents.tool_calls import AgentToolCall
from ai_platform.agents.observability import (
    AgentExecutionEvent,
    AgentExecutionEventType,
)


class FakeLLMGateway:
    def __init__(self) -> None:
        self.requests: list[dict[str, Any]] = []

    async def route_chat(
        self,
        request: dict[str, Any],
    ) -> dict[str, Any]:
        self.requests.append(request)

        return {
            "provider": "fake",
            "model": request["model"],
            "reply": "Generated answer.",
            "usage": {
                "prompt_tokens": 10,
                "completion_tokens": 5,
                "total_tokens": 15,
            },
        }


class FakeLLMAgent:
    """
    Minimal executable Agent implementation used to validate
    the AgentExecutionContext + AgentLLMContext contract.

    This is intentionally a test fixture, not production code.
    """

    def __init__(self) -> None:
        self._definition = AgentDefinition(
            name="test-llm-agent",
            description="Test LLM-backed agent.",
            system_prompt="You are a test LLM agent.",
            model="mock-gpt",
        )

    @property
    def definition(self) -> AgentDefinition:
        return self._definition

    async def run(
        self,
        context: AgentExecutionContext,
    ) -> AgentResponse:
        messages = context.build_llm_messages()

        assert messages[0].role is AgentMessageRole.SYSTEM
        assert messages[0].content == self.definition.system_prompt

        result = await context.llm.generate(
            prompt=context.request.input,
            messages=messages,
            user_id=context.user_id,
        )

        return AgentResponse(
            agent_name=self.definition.name,
            output=result.text,
            session_id=context.session_id,
            metadata={
                "provider": result.provider,
                "model": result.model,
                "usage": {
                    "prompt_tokens": result.usage.prompt_tokens,
                    "completion_tokens": result.usage.completion_tokens,
                    "total_tokens": result.usage.total_tokens,
                },
            },
        )


def make_context(
    *,
    request: AgentRequest | None = None,
    history: tuple[Any, ...] = (),
    run_id: str | None = None,
    agent_run_steps_repository_factory=None,
) -> tuple[AgentExecutionContext, FakeLLMGateway]:
    definition = AgentDefinition(
        name="test-llm-agent",
        description="Test LLM-backed agent.",
        system_prompt="You are a test LLM agent.",
        model="mock-gpt",
    )

    gateway = FakeLLMGateway()

    llm_context = AgentLLMContext(
        gateway,
        AgentLLMConfig(
            model=definition.model,
            system_prompt=definition.system_prompt,
        ),
    )

    tools = AgentToolContext(
        InMemoryToolRegistry(),
        definition,
    )

    context = AgentExecutionContext(
        (
            request
            if request is not None
            else AgentRequest(
                input="Hello.",
            )
        ),
        tools=tools,
        llm=llm_context,
        history=history,
        run_id=run_id,
        agent_run_steps_repository_factory=agent_run_steps_repository_factory,
    )

    return context, gateway


def test_llm_agent_rejects_invalid_runtime_decision_provider_result() -> None:
    context, _ = make_context()
    context.runtime_state.transition_to(AgentRuntimePhase.ACT)
    context.runtime_state.transition_to(AgentRuntimePhase.OBSERVE)
    context.runtime_state.transition_to(AgentRuntimePhase.EVALUATE)

    class InvalidDecisionProvider:
        def evaluate(
            self,
            context: AgentExecutionContext,
            evaluation: AgentRuntimeEvaluationSnapshot,
        ) -> AgentRuntimeDecision:
            return AgentRuntimeDecision.STOP

    context.decision_provider = InvalidDecisionProvider()

    response = AgentResponse(
        agent_name="test-llm-agent",
        output="Generated answer.",
        session_id=context.session_id,
    )

    with pytest.raises(
        TypeError,
        match="Runtime decision provider must return an AgentRuntimeDecisionResult",
    ):
        LLMAgent._evaluate_runtime_decision(
            context,
            response=response,
            tool_rounds=0,
        )

    assert context.runtime_state.phase is AgentRuntimePhase.EVALUATE
    assert context.runtime_state.decision is None
    assert context.runtime_state.decision_reason is None


@pytest.mark.asyncio
async def test_executable_llm_agent_implements_agent_contract() -> None:
    agent: Agent = FakeLLMAgent()

    assert agent.definition.name == "test-llm-agent"
    assert agent.definition.model == "mock-gpt"


@pytest.mark.asyncio
async def test_executable_llm_agent_calls_gateway() -> None:
    agent = FakeLLMAgent()

    context, gateway = make_context(
        request=AgentRequest(
            input="Explain RAG.",
            user_id="user-123",
        ),
    )

    response = await agent.run(context)

    assert response.agent_name == "test-llm-agent"
    assert response.output == "Generated answer."
    assert response.metadata == {
        "provider": "fake",
        "model": "mock-gpt",
        "usage": {
            "prompt_tokens": 10,
            "completion_tokens": 5,
            "total_tokens": 15,
        },
    }
    assert gateway.requests == [
        {
            "prompt": "Explain RAG.",
            "messages": [
                {
                    "role": "system",
                    "content": "You are a test LLM agent.",
                },
                {
                    "role": "user",
                    "content": "Explain RAG.",
                },
            ],
            "model": "mock-gpt",
            "temperature": 0.7,
            "max_tokens": 1024,
            "stream": False,
            "user_id": "user-123",
        }
    ]


@pytest.mark.asyncio
async def test_executable_llm_agent_preserves_session_id() -> None:
    agent = FakeLLMAgent()

    context, _ = make_context(
        request=AgentRequest(
            input="Continue the conversation.",
            session_id="session-456",
        ),
    )

    response = await agent.run(context)

    assert response.session_id == "session-456"


@pytest.mark.asyncio
async def test_executable_llm_agent_builds_canonical_messages() -> None:
    FakeLLMAgent()

    history = (
        user_message("What is RAG?"),
        assistant_message("RAG retrieves relevant context."),
    )

    context, _ = make_context(
        request=AgentRequest(
            input="Why is retrieval useful?",
        ),
        history=history,
    )

    messages = context.build_llm_messages()

    assert messages == (
        system_message("You are a test LLM agent."),
        user_message("What is RAG?"),
        assistant_message("RAG retrieves relevant context."),
        user_message("Why is retrieval useful?"),
    )


@pytest.mark.asyncio
async def test_executable_llm_agent_sends_history_to_gateway() -> None:
    agent = FakeLLMAgent()

    history = (
        user_message("What is RAG?"),
        assistant_message("RAG retrieves relevant context."),
    )

    context, gateway = make_context(
        request=AgentRequest(
            input="Why is retrieval useful?",
        ),
        history=history,
    )

    await agent.run(context)

    assert gateway.requests[0]["messages"] == [
        {
            "role": "system",
            "content": "You are a test LLM agent.",
        },
        {
            "role": "user",
            "content": "What is RAG?",
        },
        {
            "role": "assistant",
            "content": "RAG retrieves relevant context.",
        },
        {
            "role": "user",
            "content": "Why is retrieval useful?",
        },
    ]


@pytest.mark.asyncio
async def test_executable_llm_agent_can_build_messages_with_tool_results() -> None:
    context, _ = make_context(
        request=AgentRequest(
            input="What is the pipeline status?",
        ),
    )

    messages = context.build_llm_messages(
        tool_results=('{"status": "healthy"}',),
    )

    assert messages == (
        system_message("You are a test LLM agent."),
        user_message("What is the pipeline status?"),
        tool_message('{"status": "healthy"}'),
    )


@pytest.mark.asyncio
async def test_llm_agent_run_completes_current_agent_response_orchestration_step() -> None:
    from ai_platform.agents.orchestration import (
        OrchestrationPlan,
        OrchestrationStep,
        OrchestrationStepCompletionPolicy,
        OrchestrationStepStatus,
    )

    context, gateway = make_context()

    plan = OrchestrationPlan(
        steps=(
            OrchestrationStep(
                step_id="answer",
                step_index=0,
                name="Produce answer",
                status=OrchestrationStepStatus.PENDING,
                completion_policy=OrchestrationStepCompletionPolicy.ON_AGENT_RESPONSE,
            ),
        )
    )
    context.orchestration_plan = plan
    context.orchestration_state = plan.materialize_state()

    agent = LLMAgent(
        AgentDefinition(
            name="test-llm-agent",
            description="Test LLM-backed agent.",
            system_prompt="You are a test LLM agent.",
            model="mock-gpt",
        )
    )

    response = await agent.run(context)

    assert response.output == "Generated answer."
    assert len(gateway.requests) == 1

    state = context.orchestration_state

    assert state.current_step_index is None

    step = state.steps[0]
    assert step.step_id == "answer"
    assert step.status is OrchestrationStepStatus.COMPLETED
    assert step.tool_round == 0

    result = state.get_completed_step_result("answer")
    assert result.output == "Generated answer."


@pytest.mark.asyncio
async def test_llm_agent_persists_redacted_output_in_orchestration_state() -> None:
    definition = AgentDefinition(
        name="production-llm-agent",
        description="Production LLM agent.",
        system_prompt="You are a production LLM agent.",
        model="mock-gpt",
    )

    context, gateway = make_context(
        request=AgentRequest(
            input="Generate the answer.",
            session_id="session-output-redaction",
        ),
    )

    raw_output = "The answer contains secret=abc123."
    safe_output = "The answer contains ********."

    context._output_evaluator = lambda text: (
        OutputGovernanceDecision(
            allowed=True,
            redacted_output=safe_output,
            policy_id="policy-output",
            policy_version="v3",
            governance_metadata={
                "blocked": False,
                "redacted": True,
                "rule_type": "redact_output_pattern",
            },
        )
        if text == raw_output
        else OutputGovernanceDecision(
            allowed=True,
            redacted_output=text,
        )
    )

    context.orchestration_plan = OrchestrationPlan(
        steps=(
            OrchestrationStep(
                step_id="answer",
                step_index=0,
                name="Produce answer",
                status=OrchestrationStepStatus.PENDING,
                completion_policy=OrchestrationStepCompletionPolicy.ON_AGENT_RESPONSE,
            ),
        )
    )
    context.orchestration_state = context.orchestration_plan.materialize_state()

    # Make the fake provider return the sensitive raw output.
    async def route_chat(request: dict[str, Any]) -> dict[str, Any]:
        gateway.requests.append(request)
        return {
            "provider": "fake",
            "model": request["model"],
            "reply": raw_output,
            "usage": {
                "prompt_tokens": 10,
                "completion_tokens": 5,
                "total_tokens": 15,
            },
        }

    gateway.route_chat = route_chat

    agent = LLMAgent(definition)

    response = await agent.run(context)

    assert response.output == safe_output
    assert raw_output not in response.output

    step_result = context.orchestration_state.get_completed_step_result("answer")

    assert step_result is not None
    assert step_result.output == safe_output
    assert raw_output not in step_result.output


@pytest.mark.asyncio
async def test_llm_agent_blocks_output_without_leaking_raw_text() -> None:
    definition = AgentDefinition(
        name="production-llm-agent",
        description="Production LLM agent.",
        system_prompt="You are a production LLM agent.",
        model="mock-gpt",
    )

    context, gateway = make_context(
        request=AgentRequest(
            input="Generate the answer.",
            session_id="session-output-block",
        ),
    )

    raw_output = "FORBIDDEN secret=TOP_SECRET_VALUE"

    context._output_evaluator = lambda text: OutputGovernanceDecision(
        allowed=False,
        redacted_output="",
        policy_id="policy-output",
        policy_version="v9",
        governance_metadata={
            "blocked": True,
            "rule_type": "blocked_output_pattern",
        },
    )

    context.orchestration_plan = OrchestrationPlan(
        steps=(
            OrchestrationStep(
                step_id="answer",
                step_index=0,
                name="Produce answer",
                status=OrchestrationStepStatus.PENDING,
                completion_policy=OrchestrationStepCompletionPolicy.ON_AGENT_RESPONSE,
            ),
        )
    )
    context.orchestration_state = context.orchestration_plan.materialize_state()

    async def route_chat(request: dict[str, Any]) -> dict[str, Any]:
        gateway.requests.append(request)
        return {
            "provider": "fake",
            "model": request["model"],
            "reply": raw_output,
            "usage": {
                "prompt_tokens": 10,
                "completion_tokens": 5,
                "total_tokens": 15,
            },
        }

    gateway.route_chat = route_chat

    agent = LLMAgent(definition)

    with pytest.raises(
        AgentOutputPolicyError,
        match=r"policy-output.*version v9",
    ) as exc_info:
        await agent.run(context)

    assert raw_output not in str(exc_info.value)

    step = context.orchestration_state.steps[0]
    assert step.status is OrchestrationStepStatus.FAILED

    step_result = context.orchestration_state.get_step_result("answer")

    if step_result is not None:
        assert raw_output not in str(step_result.output)


@pytest.mark.asyncio
async def test_llm_agent_starts_first_orchestration_step() -> None:
    from ai_platform.agents.orchestration import (
        OrchestrationPlan,
        OrchestrationStep,
        OrchestrationStepStatus,
    )

    context, _ = make_context()

    context.orchestration_plan = OrchestrationPlan(
        steps=(
            OrchestrationStep(
                step_id="step-1",
                step_index=0,
                name="First step",
                status=OrchestrationStepStatus.PENDING,
            ),
        )
    )
    context.orchestration_state = context.orchestration_plan.materialize_state()

    agent = LLMAgent(
        AgentDefinition(
            name="test-llm-agent",
            description="Test LLM-backed agent.",
            system_prompt="You are a test LLM agent.",
            model="mock-gpt",
        )
    )

    step_index = await agent._start_orchestration_step(context)

    assert step_index == 0
    assert context.orchestration_state.current_step_index == 0
    assert context.orchestration_state.current_step is not None
    assert context.orchestration_state.current_step.status is OrchestrationStepStatus.RUNNING


@pytest.mark.asyncio
async def test_llm_agent_persists_orchestration_step_as_running() -> None:
    from app.control_plane.agent_run_steps.in_memory import (
        InMemoryAgentRunStepsRepository,
    )
    from app.control_plane.agent_run_steps.models import AgentRunStepStatus
    from ai_platform.agents.orchestration import (
        OrchestrationPlan,
        OrchestrationStep,
        OrchestrationStepStatus,
    )

    repository = InMemoryAgentRunStepsRepository()
    context, _ = make_context(
        run_id="run-durable-step",
        agent_run_steps_repository_factory=lambda: repository,
    )

    context.orchestration_plan = OrchestrationPlan(
        steps=(
            OrchestrationStep(
                step_id="step-1",
                step_index=0,
                name="First step",
                status=OrchestrationStepStatus.PENDING,
                metadata={"phase": "test", "custom": "preserved"},
            ),
        )
    )
    context.orchestration_state = context.orchestration_plan.materialize_state()

    agent = LLMAgent(
        AgentDefinition(
            name="test-llm-agent",
            description="Test LLM-backed agent.",
            system_prompt="You are a test LLM agent.",
            model="mock-gpt",
        )
    )

    await agent._persist_orchestration_step_planned(
        context,
        context.orchestration_state.steps[0],
    )

    planned_step = repository.get(
        "run-durable-step",
        "step-1",
    )

    assert planned_step is not None
    assert planned_step.status is AgentRunStepStatus.PLANNED
    assert planned_step.metadata["phase"] == "test"
    assert planned_step.metadata["custom"] == "preserved"
    assert planned_step.metadata["runtime"] == {
        "phase": "plan",
        "decision": None,
        "decision_reason": None,
        "current_step_index": None,
        "iteration": 1,
    }

    await agent._start_orchestration_step(context)

    durable_step = repository.get(
        "run-durable-step",
        "step-1",
    )

    assert durable_step is not None
    assert durable_step.status is AgentRunStepStatus.RUNNING
    assert durable_step.step_index == 0
    assert durable_step.step_type == "model"
    assert durable_step.attempt == 1

    assert context.runtime_state.phase.value == "act"
    assert context.runtime_state.current_step_index == 0
    assert context.runtime_state.decision is None

    assert durable_step.metadata["phase"] == "test"
    assert durable_step.metadata["custom"] == "preserved"
    assert durable_step.metadata["runtime"] == {
        "phase": "act",
        "decision": None,
        "decision_reason": None,
        "current_step_index": 0,
        "iteration": 1,
    }


@pytest.mark.asyncio
async def test_llm_agent_does_not_start_step_without_orchestration_plan() -> None:
    context, _ = make_context()

    agent = LLMAgent(
        AgentDefinition(
            name="test-llm-agent",
            description="Test LLM-backed agent.",
            system_prompt="You are a test LLM agent.",
            model="mock-gpt",
        )
    )

    step_index = await agent._start_orchestration_step(context)

    assert step_index is None
    assert context.orchestration_state.current_step_index is None
    assert context.orchestration_state.current_step is None


@pytest.mark.asyncio
async def test_llm_agent_reuses_current_orchestration_step() -> None:
    from ai_platform.agents.orchestration import (
        OrchestrationPlan,
        OrchestrationStep,
        OrchestrationStepStatus,
    )

    context, _ = make_context()

    context.orchestration_plan = OrchestrationPlan(
        steps=(
            OrchestrationStep(
                step_id="step-1",
                step_index=0,
                name="First step",
                status=OrchestrationStepStatus.PENDING,
            ),
            OrchestrationStep(
                step_id="step-2",
                step_index=1,
                name="Second step",
                status=OrchestrationStepStatus.PENDING,
            ),
        )
    )
    context.orchestration_state = context.orchestration_plan.materialize_state()
    context.orchestration_state.start_step(0)

    agent = LLMAgent(
        AgentDefinition(
            name="test-llm-agent",
            description="Test LLM-backed agent.",
            system_prompt="You are a test LLM agent.",
            model="mock-gpt",
        )
    )

    step_index = await agent._start_orchestration_step(context)

    assert step_index == 0
    assert context.orchestration_state.current_step_index == 0


@pytest.mark.asyncio
async def test_llm_agent_completes_orchestration_step() -> None:
    from ai_platform.agents.orchestration import (
        OrchestrationPlan,
        OrchestrationStep,
        OrchestrationStepStatus,
    )

    from app.control_plane.agent_run_steps.in_memory import (
        InMemoryAgentRunStepsRepository,
    )
    from app.control_plane.agent_run_steps.models import AgentRunStepStatus

    repository = InMemoryAgentRunStepsRepository()
    context, _ = make_context(
        run_id="run-durable-complete",
        agent_run_steps_repository_factory=lambda: repository,
    )

    context.orchestration_plan = OrchestrationPlan(
        steps=(
            OrchestrationStep(
                step_id="step-1",
                step_index=0,
                name="First step",
                status=OrchestrationStepStatus.PENDING,
            ),
        )
    )
    context.orchestration_state = context.orchestration_plan.materialize_state()

    agent = LLMAgent(
        AgentDefinition(
            name="test-llm-agent",
            description="Test LLM-backed agent.",
            system_prompt="You are a test LLM agent.",
            model="mock-gpt",
        )
    )

    step_index = await agent._start_orchestration_step(context)

    await agent._complete_orchestration_step(
        context,
        step_index,
        tool_round=2,
    )

    step = context.orchestration_state.steps[0]

    assert step.status is OrchestrationStepStatus.COMPLETED
    assert step.tool_round == 2
    assert context.orchestration_state.current_step_index == 0

    assert context.runtime_state.phase.value == "evaluate"
    assert context.runtime_state.current_step_index == 0
    assert context.runtime_state.decision is None

    durable_step = repository.get(
        "run-durable-complete",
        "step-1",
    )

    assert durable_step is not None
    assert durable_step.status is AgentRunStepStatus.COMPLETED
    assert durable_step.completed_at is not None
    assert durable_step.metadata["runtime"] == {
        "phase": "evaluate",
        "decision": None,
        "decision_reason": None,
        "current_step_index": 0,
        "iteration": 1,
    }


@pytest.mark.asyncio
async def test_llm_agent_keeps_durable_steps_consistent_across_runtime_replanning() -> None:
    class ReplanningPlanProvider:
        def __init__(self) -> None:
            self.plans: list[OrchestrationPlan] = []

        def build_plan(
            self,
            context: AgentExecutionContext,
        ) -> OrchestrationPlan:
            plan_number = len(self.plans) + 2
            plan = OrchestrationPlan(
                steps=(
                    OrchestrationStep(
                        step_id=f"answer-{plan_number}",
                        step_index=0,
                        name="Produce answer",
                        status=OrchestrationStepStatus.PENDING,
                        completion_policy=(OrchestrationStepCompletionPolicy.ON_AGENT_RESPONSE),
                    ),
                )
            )
            self.plans.append(plan)
            return plan

    class ReplanningDecisionProvider:
        def __init__(self) -> None:
            self.results: list[AgentRuntimeDecision] = []
            self.evaluations: list[AgentRuntimeEvaluationSnapshot] = []

        def evaluate(
            self,
            context: AgentExecutionContext,
            evaluation: AgentRuntimeEvaluationSnapshot,
        ):
            self.evaluations.append(evaluation)

            decision = (
                AgentRuntimeDecision.CONTINUE if not self.results else AgentRuntimeDecision.STOP
            )
            self.results.append(decision)

            reason = (
                "iteration_budget_remaining"
                if decision is AgentRuntimeDecision.CONTINUE
                else "iteration_budget_exhausted"
            )

            from ai_platform.agents.decision_provider import (
                AgentRuntimeDecisionReason,
                AgentRuntimeDecisionResult,
            )

            return AgentRuntimeDecisionResult(
                decision=decision,
                reason=AgentRuntimeDecisionReason(reason),
            )

    repository = InMemoryAgentRunStepsRepository()
    plan_provider = ReplanningPlanProvider()
    decision_provider = ReplanningDecisionProvider()
    observer = FakeAgentExecutionObserver()

    context, _ = make_context(
        run_id="run-runtime-replanning-steps",
        agent_run_steps_repository_factory=lambda: repository,
    )

    initial_plan = OrchestrationPlan(
        steps=(
            OrchestrationStep(
                step_id="answer-1",
                step_index=0,
                name="Produce answer",
                status=OrchestrationStepStatus.PENDING,
                completion_policy=(OrchestrationStepCompletionPolicy.ON_AGENT_RESPONSE),
            ),
        )
    )
    context.install_orchestration_plan(initial_plan)
    context.plan_provider = plan_provider
    context.decision_provider = decision_provider

    agent = LLMAgent(
        AgentDefinition(
            name="test-llm-agent",
            description="Test LLM-backed agent.",
            system_prompt="You are a test LLM agent.",
            model="mock-gpt",
        ),
        observer=observer,
    )

    response = await agent.run(context)

    assert response.output == "Generated answer."

    assert decision_provider.results == [
        AgentRuntimeDecision.CONTINUE,
        AgentRuntimeDecision.STOP,
    ]
    assert [evaluation.iteration for evaluation in decision_provider.evaluations] == [
        1,
        2,
    ]

    assert len(plan_provider.plans) == 1
    assert plan_provider.plans[0].steps[0].step_id == "answer-2"

    first_step = repository.get(
        "run-runtime-replanning-steps",
        "iteration-1:answer-1",
    )
    second_step = repository.get(
        "run-runtime-replanning-steps",
        "iteration-2:answer-2",
    )

    assert first_step is not None
    assert second_step is not None

    assert first_step.status is AgentRunStepStatus.COMPLETED
    assert first_step.completed_at is not None
    assert first_step.attempt == 1

    assert second_step.status is AgentRunStepStatus.COMPLETED
    assert second_step.completed_at is not None
    assert second_step.attempt == 1

    assert context.runtime_state.phase is AgentRuntimePhase.EVALUATE
    assert context.runtime_state.iteration == 2
    assert context.runtime_state.decision is AgentRuntimeDecision.STOP
    assert context.runtime_state.current_step_index == 0

    assert first_step.metadata["runtime"]["iteration"] == 1
    assert second_step.metadata["runtime"]["iteration"] == 2

    assert first_step.status is not AgentRunStepStatus.RUNNING
    assert second_step.status is not AgentRunStepStatus.RUNNING

    orchestration_events = [
        event
        for event in observer.events
        if event.event_type
        in {
            AgentExecutionEventType.ORCHESTRATION_STEP_STARTED,
            AgentExecutionEventType.ORCHESTRATION_STEP_COMPLETED,
            AgentExecutionEventType.RUNTIME_DECISION,
        }
    ]

    assert [(event.event_type, event.step_id) for event in orchestration_events] == [
        (AgentExecutionEventType.ORCHESTRATION_STEP_STARTED, "iteration-1:answer-1"),
        (AgentExecutionEventType.ORCHESTRATION_STEP_COMPLETED, "iteration-1:answer-1"),
        (AgentExecutionEventType.RUNTIME_DECISION, None),
        (AgentExecutionEventType.ORCHESTRATION_STEP_STARTED, "iteration-2:answer-2"),
        (AgentExecutionEventType.ORCHESTRATION_STEP_COMPLETED, "iteration-2:answer-2"),
        (AgentExecutionEventType.RUNTIME_DECISION, None),
    ]

    first_runtime_decision = orchestration_events[2]
    assert first_runtime_decision.metadata == {
        "decision": "continue",
        "reason": "iteration_budget_remaining",
        "iteration": 1,
    }

    second_runtime_decision = orchestration_events[5]
    assert second_runtime_decision.metadata == {
        "decision": "stop",
        "reason": "iteration_budget_exhausted",
        "iteration": 2,
    }


@pytest.mark.asyncio
async def test_llm_agent_cancellation_stops_before_runtime_replanned_step() -> None:
    class ReplanningPlanProvider:
        def __init__(self) -> None:
            self.plans: list[OrchestrationPlan] = []

        def build_plan(
            self,
            context: AgentExecutionContext,
        ) -> OrchestrationPlan:
            plan_number = len(self.plans) + 2
            plan = OrchestrationPlan(
                steps=(
                    OrchestrationStep(
                        step_id=f"answer-{plan_number}",
                        step_index=0,
                        name="Produce answer",
                        status=OrchestrationStepStatus.PENDING,
                        completion_policy=(OrchestrationStepCompletionPolicy.ON_AGENT_RESPONSE),
                    ),
                )
            )
            self.plans.append(plan)
            return plan

    class ReplanningDecisionProvider:
        def __init__(self, cancellation_requested: asyncio.Event) -> None:
            self.cancellation_requested = cancellation_requested
            self.results: list[AgentRuntimeDecision] = []

        def evaluate(
            self,
            context: AgentExecutionContext,
            evaluation: AgentRuntimeEvaluationSnapshot,
        ):
            decision = (
                AgentRuntimeDecision.CONTINUE if not self.results else AgentRuntimeDecision.STOP
            )
            self.results.append(decision)

            if decision is AgentRuntimeDecision.CONTINUE:
                self.cancellation_requested.set()

            from ai_platform.agents.decision_provider import (
                AgentRuntimeDecisionReason,
                AgentRuntimeDecisionResult,
            )

            return AgentRuntimeDecisionResult(
                decision=decision,
                reason=AgentRuntimeDecisionReason(
                    "iteration_budget_remaining"
                    if decision is AgentRuntimeDecision.CONTINUE
                    else "iteration_budget_exhausted"
                ),
            )

    repository = InMemoryAgentRunStepsRepository()
    cancellation_requested = asyncio.Event()
    plan_provider = ReplanningPlanProvider()
    decision_provider = ReplanningDecisionProvider(cancellation_requested)

    context, gateway = make_context(
        run_id="run-runtime-replanning-cancellation",
        agent_run_steps_repository_factory=lambda: repository,
    )

    initial_plan = OrchestrationPlan(
        steps=(
            OrchestrationStep(
                step_id="answer-1",
                step_index=0,
                name="Produce answer",
                status=OrchestrationStepStatus.PENDING,
                completion_policy=OrchestrationStepCompletionPolicy.ON_AGENT_RESPONSE,
            ),
        )
    )
    context.install_orchestration_plan(initial_plan)
    context.plan_provider = plan_provider
    context.decision_provider = decision_provider
    context.cancellation_requested = cancellation_requested

    agent = LLMAgent(
        AgentDefinition(
            name="test-llm-agent",
            description="Test LLM-backed agent.",
            system_prompt="You are a test LLM agent.",
            model="mock-gpt",
        )
    )

    with pytest.raises(
        asyncio.CancelledError,
        match="Agent execution cancellation requested.",
    ):
        await agent.run(context)

    first_step = repository.get(
        "run-runtime-replanning-cancellation",
        "iteration-1:answer-1",
    )

    assert first_step is not None
    assert first_step.status is AgentRunStepStatus.COMPLETED

    assert plan_provider.plans == []
    assert decision_provider.results == [AgentRuntimeDecision.CONTINUE]
    assert len(gateway.requests) == 1

    assert context.orchestration_state.steps[0].step_id == "iteration-1:answer-1"
    assert context.orchestration_state.steps[0].status is OrchestrationStepStatus.COMPLETED


@pytest.mark.asyncio
async def test_llm_agent_completion_is_noop_without_orchestration_step() -> None:
    context, _ = make_context()

    agent = LLMAgent(
        AgentDefinition(
            name="test-llm-agent",
            description="Test LLM-backed agent.",
            system_prompt="You are a test LLM agent.",
            model="mock-gpt",
        )
    )

    await agent._complete_orchestration_step(
        context,
        None,
        tool_round=2,
    )

    assert context.orchestration_state.current_step_index is None
    assert context.orchestration_state.steps == []


def test_llm_agent_advances_to_next_orchestration_step() -> None:
    from ai_platform.agents.orchestration import (
        OrchestrationStepStatus,
    )
    from ai_platform.agents.plans import build_enterprise_rag_analyst_plan

    context, _ = make_context()

    context.orchestration_plan = build_enterprise_rag_analyst_plan()
    context.orchestration_state = context.orchestration_plan.materialize_state()

    agent = LLMAgent(
        AgentDefinition(
            name="enterprise-rag-analyst",
            description="Enterprise RAG analyst.",
            system_prompt="You are an enterprise RAG analyst.",
            model="mock-gpt",
        )
    )

    context.orchestration_state.start_step(0)
    context.orchestration_state.complete_step(tool_round=1)

    next_step_index = agent._advance_orchestration_step(context)

    assert next_step_index == 1
    assert context.orchestration_state.current_step_index == 1

    assert context.orchestration_state.steps[0].status is (OrchestrationStepStatus.COMPLETED)
    assert context.orchestration_state.steps[0].tool_round == 1

    assert context.orchestration_state.steps[1].status is (OrchestrationStepStatus.RUNNING)
    assert context.orchestration_state.steps[1].tool_round is None


def test_llm_agent_finishes_orchestration_without_advancing_past_final_step() -> None:
    from ai_platform.agents.orchestration import (
        OrchestrationPlan,
        OrchestrationStep,
        OrchestrationStepStatus,
    )

    context, _ = make_context()

    context.orchestration_plan = OrchestrationPlan(
        steps=(
            OrchestrationStep(
                step_id="final",
                step_index=0,
                name="Final step",
                status=OrchestrationStepStatus.PENDING,
            ),
        )
    )
    context.orchestration_state = context.orchestration_plan.materialize_state()

    agent = LLMAgent(
        AgentDefinition(
            name="test-llm-agent",
            description="Test LLM-backed agent.",
            system_prompt="You are a test LLM agent.",
            model="mock-gpt",
        )
    )

    context.orchestration_state.start_step(0)
    context.orchestration_state.complete_step(tool_round=2)

    next_step_index = agent._advance_orchestration_step(context)

    assert next_step_index is None
    assert context.orchestration_state.current_step_index is None
    assert context.orchestration_state.steps[0].status is (OrchestrationStepStatus.COMPLETED)


def test_llm_agent_does_not_advance_without_orchestration_plan() -> None:
    context, _ = make_context()

    agent = LLMAgent(
        AgentDefinition(
            name="test-llm-agent",
            description="Test LLM-backed agent.",
            system_prompt="You are a test LLM agent.",
            model="mock-gpt",
        )
    )

    assert agent._advance_orchestration_step(context) is None
    assert context.orchestration_state.current_step_index is None


@pytest.mark.asyncio
async def test_llm_agent_implements_agent_contract() -> None:
    from ai_platform.agents.contracts import Agent

    definition = AgentDefinition(
        name="production-llm-agent",
        description="Production LLM agent.",
        system_prompt="You are a production LLM agent.",
        model="mock-gpt",
    )

    agent: Agent = LLMAgent(definition)

    assert agent.definition is definition


@pytest.mark.asyncio
async def test_llm_agent_returns_agent_response() -> None:
    definition = AgentDefinition(
        name="production-llm-agent",
        description="Production LLM agent.",
        system_prompt="You are a production LLM agent.",
        model="mock-gpt",
    )

    gateway = FakeLLMGateway()

    llm_context = AgentLLMContext(
        gateway,
        AgentLLMConfig(
            model=definition.model,
            system_prompt=definition.system_prompt,
        ),
    )

    tools = AgentToolContext(
        InMemoryToolRegistry(),
        definition,
    )

    context = AgentExecutionContext(
        AgentRequest(
            input="Explain RAG.",
            user_id="user-123",
            session_id="session-456",
        ),
        tools=tools,
        llm=llm_context,
    )

    agent = LLMAgent(definition)

    response = await agent.run(context)
    assert response.metadata["tool_rounds"] == 0
    assert response == AgentResponse(
        agent_name="production-llm-agent",
        output="Generated answer.",
        session_id="session-456",
        metadata={
            "provider": "fake",
            "model": "mock-gpt",
            "usage": {
                "prompt_tokens": 10,
                "completion_tokens": 5,
                "total_tokens": 15,
            },
            "tool_rounds": 0,
        },
    )


@pytest.mark.asyncio
async def test_llm_agent_publishes_orchestration_step_result() -> None:
    definition = AgentDefinition(
        name="production-llm-agent",
        description="Production LLM agent.",
        system_prompt="You are a production LLM agent.",
        model="mock-gpt",
    )

    context, _ = make_context()

    plan = OrchestrationPlan(
        steps=(
            OrchestrationStep(
                step_id="step-0",
                step_index=0,
                name="Test step",
                status=OrchestrationStepStatus.PENDING,
            ),
        )
    )

    context.orchestration_plan = plan
    context.orchestration_state = plan.materialize_state()

    agent = LLMAgent(definition)

    step_index = await agent._start_orchestration_step(context)

    response = AgentResponse(
        agent_name="production-llm-agent",
        output="Generated answer.",
        session_id=context.session_id,
        metadata={
            "provider": "fake",
            "model": "mock-gpt",
            "tool_rounds": 1,
        },
    )

    agent._set_orchestration_step_result(
        context,
        step_index,
        response,
    )

    result = context.orchestration_state.get_step_result("step-0")

    assert result is not None
    assert result.step_id == "step-0"
    assert result.output == "Generated answer."
    assert result.metadata == response.metadata


def test_llm_agent_does_not_publish_result_without_orchestration_step() -> None:
    definition = AgentDefinition(
        name="production-llm-agent",
        description="Production LLM agent.",
        system_prompt="You are a production LLM agent.",
        model="mock-gpt",
    )

    context, _ = make_context()

    agent = LLMAgent(definition)

    response = AgentResponse(
        agent_name="production-llm-agent",
        output="Generated answer.",
    )

    agent._set_orchestration_step_result(
        context,
        None,
        response,
    )

    assert context.orchestration_state.step_results == {}


@pytest.mark.asyncio
async def test_llm_agent_sends_canonical_messages() -> None:
    definition = AgentDefinition(
        name="production-llm-agent",
        description="Production LLM agent.",
        system_prompt="You are a production LLM agent.",
        model="mock-gpt",
    )

    gateway = FakeLLMGateway()

    llm_context = AgentLLMContext(
        gateway,
        AgentLLMConfig(
            model=definition.model,
            system_prompt=definition.system_prompt,
        ),
    )

    tools = AgentToolContext(
        InMemoryToolRegistry(),
        definition,
    )

    context = AgentExecutionContext(
        AgentRequest(input="Explain RAG."),
        tools=tools,
        llm=llm_context,
    )

    agent = LLMAgent(definition)

    await agent.run(context)

    assert gateway.requests[0]["messages"] == [
        {
            "role": "system",
            "content": "You are a production LLM agent.",
        },
        {
            "role": "user",
            "content": "Explain RAG.",
        },
    ]


class FakeRAGTool:
    def __init__(self, name: str = "rag.search") -> None:
        self._definition = ToolDefinition(
            name=name,
            description="A test RAG search tool.",
        )
        self.execute_count = 0

    @property
    def definition(self) -> ToolDefinition:
        return self._definition

    async def execute(self, arguments):
        self.execute_count += 1

        return {
            "query": arguments["query"],
            "retrieved_count": 2,
            "results": [
                {
                    "chunk_id": "chunk-rag-001",
                    "document_id": "doc-rag-001",
                    "content": "Sensitive enterprise context.",
                    "score": 0.91,
                    "retrieval_score": 0.72,
                    "reranker_score": 0.91,
                    "metadata": {
                        "classification": "confidential",
                    },
                },
                {
                    "chunk_id": "chunk-rag-002",
                    "document_id": "doc-rag-002",
                    "content": "Another sensitive context.",
                    "score": 0.83,
                    "retrieval_score": 0.61,
                    "reranker_score": 0.88,
                    "metadata": {
                        "classification": "restricted",
                    },
                },
            ],
        }


class FakeToolCallingLLMGateway:
    def __init__(self, tool_name: str = "search") -> None:
        self.requests: list[dict[str, Any]] = []
        self.tool_name = tool_name

    async def route_chat(
        self,
        request: dict[str, Any],
    ) -> dict[str, Any]:
        self.requests.append(request)

        if len(self.requests) == 1:
            return {
                "provider": "fake",
                "model": request["model"],
                "reply": "",
                "usage": {
                    "prompt_tokens": 10,
                    "completion_tokens": 5,
                    "total_tokens": 15,
                },
                "tool_calls": [
                    AgentToolCall(
                        call_id="call-123",
                        name=self.tool_name,
                        arguments={
                            "query": "RAG",
                        },
                    ),
                ],
            }

        return {
            "provider": "fake",
            "model": request["model"],
            "reply": "RAG retrieves relevant context for generation.",
            "usage": {
                "prompt_tokens": 25,
                "completion_tokens": 10,
                "total_tokens": 35,
            },
        }


class OwnershipLossDuringLLMGateway(FakeToolCallingLLMGateway):
    def __init__(self, ownership_lost: asyncio.Event) -> None:
        super().__init__()
        self.ownership_lost = ownership_lost

    async def route_chat(
        self,
        request: dict[str, Any],
    ) -> dict[str, Any]:
        response = await super().route_chat(request)
        self.ownership_lost.set()
        return response


@pytest.mark.asyncio
async def test_llm_agent_stops_before_llm_when_execution_ownership_is_already_lost() -> None:
    ownership_lost = asyncio.Event()
    ownership_lost.set()

    definition = AgentDefinition(
        name="production-llm-agent",
        description="Production LLM agent.",
        system_prompt="You are a production LLM agent.",
        model="mock-gpt",
    )

    gateway = FakeLLMGateway()

    llm_context = AgentLLMContext(
        gateway,
        AgentLLMConfig(
            model=definition.model,
            system_prompt=definition.system_prompt,
        ),
    )

    tools = AgentToolContext(
        InMemoryToolRegistry(),
        definition,
    )

    context = AgentExecutionContext(
        AgentRequest(
            input="Find information about RAG.",
            user_id="user-123",
        ),
        tools=tools,
        llm=llm_context,
        execution_ownership_lost=ownership_lost,
    )

    agent = LLMAgent(definition)

    with pytest.raises(
        AgentExecutionOwnershipLostError,
        match="lost durable run ownership",
    ):
        await agent.run(context)

    assert gateway.requests == []


@pytest.mark.asyncio
async def test_llm_agent_does_not_execute_tool_after_ownership_is_lost_during_llm_call() -> None:
    ownership_lost = asyncio.Event()

    definition = AgentDefinition(
        name="production-llm-agent",
        description="Production LLM agent.",
        system_prompt="You are a production LLM agent.",
        model="mock-gpt",
        tool_names=("search",),
    )

    gateway = OwnershipLossDuringLLMGateway(ownership_lost)

    llm_context = AgentLLMContext(
        gateway,
        AgentLLMConfig(
            model=definition.model,
            system_prompt=definition.system_prompt,
        ),
    )

    tool = FakeTool(name="search")

    registry = InMemoryToolRegistry()
    await registry.register(tool)

    tools = AgentToolContext(
        registry,
        definition,
    )

    context = AgentExecutionContext(
        AgentRequest(
            input="Find information about RAG.",
            user_id="user-123",
        ),
        tools=tools,
        llm=llm_context,
        execution_ownership_lost=ownership_lost,
    )

    agent = LLMAgent(definition)

    with pytest.raises(
        AgentExecutionOwnershipLostError,
        match="lost durable run ownership",
    ):
        await agent.run(context)

    assert len(gateway.requests) == 1
    assert tool.execution_count == 0


@pytest.mark.asyncio
async def test_llm_agent_accumulates_tool_call_and_tool_result_messages() -> None:
    definition = AgentDefinition(
        name="production-llm-agent",
        description="Production LLM agent.",
        system_prompt="You are a production LLM agent.",
        model="mock-gpt",
        tool_names=("search",),
    )

    gateway = FakeLLMGateway()

    llm_context = AgentLLMContext(
        gateway,
        AgentLLMConfig(
            model=definition.model,
            system_prompt=definition.system_prompt,
        ),
    )

    tools = AgentToolContext(
        InMemoryToolRegistry(),
        definition,
    )

    context = AgentExecutionContext(
        AgentRequest(
            input="Find information about RAG.",
            user_id="user-123",
        ),
        tools=tools,
        llm=llm_context,
    )

    agent = LLMAgent(definition)

    tool_call = AgentToolCall(
        call_id="call-123",
        name="search",
        arguments={
            "query": "RAG",
        },
    )

    messages = list(context.build_llm_messages())

    await agent._accumulate_tool_call_messages(
        messages,
        context,
        (tool_call,),
        tool_round=1,
        assistant_content="I will search for that.",
    )

    assert messages == [
        system_message("You are a production LLM agent."),
        user_message("Find information about RAG."),
        assistant_tool_call_message(
            tool_calls=(tool_call,),
            content="I will search for that.",
        ),
        tool_result_message(
            call_id="call-123",
            tool_name="search",
            error="Tool not found: search",
        ),
    ]


@pytest.mark.asyncio
async def test_llm_agent_reinvokes_llm_after_tool_execution() -> None:
    definition = AgentDefinition(
        name="production-llm-agent",
        description="Production LLM agent.",
        system_prompt="You are a production LLM agent.",
        model="mock-gpt",
        tool_names=("search",),
    )

    gateway = FakeToolCallingLLMGateway()

    llm_context = AgentLLMContext(
        gateway,
        AgentLLMConfig(
            model=definition.model,
            system_prompt=definition.system_prompt,
        ),
    )

    tools = AgentToolContext(
        InMemoryToolRegistry(),
        definition,
    )

    context = AgentExecutionContext(
        AgentRequest(
            input="Find information about RAG.",
            user_id="user-123",
            session_id="session-456",
        ),
        tools=tools,
        llm=llm_context,
    )

    agent = LLMAgent(definition)

    response = await agent.run(context)

    assert response == AgentResponse(
        agent_name="production-llm-agent",
        output="RAG retrieves relevant context for generation.",
        session_id="session-456",
        metadata={
            "provider": "fake",
            "model": "mock-gpt",
            "usage": {
                "prompt_tokens": 25,
                "completion_tokens": 10,
                "total_tokens": 35,
            },
            "tool_rounds": 1,
        },
    )

    assert len(gateway.requests) == 2
    assert response.metadata["tool_rounds"] == 1

    assert gateway.requests[0]["messages"] == [
        {
            "role": "system",
            "content": "You are a production LLM agent.",
        },
        {
            "role": "user",
            "content": "Find information about RAG.",
        },
    ]

    assert gateway.requests[1]["messages"] == [
        {
            "role": "system",
            "content": "You are a production LLM agent.",
        },
        {
            "role": "user",
            "content": "Find information about RAG.",
        },
        {
            "role": "assistant",
            "content": "Tool call requested.",
            "tool_calls": [
                {
                    "call_id": "call-123",
                    "name": "search",
                    "arguments": {
                        "query": "RAG",
                    },
                },
            ],
        },
        {
            "role": "tool",
            "content": '{"call_id": "call-123", "error": "Tool not found: search", "success": false, "tool_name": "search"}',
            "tool_call_id": "call-123",
            "tool_name": "search",
        },
    ]


@pytest.mark.asyncio
async def test_llm_agent_stops_after_max_tool_rounds() -> None:
    definition = AgentDefinition(
        name="production-llm-agent",
        description="Production LLM agent.",
        system_prompt="You are a production LLM agent.",
        model="mock-gpt",
        tool_names=("search",),
    )

    gateway = FakeMultiRoundToolCallingLLMGateway()

    llm_context = AgentLLMContext(
        gateway,
        AgentLLMConfig(
            model=definition.model,
            system_prompt=definition.system_prompt,
        ),
    )

    tools = AgentToolContext(
        InMemoryToolRegistry(),
        definition,
    )

    context = AgentExecutionContext(
        AgentRequest(
            input="Find information about RAG.",
            user_id="user-123",
            session_id="session-456",
        ),
        tools=tools,
        llm=llm_context,
    )

    agent = LLMAgent(definition)

    with pytest.raises(
        AgentToolLoopLimitError,
        match="Agent 'production-llm-agent' exceeded the maximum tool-call rounds \\(3\\)",
    ):
        await agent.run(context)

    assert len(gateway.requests) == 4

    assert gateway.requests[0]["messages"] == [
        {
            "role": "system",
            "content": "You are a production LLM agent.",
        },
        {
            "role": "user",
            "content": "Find information about RAG.",
        },
    ]

    assert gateway.requests[1]["messages"] == [
        {
            "role": "system",
            "content": "You are a production LLM agent.",
        },
        {
            "role": "user",
            "content": "Find information about RAG.",
        },
        {
            "role": "assistant",
            "content": "Tool call requested.",
            "tool_calls": [
                {
                    "call_id": "call-1",
                    "name": "search",
                    "arguments": {
                        "query": "round-1",
                    },
                },
            ],
        },
        {
            "role": "tool",
            "content": '{"call_id": "call-1", "error": "Tool not found: search", "success": false, "tool_name": "search"}',
            "tool_call_id": "call-1",
            "tool_name": "search",
        },
    ]

    assert gateway.requests[2]["messages"] == [
        {
            "role": "system",
            "content": "You are a production LLM agent.",
        },
        {
            "role": "user",
            "content": "Find information about RAG.",
        },
        {
            "role": "assistant",
            "content": "Tool call requested.",
            "tool_calls": [
                {
                    "call_id": "call-1",
                    "name": "search",
                    "arguments": {
                        "query": "round-1",
                    },
                },
            ],
        },
        {
            "role": "tool",
            "content": '{"call_id": "call-1", "error": "Tool not found: search", "success": false, "tool_name": "search"}',
            "tool_call_id": "call-1",
            "tool_name": "search",
        },
        {
            "role": "assistant",
            "content": "Tool call requested.",
            "tool_calls": [
                {
                    "call_id": "call-2",
                    "name": "search",
                    "arguments": {
                        "query": "round-2",
                    },
                },
            ],
        },
        {
            "role": "tool",
            "content": '{"call_id": "call-2", "error": "Tool not found: search", "success": false, "tool_name": "search"}',
            "tool_call_id": "call-2",
            "tool_name": "search",
        },
    ]


class FakeMultiRoundToolCallingLLMGateway:
    def __init__(self) -> None:
        self.requests: list[dict[str, Any]] = []

    async def route_chat(
        self,
        request: dict[str, Any],
    ) -> dict[str, Any]:
        self.requests.append(request)

        round_number = len(self.requests)

        if round_number <= 4:
            return {
                "provider": "fake",
                "model": request["model"],
                "reply": "",
                "usage": {
                    "prompt_tokens": 10,
                    "completion_tokens": 5,
                    "total_tokens": 15,
                },
                "tool_calls": [
                    AgentToolCall(
                        call_id=f"call-{round_number}",
                        name="search",
                        arguments={
                            "query": f"round-{round_number}",
                        },
                    ),
                ],
            }

        return {
            "provider": "fake",
            "model": request["model"],
            "reply": "This response should never be reached.",
            "usage": {
                "prompt_tokens": 20,
                "completion_tokens": 5,
                "total_tokens": 25,
            },
        }


class FakeResumeContinuationLLMGateway:
    def __init__(self) -> None:
        self.requests: list[dict[str, Any]] = []

    async def route_chat(
        self,
        request: dict[str, Any],
    ) -> dict[str, Any]:
        self.requests.append(request)

        if len(self.requests) == 1:
            return {
                "provider": "fake",
                "model": request["model"],
                "reply": "",
                "usage": {
                    "prompt_tokens": 20,
                    "completion_tokens": 5,
                    "total_tokens": 25,
                },
                "tool_calls": [
                    AgentToolCall(
                        call_id="call-2",
                        name="search",
                        arguments={
                            "query": "follow-up",
                        },
                    ),
                ],
            }

        return {
            "provider": "fake",
            "model": request["model"],
            "reply": "Resumed execution completed successfully.",
            "usage": {
                "prompt_tokens": 30,
                "completion_tokens": 10,
                "total_tokens": 40,
            },
        }


class FakeAgentExecutionObserver:
    def __init__(self) -> None:
        self.events: list[AgentExecutionEvent] = []

    async def record(
        self,
        event: AgentExecutionEvent,
    ) -> None:
        self.events.append(event)


class FakeAgentCheckpointHandler:
    def __init__(self) -> None:
        self.checkpoints: list[AgentExecutionCheckpoint] = []

    async def save(
        self,
        checkpoint: AgentExecutionCheckpoint,
        *,
        lease_id: str | None = None,
    ) -> None:
        self.checkpoints.append(checkpoint)


class FailingAgentCheckpointHandler:
    def __init__(
        self,
        *,
        fail_position: AgentCheckpointPosition = AgentCheckpointPosition.AFTER_TOOL_EXECUTION,
    ) -> None:
        self.fail_position = fail_position
        self.attempted_checkpoints: list[AgentExecutionCheckpoint] = []

    async def save(
        self,
        checkpoint: AgentExecutionCheckpoint,
        *,
        lease_id: str | None = None,
    ) -> None:
        self.attempted_checkpoints.append(checkpoint)
        if checkpoint.position is self.fail_position:
            raise RuntimeError("checkpoint persistence failed")


@pytest.mark.asyncio
async def test_llm_agent_captures_checkpoint_after_tool_execution() -> None:
    definition = AgentDefinition(
        name="production-llm-agent",
        description="Production LLM agent",
        system_prompt="You are a production assistant.",
        model="gpt-test",
        tool_names=("search",),
    )

    checkpoint_handler = FakeAgentCheckpointHandler()
    gateway = FakeToolCallingLLMGateway()

    llm_context = AgentLLMContext(
        gateway,
        AgentLLMConfig(
            model=definition.model,
            system_prompt=definition.system_prompt,
        ),
    )

    tools = AgentToolContext(
        InMemoryToolRegistry(),
        definition,
    )

    context = AgentExecutionContext(
        AgentRequest(
            input="Find information about RAG.",
            user_id="user-123",
            session_id="session-456",
            metadata={"request_id": "request-789"},
        ),
        tools=tools,
        llm=llm_context,
        run_id="run-123",
    )

    agent = LLMAgent(
        definition,
        checkpoint_handler=checkpoint_handler,
    )

    await agent.run(context)

    assert len(checkpoint_handler.checkpoints) == 2

    before_checkpoint = checkpoint_handler.checkpoints[0]

    assert before_checkpoint.run_id == "run-123"
    assert before_checkpoint.agent_name == "production-llm-agent"
    assert before_checkpoint.session_id == "session-456"
    assert before_checkpoint.user_id == "user-123"
    assert before_checkpoint.tool_round == 1
    assert before_checkpoint.position is AgentCheckpointPosition.BEFORE_TOOL_EXECUTION
    assert before_checkpoint.metadata["request_id"] == "request-789"
    assert before_checkpoint.metadata["runtime"] == {
        "phase": AgentRuntimePhase.ACT.value,
        "decision": None,
        "decision_reason": None,
        "current_step_index": None,
        "iteration": 1,
    }
    assert before_checkpoint.messages[-1].role is AgentMessageRole.ASSISTANT
    assert "call-123" in before_checkpoint.messages[-1].content
    assert len(before_checkpoint.messages) == 3

    after_checkpoint = checkpoint_handler.checkpoints[1]

    assert after_checkpoint.run_id == "run-123"
    assert after_checkpoint.agent_name == "production-llm-agent"
    assert after_checkpoint.session_id == "session-456"
    assert after_checkpoint.user_id == "user-123"
    assert after_checkpoint.tool_round == 1
    assert after_checkpoint.position is AgentCheckpointPosition.AFTER_TOOL_EXECUTION
    assert after_checkpoint.metadata["request_id"] == "request-789"
    assert after_checkpoint.metadata["runtime"] == {
        "phase": AgentRuntimePhase.ACT.value,
        "decision": None,
        "decision_reason": None,
        "current_step_index": None,
        "iteration": 1,
    }

    assert after_checkpoint.messages[-1].role is AgentMessageRole.TOOL
    assert "call-123" in after_checkpoint.messages[-1].content
    assert len(after_checkpoint.messages) == 4


@pytest.mark.asyncio
async def test_llm_agent_checkpoint_preserves_rendered_memory_context() -> None:
    definition = AgentDefinition(
        name="production-memory-agent",
        description="Production memory-enabled LLM agent",
        system_prompt="You are a production assistant.",
        model="gpt-test",
        tool_names=("search",),
    )

    checkpoint_handler = FakeAgentCheckpointHandler()
    gateway = FakeToolCallingLLMGateway()

    llm_context = AgentLLMContext(
        gateway,
        AgentLLMConfig(
            model=definition.model,
            system_prompt=definition.system_prompt,
        ),
    )

    tools = AgentToolContext(
        InMemoryToolRegistry(),
        definition,
    )

    memory_item = MemoryItem(
        id="memory-001",
        memory_type="semantic",
        content="The production deployment uses the approved release workflow.",
        namespace="enterprise-agent",
        created_at=datetime.now(),
    )

    memory_context = MemoryContext(
        working=(),
        semantic=(memory_item,),
        episodic=(),
    )

    context = AgentExecutionContext(
        AgentRequest(
            input="How does the production deployment work?",
            user_id="user-123",
            session_id="session-456",
            memory_namespace="enterprise-agent",
        ),
        tools=tools,
        llm=llm_context,
        memory=memory_context,
        run_id="run-memory-checkpoint-1",
    )

    agent = LLMAgent(
        definition,
        checkpoint_handler=checkpoint_handler,
    )

    await agent.run(context)

    assert len(checkpoint_handler.checkpoints) == 2

    before_checkpoint = checkpoint_handler.checkpoints[0]

    memory_messages = [
        message
        for message in before_checkpoint.messages
        if message.role is AgentMessageRole.SYSTEM
        and "The following information was retrieved from agent memory." in message.content
    ]

    assert len(memory_messages) == 1
    assert (
        "The production deployment uses the approved release workflow."
        in memory_messages[0].content
    )

    after_checkpoint = checkpoint_handler.checkpoints[1]

    assert after_checkpoint.messages[: len(before_checkpoint.messages)] == (
        before_checkpoint.messages
    )

    serialized = before_checkpoint.to_dict()
    restored = AgentExecutionCheckpoint.from_dict(serialized)

    assert restored.messages == before_checkpoint.messages


@pytest.mark.asyncio
async def test_llm_agent_checkpoint_failure_before_tool_prevents_side_effect() -> None:
    definition = AgentDefinition(
        name="production-llm-agent",
        description="Production LLM agent",
        system_prompt="You are a production assistant.",
        model="gpt-test",
        tool_names=("search",),
    )

    checkpoint_handler = FailingAgentCheckpointHandler(
        fail_position=AgentCheckpointPosition.BEFORE_TOOL_EXECUTION,
    )
    gateway = FakeToolCallingLLMGateway()

    llm_context = AgentLLMContext(
        gateway,
        AgentLLMConfig(
            model=definition.model,
            system_prompt=definition.system_prompt,
        ),
    )

    tool_registry = InMemoryToolRegistry()
    tool = FakeRAGTool(name="search")
    await tool_registry.register(tool)

    tools = AgentToolContext(
        tool_registry,
        definition,
    )

    context = AgentExecutionContext(
        AgentRequest(
            input="Find information about RAG.",
            user_id="user-123",
            session_id="session-456",
            metadata={"request_id": "request-789"},
        ),
        tools=tools,
        llm=llm_context,
        run_id="run-checkpoint-before-tool-failure-1",
    )

    agent = LLMAgent(
        definition,
        checkpoint_handler=checkpoint_handler,
    )

    with pytest.raises(
        RuntimeError,
        match="checkpoint persistence failed",
    ):
        await agent.run(context)

    assert tool.execute_count == 0
    assert len(gateway.requests) == 1
    assert len(checkpoint_handler.attempted_checkpoints) == 1

    attempted_checkpoint = checkpoint_handler.attempted_checkpoints[0]

    assert attempted_checkpoint.run_id == "run-checkpoint-before-tool-failure-1"
    assert attempted_checkpoint.tool_round == 1
    assert attempted_checkpoint.position is AgentCheckpointPosition.BEFORE_TOOL_EXECUTION
    assert attempted_checkpoint.messages[-1].role is AgentMessageRole.ASSISTANT
    assert "call-123" in attempted_checkpoint.messages[-1].content


@pytest.mark.asyncio
async def test_llm_agent_checkpoint_failure_occurs_after_tool_side_effect() -> None:
    definition = AgentDefinition(
        name="production-llm-agent",
        description="Production LLM agent",
        system_prompt="You are a production assistant.",
        model="gpt-test",
        tool_names=("search",),
    )

    checkpoint_handler = FailingAgentCheckpointHandler()
    gateway = FakeToolCallingLLMGateway()

    llm_context = AgentLLMContext(
        gateway,
        AgentLLMConfig(
            model=definition.model,
            system_prompt=definition.system_prompt,
        ),
    )

    tool_registry = InMemoryToolRegistry()
    tool = FakeRAGTool(name="search")
    await tool_registry.register(tool)

    tools = AgentToolContext(
        tool_registry,
        definition,
    )

    context = AgentExecutionContext(
        AgentRequest(
            input="Find information about RAG.",
            user_id="user-123",
            session_id="session-456",
            metadata={"request_id": "request-789"},
        ),
        tools=tools,
        llm=llm_context,
        run_id="run-checkpoint-failure-1",
    )

    agent = LLMAgent(
        definition,
        checkpoint_handler=checkpoint_handler,
    )

    with pytest.raises(
        RuntimeError,
        match="checkpoint persistence failed",
    ):
        await agent.run(context)

    assert tool.execute_count == 1
    assert len(gateway.requests) == 1

    assert len(checkpoint_handler.attempted_checkpoints) == 2

    before_checkpoint = checkpoint_handler.attempted_checkpoints[0]

    assert before_checkpoint.run_id == "run-checkpoint-failure-1"
    assert before_checkpoint.tool_round == 1
    assert before_checkpoint.position is AgentCheckpointPosition.BEFORE_TOOL_EXECUTION
    assert before_checkpoint.messages[-1].role is AgentMessageRole.ASSISTANT
    assert "call-123" in before_checkpoint.messages[-1].content

    after_checkpoint = checkpoint_handler.attempted_checkpoints[1]

    assert after_checkpoint.run_id == "run-checkpoint-failure-1"
    assert after_checkpoint.tool_round == 1
    assert after_checkpoint.position is AgentCheckpointPosition.AFTER_TOOL_EXECUTION
    assert after_checkpoint.messages[-1].role is AgentMessageRole.TOOL
    assert "call-123" in after_checkpoint.messages[-1].content


@pytest.mark.asyncio
async def test_llm_agent_checkpoint_failure_preserves_idempotent_tool_result() -> None:
    definition = AgentDefinition(
        name="production-llm-agent",
        description="Production LLM agent",
        system_prompt="You are a production assistant.",
        model="gpt-test",
        tool_names=("search",),
    )

    checkpoint_handler = FailingAgentCheckpointHandler()
    gateway = FakeToolCallingLLMGateway()

    llm_context = AgentLLMContext(
        gateway,
        AgentLLMConfig(
            model=definition.model,
            system_prompt=definition.system_prompt,
        ),
    )

    tool_registry = InMemoryToolRegistry()
    tool = FakeRAGTool(name="search")
    await tool_registry.register(tool)

    idempotency_store = InMemoryToolExecutionIdempotencyStore()
    execution_service = ToolExecutionService(
        tool_registry,
        idempotency_store=idempotency_store,
    )

    tools = AgentToolContext(
        tool_registry,
        definition,
        execution_service=execution_service,
    )

    context = AgentExecutionContext(
        AgentRequest(
            input="Find information about RAG.",
            user_id="user-123",
            session_id="session-456",
            metadata={"request_id": "request-789"},
        ),
        tools=tools,
        llm=llm_context,
        run_id="run-checkpoint-idempotency-1",
    )

    agent = LLMAgent(
        definition,
        checkpoint_handler=checkpoint_handler,
    )

    with pytest.raises(
        RuntimeError,
        match="checkpoint persistence failed",
    ):
        await agent.run(context)

    assert tool.execute_count == 1
    assert len(gateway.requests) == 1
    assert len(checkpoint_handler.attempted_checkpoints) == 2

    before_checkpoint = checkpoint_handler.attempted_checkpoints[0]

    assert before_checkpoint.run_id == "run-checkpoint-idempotency-1"
    assert before_checkpoint.tool_round == 1
    assert before_checkpoint.position is AgentCheckpointPosition.BEFORE_TOOL_EXECUTION
    assert before_checkpoint.messages[-1].role is AgentMessageRole.ASSISTANT
    assert "call-123" in before_checkpoint.messages[-1].content

    after_checkpoint = checkpoint_handler.attempted_checkpoints[1]

    assert after_checkpoint.run_id == "run-checkpoint-idempotency-1"
    assert after_checkpoint.tool_round == 1
    assert after_checkpoint.position is AgentCheckpointPosition.AFTER_TOOL_EXECUTION
    assert after_checkpoint.messages[-1].role is AgentMessageRole.TOOL
    assert "call-123" in after_checkpoint.messages[-1].content

    replay_service = ToolExecutionService(
        tool_registry,
        idempotency_store=idempotency_store,
    )

    replay_context = ToolExecutionContext(
        run_id="run-checkpoint-idempotency-1",
        call_id="call-123",
        agent_name="production-llm-agent",
        session_id="session-456",
        user_id="user-123",
        request_metadata={"request_id": "request-789"},
    )

    replay = await replay_service.execute(
        "search",
        {"query": "RAG"},
        execution_context=replay_context,
    )

    assert replay.success is True
    assert replay.output == {
        "query": "RAG",
        "retrieved_count": 2,
        "results": [
            {
                "chunk_id": "chunk-rag-001",
                "document_id": "doc-rag-001",
                "content": "Sensitive enterprise context.",
                "score": 0.91,
                "retrieval_score": 0.72,
                "reranker_score": 0.91,
                "metadata": {
                    "classification": "confidential",
                },
            },
            {
                "chunk_id": "chunk-rag-002",
                "document_id": "doc-rag-002",
                "content": "Another sensitive context.",
                "score": 0.83,
                "retrieval_score": 0.61,
                "reranker_score": 0.88,
                "metadata": {
                    "classification": "restricted",
                },
            },
        ],
    }

    assert tool.execute_count == 1


@pytest.mark.asyncio
async def test_llm_agent_does_not_capture_checkpoint_without_run_id() -> None:
    definition = AgentDefinition(
        name="production-llm-agent",
        description="Production LLM agent",
        system_prompt="You are a production assistant.",
        model="gpt-test",
        tool_names=("search",),
    )

    checkpoint_handler = FakeAgentCheckpointHandler()
    gateway = FakeToolCallingLLMGateway()

    llm_context = AgentLLMContext(
        gateway,
        AgentLLMConfig(
            model=definition.model,
            system_prompt=definition.system_prompt,
        ),
    )

    tools = AgentToolContext(
        InMemoryToolRegistry(),
        definition,
    )

    context = AgentExecutionContext(
        AgentRequest(
            input="Find information about RAG.",
            user_id="user-123",
            session_id="session-456",
        ),
        tools=tools,
        llm=llm_context,
    )

    agent = LLMAgent(
        definition,
        checkpoint_handler=checkpoint_handler,
    )

    await agent.run(context)

    assert checkpoint_handler.checkpoints == []


@pytest.mark.asyncio
async def test_llm_agent_emits_event_to_observer() -> None:
    definition = AgentDefinition(
        name="production-llm-agent",
        description="Production LLM agent",
        system_prompt="You are a production assistant.",
        model="gpt-test",
    )

    observer = FakeAgentExecutionObserver()

    agent = LLMAgent(
        definition,
        observer=observer,
    )

    event = AgentExecutionEvent(
        event_type=AgentExecutionEventType.AGENT_STARTED,
        agent_name=definition.name,
    )

    await agent._emit(event)

    assert observer.events == [event]


@pytest.mark.asyncio
async def test_llm_agent_emit_is_noop_without_observer() -> None:
    definition = AgentDefinition(
        name="production-llm-agent",
        description="Production LLM agent",
        system_prompt="You are a production assistant.",
        model="gpt-test",
    )

    agent = LLMAgent(definition)

    event = AgentExecutionEvent(
        event_type=AgentExecutionEventType.AGENT_STARTED,
        agent_name=definition.name,
    )

    await agent._emit(event)


@pytest.mark.asyncio
async def test_llm_agent_emits_normal_execution_lifecycle_events() -> None:
    definition = AgentDefinition(
        name="production-llm-agent",
        description="Production LLM agent.",
        system_prompt="You are a production assistant.",
        model="gpt-test",
    )

    gateway = FakeLLMGateway()

    llm_context = AgentLLMContext(
        gateway,
        AgentLLMConfig(
            model=definition.model,
            system_prompt=definition.system_prompt,
        ),
    )

    tools = AgentToolContext(
        InMemoryToolRegistry(),
        definition,
    )

    context = AgentExecutionContext(
        AgentRequest(
            input="Explain RAG.",
            user_id="user-123",
            session_id="session-456",
        ),
        tools=tools,
        llm=llm_context,
    )

    observer = FakeAgentExecutionObserver()

    agent = LLMAgent(
        definition,
        observer=observer,
    )

    response = await agent.run(context)

    assert response.output == "Generated answer."

    assert [event.event_type for event in observer.events] == [
        AgentExecutionEventType.AGENT_STARTED,
        AgentExecutionEventType.CONTEXT_ASSEMBLY_COMPLETED,
        AgentExecutionEventType.LLM_REQUESTED,
        AgentExecutionEventType.LLM_COMPLETED,
        AgentExecutionEventType.RUNTIME_DECISION,
        AgentExecutionEventType.AGENT_COMPLETED,
    ]

    runtime_decision = observer.events[4]
    assert runtime_decision.metadata == {
        "decision": "stop",
        "reason": "iteration_budget_exhausted",
        "iteration": 1,
    }


@pytest.mark.asyncio
async def test_llm_agent_lifecycle_events_include_execution_metadata() -> None:
    definition = AgentDefinition(
        name="production-llm-agent",
        description="Production LLM agent.",
        system_prompt="You are a production assistant.",
        model="gpt-test",
    )

    gateway = FakeLLMGateway()

    llm_context = AgentLLMContext(
        gateway,
        AgentLLMConfig(
            model=definition.model,
            system_prompt=definition.system_prompt,
        ),
    )

    tools = AgentToolContext(
        InMemoryToolRegistry(),
        definition,
    )

    context = AgentExecutionContext(
        AgentRequest(
            input="Explain RAG.",
            user_id="user-123",
            session_id="session-456",
        ),
        tools=tools,
        llm=llm_context,
        run_id="run-123",
    )

    observer = FakeAgentExecutionObserver()

    agent = LLMAgent(
        definition,
        observer=observer,
    )

    await agent.run(context)

    started = observer.events[0]
    context_assembly = observer.events[1]
    llm_requested = observer.events[2]
    llm_completed = observer.events[3]
    runtime_decision = observer.events[4]
    completed = observer.events[5]

    assert started.agent_name == "production-llm-agent"
    assert context_assembly.event_type == AgentExecutionEventType.CONTEXT_ASSEMBLY_COMPLETED
    assert context_assembly.run_id == "run-123"
    assert context_assembly.metadata["total_messages"] == 2
    assert context_assembly.metadata["source_counts"] == {
        "system_prompt": 1,
        "user_input": 1,
    }
    assert started.session_id == "session-456"
    assert started.user_id == "user-123"

    assert llm_requested.tool_round == 0
    assert llm_requested.user_id == "user-123"

    assert llm_completed.tool_round == 0
    assert llm_completed.user_id == "user-123"
    assert llm_completed.provider == "fake"
    assert llm_completed.model == "gpt-test"
    assert llm_completed.metadata == {
        "prompt_tokens": 10,
        "completion_tokens": 5,
        "total_tokens": 15,
    }

    assert runtime_decision.tool_round == 0
    assert runtime_decision.user_id == "user-123"
    assert runtime_decision.provider == "fake"
    assert runtime_decision.model == "gpt-test"
    assert runtime_decision.step_index == 0
    assert runtime_decision.metadata == {
        "decision": "stop",
        "reason": "iteration_budget_exhausted",
        "iteration": 1,
    }

    assert completed.tool_round == 0
    assert completed.user_id == "user-123"
    assert completed.provider == "fake"
    assert completed.model == "gpt-test"

    assert [event.run_id for event in observer.events] == [
        "run-123",
        "run-123",
        "run-123",
        "run-123",
        "run-123",
        "run-123",
    ]


@pytest.mark.asyncio
async def test_llm_agent_emits_tool_call_lifecycle_events() -> None:
    definition = AgentDefinition(
        name="production-llm-agent",
        description="Production LLM agent",
        system_prompt="You are a production assistant.",
        model="gpt-test",
        tool_names=("search",),
    )

    gateway = FakeToolCallingLLMGateway()

    tool_registry = InMemoryToolRegistry()

    await tool_registry.register(
        FakeTool(name="search"),
    )

    observer = FakeAgentExecutionObserver()

    agent = LLMAgent(
        definition,
        observer=observer,
    )

    context = AgentExecutionContext(
        AgentRequest(
            input="Find something.",
            session_id="session-123",
        ),
        tools=AgentToolContext(
            tool_registry,
            definition,
        ),
        llm=AgentLLMContext(
            gateway,
            definition.llm_config,
        ),
    )

    await agent.run(context)

    tool_events = [
        event
        for event in observer.events
        if event.event_type
        in {
            AgentExecutionEventType.TOOL_CALL_REQUESTED,
            AgentExecutionEventType.TOOL_CALL_COMPLETED,
            AgentExecutionEventType.TOOL_CALL_FAILED,
        }
    ]

    assert [event.event_type for event in tool_events] == [
        AgentExecutionEventType.TOOL_CALL_REQUESTED,
        AgentExecutionEventType.TOOL_CALL_COMPLETED,
    ]

    assert tool_events[0].agent_name == definition.name
    assert tool_events[0].session_id == "session-123"
    assert tool_events[0].tool_round == 1
    assert tool_events[0].tool_name == "search"
    assert tool_events[0].call_id == "call-123"


@pytest.mark.asyncio
async def test_llm_agent_completes_rag_orchestration_step_on_tool_result() -> None:
    definition = AgentDefinition(
        name="enterprise-rag-analyst",
        description="Enterprise RAG analyst.",
        system_prompt="You are an enterprise RAG analyst.",
        model="gpt-test",
        tool_names=("rag.search",),
    )

    gateway = FakeToolCallingLLMGateway(tool_name="rag.search")
    tool_registry = InMemoryToolRegistry()

    await tool_registry.register(FakeRAGTool())

    plan = build_enterprise_rag_analyst_plan()
    state = plan.materialize_state()

    context = AgentExecutionContext(
        AgentRequest(
            input="Find information about RAG.",
            session_id="session-rag-orchestration",
        ),
        tools=AgentToolContext(
            tool_registry,
            definition,
        ),
        llm=AgentLLMContext(
            gateway,
            definition.llm_config,
        ),
    )

    context.orchestration_plan = plan
    context.orchestration_state = state

    agent = LLMAgent(definition)

    response = await agent.run(context)

    retrieve_step = state.steps[0]

    assert retrieve_step.status is OrchestrationStepStatus.COMPLETED
    assert retrieve_step.tool_round == 1

    assert context.runtime_state.phase.value == "evaluate"
    assert context.runtime_state.current_step_index == 2
    assert context.runtime_state.decision.value == "stop"

    result = state.get_completed_step_result("retrieve_evidence")

    assert result.output["query"] == "RAG"
    assert result.output["retrieved_count"] == 2
    assert result.metadata["rag_provenance"]["retrieved_count"] == 2
    assert result.metadata["rag_provenance"]["sources"] == [
        {
            "chunk_id": "chunk-rag-001",
            "document_id": "doc-rag-001",
            "retrieval_score": 0.72,
            "reranker_score": 0.91,
            "score": 0.91,
        },
        {
            "chunk_id": "chunk-rag-002",
            "document_id": "doc-rag-002",
            "retrieval_score": 0.61,
            "reranker_score": 0.88,
            "score": 0.83,
        },
    ]

    assert response.output == "RAG retrieves relevant context for generation."


@pytest.mark.asyncio
async def test_llm_agent_persists_rag_tool_execution_binding() -> None:
    from app.control_plane.agent_run_steps.in_memory import (
        InMemoryAgentRunStepsRepository,
    )

    definition = AgentDefinition(
        name="enterprise-rag-analyst",
        description="Enterprise RAG analyst.",
        system_prompt="You are an enterprise RAG analyst.",
        model="gpt-test",
        tool_names=("rag.search",),
    )

    gateway = FakeToolCallingLLMGateway(tool_name="rag.search")
    tool_registry = InMemoryToolRegistry()

    await tool_registry.register(FakeRAGTool())

    plan = build_enterprise_rag_analyst_plan()
    state = plan.materialize_state(iteration=1)
    repository = InMemoryAgentRunStepsRepository()

    context = AgentExecutionContext(
        AgentRequest(
            input="Find information about RAG.",
            session_id="session-rag-binding",
        ),
        tools=AgentToolContext(
            tool_registry,
            definition,
        ),
        llm=AgentLLMContext(
            gateway,
            definition.llm_config,
        ),
        run_id="run-rag-binding",
        agent_run_steps_repository_factory=lambda: repository,
    )

    context.orchestration_plan = plan
    context.orchestration_state = state

    agent = LLMAgent(definition)

    response = await agent.run(context)

    assert response.output == "RAG retrieves relevant context for generation."

    step = repository.get(
        "run-rag-binding",
        "iteration-1:retrieve_evidence",
    )

    assert step is not None
    assert step.step_id == "iteration-1:retrieve_evidence"
    assert step.step_id != "call-123"
    assert step.step_type == "tool"
    assert step.status is AgentRunStepStatus.COMPLETED
    assert step.tool_name == "rag.search"
    assert step.call_id == "call-123"
    assert step.input == {"query": "RAG"}
    assert step.output["query"] == "RAG"
    assert step.output["retrieved_count"] == 2
    assert step.completed_at is not None
    assert step.metadata["runtime"] == {
        "phase": "act",
        "decision": None,
        "decision_reason": None,
        "current_step_index": 0,
        "iteration": 1,
    }


@pytest.mark.asyncio
async def test_llm_agent_pauses_durable_tool_step_for_pending_approval() -> None:
    definition = AgentDefinition(
        name="enterprise-rag-analyst",
        description="Enterprise RAG analyst.",
        system_prompt="You are an enterprise RAG analyst.",
        model="gpt-test",
        tool_names=("rag.search",),
    )

    gateway = FakeToolCallingLLMGateway(tool_name="rag.search")
    tool_registry = InMemoryToolRegistry()

    rag_tool = FakeRAGTool()
    rag_tool._definition = ToolDefinition(
        name="rag.search",
        description="A test RAG search tool.",
        metadata={
            "source": "mcp",
            "mcp_server": "research-mcp",
            "risk_tier": "high",
            "side_effect": True,
            "capability": "enterprise_retrieval",
            "permission_scope": "retrieval:execute",
        },
    )
    await tool_registry.register(rag_tool)

    execution_service = ToolExecutionService(
        tool_registry,
        authorization_service=ToolAuthorizationService(
            ProvenanceToolAuthorizer(),
        ),
        idempotency_store=InMemoryToolExecutionIdempotencyStore(),
        approval_coordinator=ControlPlaneApprovalCoordinator(
            policy=SideEffectApprovalPolicy(
                approval_risk_tiers={"high", "critical"},
            ),
            repository_factory=lambda: approval_repository,
        ),
    )

    plan = build_enterprise_rag_analyst_plan()
    repository = InMemoryAgentRunStepsRepository()
    approval_repository = InMemoryApprovalRequestRepository()

    context = AgentExecutionContext(
        AgentRequest(
            input="Find information about RAG.",
            user_id="user-approval",
            principal="agent:research",
            tenant_id="tenant-acme",
            session_id="session-approval",
        ),
        tools=AgentToolContext(
            tool_registry,
            definition,
            execution_service=execution_service,
        ),
        llm=AgentLLMContext(
            gateway,
            definition.llm_config,
        ),
        run_id="run-approval",
        orchestration_plan=plan,
        agent_run_steps_repository_factory=lambda: repository,
    )

    agent = LLMAgent(definition)

    with pytest.raises(
        AgentExecutionWaitingForApprovalError,
        match="waiting for human approval",
    ):
        await agent.run(context)

    step = repository.get(
        "run-approval",
        "iteration-1:retrieve_evidence",
    )

    assert step is not None
    assert step.status is AgentRunStepStatus.RUNNING
    assert step.tool_name == "rag.search"
    assert step.call_id == "call-123"
    assert step.input == {"query": "RAG"}
    assert step.output is None
    assert step.completed_at is None

    approvals = approval_repository.list_by_run("run-approval")

    assert len(approvals) == 1

    approval = approvals[0]

    assert approval.status is ApprovalStatus.PENDING
    assert approval.run_id == "run-approval"
    assert approval.step_id == "iteration-1:retrieve_evidence"
    assert approval.call_id == "call-123"
    assert approval.tool_name == "rag.search"
    assert approval.risk_tier == "high"
    assert approval.policy_name == "side-effect-requires-approval"
    assert approval.requested_action == "Execute side-effecting tool 'rag.search'"

    assert rag_tool.execute_count == 0


@pytest.mark.asyncio
async def test_llm_agent_propagates_execution_provenance_to_event_and_durable_step() -> None:
    definition = AgentDefinition(
        name="enterprise-rag-analyst",
        description="Enterprise RAG analyst.",
        system_prompt="You are an enterprise RAG analyst.",
        model="gpt-test",
        tool_names=("rag.search",),
    )

    gateway = FakeToolCallingLLMGateway(tool_name="rag.search")
    tool_registry = InMemoryToolRegistry()

    rag_tool = FakeRAGTool()
    rag_tool._definition = ToolDefinition(
        name="rag.search",
        description="A test RAG search tool.",
        metadata={
            "source": "mcp",
            "mcp_server": "research-mcp",
        },
    )
    await tool_registry.register(rag_tool)

    execution_service = ToolExecutionService(
        tool_registry,
        authorization_service=ToolAuthorizationService(
            ProvenanceToolAuthorizer(),
        ),
        idempotency_store=InMemoryToolExecutionIdempotencyStore(),
    )

    plan = build_enterprise_rag_analyst_plan()
    repository = InMemoryAgentRunStepsRepository()

    context = AgentExecutionContext(
        AgentRequest(
            input="Find information about RAG.",
            user_id="user-provenance",
            principal="agent:research",
            tenant_id="tenant-acme",
            session_id="session-provenance",
        ),
        tools=AgentToolContext(
            tool_registry,
            definition,
            execution_service=execution_service,
        ),
        llm=AgentLLMContext(
            gateway,
            definition.llm_config,
        ),
        run_id="run-provenance",
        orchestration_plan=plan,
        agent_run_steps_repository_factory=lambda: repository,
    )

    observer = FakeAgentExecutionObserver()
    agent = LLMAgent(
        definition,
        observer=observer,
    )

    await agent.run(context)

    completed_events = [
        event
        for event in observer.events
        if event.event_type is AgentExecutionEventType.TOOL_CALL_COMPLETED
    ]

    assert len(completed_events) == 1

    event_provenance = completed_events[0].metadata["execution_provenance"]

    expected_provenance = {
        "tool_source": "mcp",
        "mcp_server": "research-mcp",
        "tenant_id": "tenant-acme",
        "authorization_decision": True,
        "authorization_policy_id": "policy-enterprise-tools",
        "authorization_policy_version": "v7",
        "idempotency_key": "deldai:run-provenance:call-123:rag.search",
        "execution_status": "completed",
    }

    assert event_provenance == expected_provenance

    step = repository.get(
        "run-provenance",
        "iteration-1:retrieve_evidence",
    )

    assert step is not None
    assert step.status is AgentRunStepStatus.COMPLETED
    assert step.metadata["execution_provenance"] == expected_provenance
    assert step.metadata["rag_provenance"]["retrieved_count"] == 2

    assert "query" not in event_provenance
    assert "content" not in event_provenance
    assert "classification" not in event_provenance
    assert "arguments" not in event_provenance
    assert "output" not in event_provenance


class FailingMCPTool:
    def __init__(self, name: str = "rag.search") -> None:
        self._definition = ToolDefinition(
            name=name,
            description="A failing MCP search tool.",
            provider=ToolProvider(
                kind="mcp",
                name="document-server",
            ),
            metadata={
                "source": "mcp",
                "mcp_server": "document-server",
                "sensitive_internal_config": "secret-token-12345",
            },
        )
        self.execute_count = 0

    @property
    def definition(self) -> ToolDefinition:
        return self._definition

    async def execute(self, arguments: dict[str, Any]) -> dict[str, Any]:
        self.execute_count += 1
        raise RuntimeError("Remote MCP protocol failure: Connection lost to document-server")


@pytest.mark.asyncio
async def test_llm_agent_mcp_failure_produces_bounded_event_metadata() -> None:
    definition = AgentDefinition(
        name="enterprise-rag-analyst",
        description="Enterprise RAG analyst.",
        system_prompt="You are an enterprise RAG analyst.",
        model="gpt-test",
        tool_names=("rag.search",),
    )

    gateway = FakeToolCallingLLMGateway(tool_name="rag.search")
    tool_registry = InMemoryToolRegistry()

    mcp_tool = FailingMCPTool()
    await tool_registry.register(mcp_tool)

    execution_service = ToolExecutionService(
        tool_registry,
        authorization_service=ToolAuthorizationService(
            ProvenanceToolAuthorizer(),
        ),
        idempotency_store=InMemoryToolExecutionIdempotencyStore(),
    )

    context = AgentExecutionContext(
        AgentRequest(
            input="Find information about RAG.",
            user_id="user-provenance",
            principal="agent:research",
            tenant_id="tenant-acme",
            session_id="session-provenance",
        ),
        tools=AgentToolContext(
            tool_registry,
            definition,
            execution_service=execution_service,
        ),
        llm=AgentLLMContext(
            gateway,
            definition.llm_config,
        ),
        run_id="run-mcp-failure",
    )

    observer = FakeAgentExecutionObserver()
    agent = LLMAgent(
        definition,
        observer=observer,
    )

    await agent.run(context)

    failed_events = [
        event
        for event in observer.events
        if event.event_type is AgentExecutionEventType.TOOL_CALL_FAILED
    ]

    assert len(failed_events) == 1
    failed_event = failed_events[0]

    assert failed_event.metadata["failure_category"] == "execution_error"

    event_provenance = failed_event.metadata["execution_provenance"]

    expected_provenance = {
        "tool_source": "mcp",
        "mcp_server": "document-server",
        "tool_provider": {
            "kind": "mcp",
            "name": "document-server",
        },
        "execution_status": "failed",
        "tenant_id": "tenant-acme",
        "authorization_decision": True,
        "authorization_policy_id": "policy-enterprise-tools",
        "authorization_policy_version": "v7",
        "idempotency_key": "deldai:run-mcp-failure:call-123:rag.search",
    }

    assert event_provenance == expected_provenance

    assert "Remote MCP protocol failure" not in str(failed_event.metadata)
    assert "sensitive_internal_config" not in str(failed_event.metadata)
    assert "secret-token-12345" not in str(failed_event.metadata)


@pytest.mark.asyncio
async def test_llm_agent_never_executes_tool_for_completed_durable_step() -> None:
    definition = AgentDefinition(
        name="enterprise-rag-analyst",
        description="Enterprise RAG analyst.",
        system_prompt="You are an enterprise RAG analyst.",
        model="gpt-test",
        tool_names=("rag.search",),
    )

    gateway = FakeToolCallingLLMGateway(tool_name="rag.search")
    tool = FakeTool(name="rag.search")

    tool_registry = InMemoryToolRegistry()
    await tool_registry.register(tool)

    plan = build_enterprise_rag_analyst_plan()
    repository = InMemoryAgentRunStepsRepository()
    run_id = "run-rag-completed-no-reexecution"

    context = AgentExecutionContext(
        AgentRequest(
            input="Find information about RAG.",
            session_id="session-completed",
        ),
        tools=AgentToolContext(
            tool_registry,
            definition,
        ),
        llm=AgentLLMContext(
            gateway,
            definition.llm_config,
        ),
        run_id=run_id,
        orchestration_plan=plan,
        agent_run_steps_repository_factory=lambda: repository,
    )

    agent = LLMAgent(definition)
    step = context.orchestration_state.steps[0]
    context.orchestration_state.start_step(0)

    repository.create(
        AgentRunStep(
            run_id=run_id,
            step_id=step.step_id,
            step_index=step.step_index,
            step_type="tool",
            status=AgentRunStepStatus.COMPLETED,
            tool_name="rag.search",
            call_id="call-123",
            input={"query": "RAG"},
            output={"query": "RAG", "retrieved_count": 2},
        )
    )

    tool_calls = (
        AgentToolCall(
            call_id="call-123",
            name="rag.search",
            arguments={"query": "RAG"},
        ),
    )

    with pytest.raises(
        RuntimeError,
        match="cannot execute tool for durable orchestration step from status: completed",
    ):
        await agent._execute_tool_calls_and_append_results(
            [],
            context,
            tool_calls,
            tool_round=1,
        )

    assert tool.execution_count == 0


@pytest.mark.asyncio
async def test_llm_agent_rejects_tool_execution_when_durable_binding_fails() -> None:

    definition = AgentDefinition(
        name="enterprise-rag-analyst",
        description="Enterprise RAG analyst.",
        system_prompt="You are an enterprise RAG analyst.",
        model="gpt-test",
        tool_names=("rag.search",),
    )

    gateway = FakeToolCallingLLMGateway(tool_name="rag.search")
    tool = FakeTool(name="rag.search")

    tool_registry = InMemoryToolRegistry()
    await tool_registry.register(tool)

    plan = build_enterprise_rag_analyst_plan()

    repository = InMemoryAgentRunStepsRepository()
    run_id = "run-rag-binding-failure"

    context = AgentExecutionContext(
        AgentRequest(
            input="Find information about RAG.",
            session_id="session-123",
        ),
        tools=AgentToolContext(
            tool_registry,
            definition,
        ),
        llm=AgentLLMContext(
            gateway,
            definition.llm_config,
        ),
        run_id=run_id,
        orchestration_plan=plan,
        agent_run_steps_repository_factory=lambda: repository,
    )

    agent = LLMAgent(definition)

    state = context.orchestration_state
    step = state.steps[0]

    repository.create(
        AgentRunStep(
            run_id=run_id,
            step_id=step.step_id,
            step_index=step.step_index,
            step_type="tool",
            status=AgentRunStepStatus.RUNNING,
            tool_name="rag.search",
            call_id="different-call-id",
        )
    )

    await agent._start_orchestration_step(context)

    tool_calls = (
        AgentToolCall(
            call_id="call-123",
            name="rag.search",
            arguments={"query": "RAG"},
        ),
    )

    with pytest.raises(
        ValueError,
        match="call_id is already bound to a different value",
    ):
        await agent._execute_tool_calls_and_append_results(
            [],
            context,
            tool_calls,
            tool_round=1,
        )

    assert tool.execution_count == 0


@pytest.mark.asyncio
async def test_llm_agent_retries_transient_provider_failure_durably() -> None:
    class RetryGateway(FakeLLMGateway):
        def __init__(self) -> None:
            super().__init__()
            self.generate_count = 0

        async def route_chat(self, request: dict[str, Any]) -> dict[str, Any]:
            self.generate_count += 1
            if self.generate_count == 1:
                raise TimeoutError("provider timeout")
            return {
                "provider": "fake",
                "model": request["model"],
                "reply": "Recovered answer.",
                "usage": {
                    "prompt_tokens": 10,
                    "completion_tokens": 5,
                    "total_tokens": 15,
                },
                "tool_calls": [],
            }

    definition = AgentDefinition(
        name="retryable-llm-agent",
        description="Test durable LLM retry.",
        system_prompt="You are a retry test agent.",
        model="mock-gpt",
    )

    gateway = RetryGateway()
    repository = InMemoryAgentRunStepsRepository()
    observer = FakeAgentExecutionObserver()
    run_id = "run-llm-retry-success"

    context = AgentExecutionContext(
        AgentRequest(
            input="Answer after retry.",
            session_id="session-llm-retry",
        ),
        tools=AgentToolContext(
            InMemoryToolRegistry(),
            definition,
        ),
        llm=AgentLLMContext(
            gateway,
            definition.llm_config,
        ),
        run_id=run_id,
        orchestration_plan=OrchestrationPlan(
            steps=(
                OrchestrationStep(
                    step_id="answer",
                    step_index=0,
                    name="Produce answer",
                    status=OrchestrationStepStatus.PENDING,
                    completion_policy=OrchestrationStepCompletionPolicy.ON_AGENT_RESPONSE,
                ),
            )
        ),
        agent_run_steps_repository_factory=lambda: repository,
    )
    context.orchestration_state = context.orchestration_plan.materialize_state()

    agent = LLMAgent(
        definition,
        observer=observer,
        retry_policy=RetryPolicy(
            max_attempts=3,
            initial_backoff_seconds=0,
            jitter=0,
        ),
    )

    response = await agent.run(context)

    assert response.output == "Recovered answer."
    assert gateway.generate_count == 2

    step = repository.get(run_id, "answer")

    assert step is not None
    assert step.status is AgentRunStepStatus.COMPLETED
    assert step.attempt == 2
    assert step.error is None
    assert step.failure_category is None
    assert step.completed_at is not None
    assert step.metadata["retry"] == {
        "category": "timeout",
        "disposition": "retryable",
        "provider_category": "timeout",
        "attempt": 1,
        "max_attempts": 3,
        "retry_allowed": True,
        "retry_reason": ("Category 'timeout' is retryable on attempt 1/3"),
        "backoff_seconds": 0.0,
    }

    orchestration_events = [
        event
        for event in observer.events
        if event.event_type
        in {
            AgentExecutionEventType.ORCHESTRATION_STEP_STARTED,
            AgentExecutionEventType.ORCHESTRATION_STEP_FAILED,
            AgentExecutionEventType.ORCHESTRATION_STEP_COMPLETED,
        }
    ]

    assert [(event.event_type, event.attempt) for event in orchestration_events] == [
        (AgentExecutionEventType.ORCHESTRATION_STEP_STARTED, 1),
        (AgentExecutionEventType.ORCHESTRATION_STEP_FAILED, 1),
        (AgentExecutionEventType.ORCHESTRATION_STEP_STARTED, 2),
        (AgentExecutionEventType.ORCHESTRATION_STEP_COMPLETED, 2),
    ]

    failed_event = orchestration_events[1]
    assert failed_event.metadata == {
        "retry": {
            "category": "timeout",
            "disposition": "retryable",
            "provider_category": "timeout",
            "attempt": 1,
            "max_attempts": 3,
            "retry_allowed": True,
            "retry_reason": ("Category 'timeout' is retryable on attempt 1/3"),
            "backoff_seconds": 0.0,
        },
        "error_type": "TimeoutError",
    }


@pytest.mark.asyncio
async def test_llm_agent_cancellation_after_llm_failure_prevents_durable_retry() -> None:
    class CancellationAfterFailureRepository(InMemoryAgentRunStepsRepository):
        def __init__(self, cancellation_requested: asyncio.Event) -> None:
            super().__init__()
            self.cancellation_requested = cancellation_requested
            self.retry_calls = 0

        def retry(self, *args, **kwargs):
            self.retry_calls += 1
            return super().retry(*args, **kwargs)

        def transition(self, *args, **kwargs):
            result = super().transition(*args, **kwargs)

            if kwargs.get("status") is AgentRunStepStatus.FAILED and result is not None:
                self.cancellation_requested.set()

            return result

    class FailingRetryGateway(FakeLLMGateway):
        def __init__(self) -> None:
            super().__init__()
            self.generate_count = 0

        async def route_chat(self, request: dict[str, Any]) -> dict[str, Any]:
            self.generate_count += 1
            raise TimeoutError("provider timeout before durable retry")

    definition = AgentDefinition(
        name="cancel-before-llm-durable-retry-agent",
        description="Test cancellation before durable LLM retry.",
        system_prompt="You are a cancellation race test agent.",
        model="mock-gpt",
    )

    cancellation_requested = asyncio.Event()
    gateway = FailingRetryGateway()
    repository = CancellationAfterFailureRepository(cancellation_requested)
    observer = FakeAgentExecutionObserver()
    run_id = "run-cancel-before-llm-durable-retry"

    plan = OrchestrationPlan(
        steps=(
            OrchestrationStep(
                step_id="answer",
                step_index=0,
                name="Produce answer",
                status=OrchestrationStepStatus.PENDING,
                completion_policy=OrchestrationStepCompletionPolicy.ON_AGENT_RESPONSE,
            ),
        )
    )

    context = AgentExecutionContext(
        AgentRequest(
            input="Trigger cancellation before LLM retry.",
            session_id="session-cancel-before-llm-durable-retry",
        ),
        tools=AgentToolContext(
            InMemoryToolRegistry(),
            definition,
        ),
        llm=AgentLLMContext(
            gateway,
            definition.llm_config,
        ),
        run_id=run_id,
        orchestration_plan=plan,
        cancellation_requested=cancellation_requested,
        agent_run_steps_repository_factory=lambda: repository,
    )
    context.orchestration_state = plan.materialize_state()

    agent = LLMAgent(
        definition,
        observer=observer,
        retry_policy=RetryPolicy(
            max_attempts=2,
            initial_backoff_seconds=0,
            jitter=0,
        ),
    )

    with pytest.raises(asyncio.CancelledError):
        await agent.run(context)

    step = repository.get(run_id, "answer")

    assert step is not None
    assert step.status is AgentRunStepStatus.FAILED
    assert step.attempt == 1
    assert step.failure_category == "timeout"
    assert gateway.generate_count == 1
    assert repository.retry_calls == 0
    assert cancellation_requested.is_set()

    orchestration_events = [
        event
        for event in observer.events
        if event.step_id == "answer"
        and event.event_type
        in {
            AgentExecutionEventType.ORCHESTRATION_STEP_STARTED,
            AgentExecutionEventType.ORCHESTRATION_STEP_FAILED,
            AgentExecutionEventType.ORCHESTRATION_STEP_COMPLETED,
        }
    ]

    assert [(event.event_type, event.attempt) for event in orchestration_events] == [
        (AgentExecutionEventType.ORCHESTRATION_STEP_STARTED, 1),
        (AgentExecutionEventType.ORCHESTRATION_STEP_FAILED, 1),
    ]


@pytest.mark.asyncio
async def test_llm_agent_applies_durable_retry_backoff(monkeypatch) -> None:
    class RetryGateway(FakeLLMGateway):
        def __init__(self) -> None:
            super().__init__()
            self.generate_count = 0

        async def route_chat(self, request: dict[str, Any]) -> dict[str, Any]:
            self.generate_count += 1
            if self.generate_count == 1:
                raise TimeoutError("provider timeout")

            return {
                "provider": "fake",
                "model": request["model"],
                "reply": "Recovered after backoff.",
                "usage": {
                    "prompt_tokens": 10,
                    "completion_tokens": 5,
                    "total_tokens": 15,
                },
                "tool_calls": [],
            }

    slept: list[float] = []

    async def fake_sleep(delay: float) -> None:
        slept.append(delay)

    monkeypatch.setattr(
        "ai_platform.agents.llm_agent.asyncio.sleep",
        fake_sleep,
    )

    definition = AgentDefinition(
        name="retryable-llm-agent-backoff",
        description="Test durable LLM retry backoff.",
        system_prompt="You are a retry test agent.",
        model="mock-gpt",
    )

    gateway = RetryGateway()
    repository = InMemoryAgentRunStepsRepository()
    run_id = "run-llm-retry-backoff"

    plan = OrchestrationPlan(
        steps=(
            OrchestrationStep(
                step_id="answer",
                step_index=0,
                name="Produce answer",
                status=OrchestrationStepStatus.PENDING,
                completion_policy=OrchestrationStepCompletionPolicy.ON_AGENT_RESPONSE,
            ),
        )
    )

    context = AgentExecutionContext(
        AgentRequest(
            input="Answer after retry backoff.",
            session_id="session-llm-retry-backoff",
        ),
        tools=AgentToolContext(
            InMemoryToolRegistry(),
            definition,
        ),
        llm=AgentLLMContext(
            gateway,
            definition.llm_config,
        ),
        run_id=run_id,
        orchestration_plan=plan,
        agent_run_steps_repository_factory=lambda: repository,
    )
    context.orchestration_state = plan.materialize_state()

    agent = LLMAgent(
        definition,
        retry_policy=RetryPolicy(
            max_attempts=3,
            initial_backoff_seconds=2.0,
            backoff_factor=2.0,
            jitter=0.0,
        ),
    )

    response = await agent.run(context)

    assert response.output == "Recovered after backoff."
    assert gateway.generate_count == 2
    assert slept == [2.0]

    step = repository.get(run_id, "answer")

    assert step is not None
    assert step.status is AgentRunStepStatus.COMPLETED
    assert step.attempt == 2
    assert step.metadata["retry"]["backoff_seconds"] == 2.0


@pytest.mark.asyncio
async def test_llm_agent_retries_transient_provider_failure_until_third_attempt() -> None:
    class RetryGateway(FakeLLMGateway):
        def __init__(self) -> None:
            super().__init__()
            self.generate_count = 0

        async def route_chat(self, request: dict[str, Any]) -> dict[str, Any]:
            self.generate_count += 1
            if self.generate_count < 3:
                raise TimeoutError(f"provider timeout {self.generate_count}")
            return {
                "provider": "fake",
                "model": request["model"],
                "reply": "Recovered on third attempt.",
                "usage": {
                    "prompt_tokens": 10,
                    "completion_tokens": 5,
                    "total_tokens": 15,
                },
                "tool_calls": [],
            }

    definition = AgentDefinition(
        name="retryable-llm-agent-three-attempts",
        description="Test multiple durable LLM retries.",
        system_prompt="You are a retry test agent.",
        model="mock-gpt",
    )

    gateway = RetryGateway()
    repository = InMemoryAgentRunStepsRepository()
    run_id = "run-llm-retry-third-attempt"

    plan = OrchestrationPlan(
        steps=(
            OrchestrationStep(
                step_id="answer",
                step_index=0,
                name="Produce answer",
                status=OrchestrationStepStatus.PENDING,
                completion_policy=OrchestrationStepCompletionPolicy.ON_AGENT_RESPONSE,
            ),
        )
    )

    context = AgentExecutionContext(
        AgentRequest(
            input="Answer after two retries.",
            session_id="session-llm-retry-three",
        ),
        tools=AgentToolContext(
            InMemoryToolRegistry(),
            definition,
        ),
        llm=AgentLLMContext(
            gateway,
            definition.llm_config,
        ),
        run_id=run_id,
        orchestration_plan=plan,
        agent_run_steps_repository_factory=lambda: repository,
    )
    context.orchestration_state = plan.materialize_state()

    agent = LLMAgent(
        definition,
        retry_policy=RetryPolicy(
            max_attempts=3,
            initial_backoff_seconds=0,
            jitter=0,
        ),
    )

    response = await agent.run(context)

    assert response.output == "Recovered on third attempt."
    assert gateway.generate_count == 3

    step = repository.get(run_id, "answer")

    assert step is not None
    assert step.status is AgentRunStepStatus.COMPLETED
    assert step.attempt == 3


@pytest.mark.asyncio
async def test_llm_agent_does_not_retry_non_retryable_provider_failure() -> None:
    class NonRetryableGateway(FakeLLMGateway):
        def __init__(self) -> None:
            super().__init__()
            self.generate_count = 0

        async def route_chat(self, request: dict[str, Any]) -> dict[str, Any]:
            self.generate_count += 1
            raise ValueError("invalid_request: malformed provider request")

    definition = AgentDefinition(
        name="non-retryable-llm-agent",
        description="Test non-retryable LLM failure.",
        system_prompt="You are a retry test agent.",
        model="mock-gpt",
    )

    gateway = NonRetryableGateway()
    repository = InMemoryAgentRunStepsRepository()
    run_id = "run-llm-non-retryable"

    plan = OrchestrationPlan(
        steps=(
            OrchestrationStep(
                step_id="answer",
                step_index=0,
                name="Produce answer",
                status=OrchestrationStepStatus.PENDING,
                completion_policy=OrchestrationStepCompletionPolicy.ON_AGENT_RESPONSE,
            ),
        )
    )

    context = AgentExecutionContext(
        AgentRequest(
            input="Trigger a non-retryable failure.",
            session_id="session-llm-non-retryable",
        ),
        tools=AgentToolContext(
            InMemoryToolRegistry(),
            definition,
        ),
        llm=AgentLLMContext(
            gateway,
            definition.llm_config,
        ),
        run_id=run_id,
        orchestration_plan=plan,
        agent_run_steps_repository_factory=lambda: repository,
    )
    context.orchestration_state = plan.materialize_state()

    agent = LLMAgent(
        definition,
        retry_policy=RetryPolicy(
            max_attempts=3,
            initial_backoff_seconds=0,
            jitter=0,
        ),
    )

    with pytest.raises(ValueError, match="invalid_request"):
        await agent.run(context)

    assert gateway.generate_count == 1

    step = repository.get(run_id, "answer")

    assert step is not None
    assert step.status is AgentRunStepStatus.FAILED
    assert step.attempt == 1
    assert step.failure_category == "invalid_request"
    assert step.error == ("ValueError: invalid_request: malformed provider request")
    assert step.completed_at is not None


@pytest.mark.asyncio
async def test_llm_agent_stops_retrying_at_max_attempts() -> None:
    class AlwaysTimeoutGateway(FakeLLMGateway):
        def __init__(self) -> None:
            super().__init__()
            self.generate_count = 0

        async def route_chat(self, request: dict[str, Any]) -> dict[str, Any]:
            self.generate_count += 1
            raise TimeoutError("provider timeout")

    definition = AgentDefinition(
        name="max-attempts-llm-agent",
        description="Test maximum durable LLM attempts.",
        system_prompt="You are a retry test agent.",
        model="mock-gpt",
    )

    gateway = AlwaysTimeoutGateway()
    repository = InMemoryAgentRunStepsRepository()
    run_id = "run-llm-max-attempts"

    plan = OrchestrationPlan(
        steps=(
            OrchestrationStep(
                step_id="answer",
                step_index=0,
                name="Produce answer",
                status=OrchestrationStepStatus.PENDING,
                completion_policy=OrchestrationStepCompletionPolicy.ON_AGENT_RESPONSE,
            ),
        )
    )

    context = AgentExecutionContext(
        AgentRequest(
            input="Trigger maximum retries.",
            session_id="session-llm-max-attempts",
        ),
        tools=AgentToolContext(
            InMemoryToolRegistry(),
            definition,
        ),
        llm=AgentLLMContext(
            gateway,
            definition.llm_config,
        ),
        run_id=run_id,
        orchestration_plan=plan,
        agent_run_steps_repository_factory=lambda: repository,
    )
    context.orchestration_state = plan.materialize_state()

    agent = LLMAgent(
        definition,
        retry_policy=RetryPolicy(
            max_attempts=2,
            initial_backoff_seconds=0,
            jitter=0,
        ),
    )

    with pytest.raises(TimeoutError, match="provider timeout"):
        await agent.run(context)

    assert gateway.generate_count == 2

    step = repository.get(run_id, "answer")

    assert step is not None
    assert step.status is AgentRunStepStatus.FAILED
    assert step.attempt == 2
    assert step.failure_category == "timeout"
    assert step.completed_at is not None


@pytest.mark.asyncio
async def test_llm_agent_marks_llm_ownership_loss_ambiguous_without_retry() -> None:
    class OwnershipLossGateway(FakeLLMGateway):
        def __init__(self) -> None:
            super().__init__()
            self.generate_count = 0

        async def route_chat(self, request: dict[str, Any]) -> dict[str, Any]:
            self.generate_count += 1
            raise AgentExecutionOwnershipLostError("execution ownership lost during LLM call")

    definition = AgentDefinition(
        name="llm-ownership-loss-agent",
        description="Test ambiguous LLM ownership loss.",
        system_prompt="You are an ownership test agent.",
        model="mock-gpt",
    )

    gateway = OwnershipLossGateway()
    repository = InMemoryAgentRunStepsRepository()
    run_id = "run-llm-ownership-loss"

    plan = OrchestrationPlan(
        steps=(
            OrchestrationStep(
                step_id="answer",
                step_index=0,
                name="Produce answer",
                status=OrchestrationStepStatus.PENDING,
                completion_policy=OrchestrationStepCompletionPolicy.ON_AGENT_RESPONSE,
            ),
        )
    )

    context = AgentExecutionContext(
        AgentRequest(
            input="Trigger ownership loss.",
            session_id="session-llm-ownership-loss",
        ),
        tools=AgentToolContext(
            InMemoryToolRegistry(),
            definition,
        ),
        llm=AgentLLMContext(
            gateway,
            definition.llm_config,
        ),
        run_id=run_id,
        orchestration_plan=plan,
        execution_ownership_lost=asyncio.Event(),
        agent_run_steps_repository_factory=lambda: repository,
    )
    context.orchestration_state = plan.materialize_state()

    agent = LLMAgent(
        definition,
        retry_policy=RetryPolicy(
            max_attempts=3,
            initial_backoff_seconds=0,
            jitter=0,
        ),
    )

    with pytest.raises(AgentExecutionOwnershipLostError):
        await agent.run(context)

    assert gateway.generate_count == 1

    step = repository.get(run_id, "answer")

    assert step is not None
    assert step.status is AgentRunStepStatus.AMBIGUOUS
    assert step.failure_category == "execution_ambiguous"
    assert step.completed_at is not None


@pytest.mark.asyncio
async def test_llm_agent_persists_failed_durable_tool_step() -> None:
    definition = AgentDefinition(
        name="test-tool-failure-agent",
        description="Test agent for durable tool failure.",
        system_prompt="You are a test agent.",
        model="gpt-test",
        tool_names=("failing_tool",),
    )

    gateway = FakeToolCallingLLMGateway(tool_name="failing_tool")
    tool_registry = InMemoryToolRegistry()
    await tool_registry.register(FailingTool())

    plan = OrchestrationPlan(
        steps=(
            OrchestrationStep(
                step_id="execute_tool",
                step_index=0,
                name="Execute failing tool",
                status=OrchestrationStepStatus.PENDING,
                completion_policy=OrchestrationStepCompletionPolicy.ON_TOOL_RESULT,
                metadata={"completion_tool_name": "failing_tool"},
            ),
        )
    )
    repository = InMemoryAgentRunStepsRepository()
    run_id = "run-tool-failed"

    context = AgentExecutionContext(
        AgentRequest(
            input="Execute the failing tool.",
            session_id="session-failed",
        ),
        tools=AgentToolContext(
            tool_registry,
            definition,
        ),
        llm=AgentLLMContext(
            gateway,
            definition.llm_config,
        ),
        run_id=run_id,
        orchestration_plan=plan,
        agent_run_steps_repository_factory=lambda: repository,
    )

    agent = LLMAgent(definition)

    await agent._start_orchestration_step(context)

    with pytest.raises(Exception):
        await agent.run(context)

    step_id = context.orchestration_state.steps[0].step_id
    step = repository.get(run_id, step_id)

    assert step is not None
    assert step.status is AgentRunStepStatus.FAILED
    assert step.failure_category == ToolExecutionFailureCategory.EXECUTION_ERROR.value
    assert step.error == "RuntimeError: simulated tool failure"
    assert step.completed_at is not None


@pytest.mark.asyncio
async def test_llm_agent_classifies_tool_failure_using_execution_policy() -> None:
    class RetryPolicyFailingTool:
        def __init__(self) -> None:
            self._definition = ToolDefinition(
                name="policy_failing_tool",
                description="A policy-aware failing test tool.",
                execution_policy=ToolExecutionPolicy(
                    max_retries=2,
                    retryable_failure_categories=frozenset(
                        {ToolExecutionFailureCategory.EXECUTION_ERROR}
                    ),
                ),
            )
            self.execution_count = 0

        @property
        def definition(self) -> ToolDefinition:
            return self._definition

        async def execute(self, arguments):
            self.execution_count += 1
            raise RuntimeError("simulated policy failure")

    definition = AgentDefinition(
        name="test-policy-failure-agent",
        description="Test agent for runtime failure classification.",
        system_prompt="You are a test agent.",
        model="gpt-test",
        tool_names=("policy_failing_tool",),
    )

    gateway = FakeToolCallingLLMGateway(tool_name="policy_failing_tool")
    tool_registry = InMemoryToolRegistry()
    tool = RetryPolicyFailingTool()
    await tool_registry.register(tool)

    plan = OrchestrationPlan(
        steps=(
            OrchestrationStep(
                step_id="execute_tool",
                step_index=0,
                name="Execute policy-aware failing tool",
                status=OrchestrationStepStatus.PENDING,
                completion_policy=OrchestrationStepCompletionPolicy.ON_TOOL_RESULT,
                metadata={"completion_tool_name": "policy_failing_tool"},
            ),
        )
    )
    repository = InMemoryAgentRunStepsRepository()
    run_id = "run-policy-tool-failed"

    context = AgentExecutionContext(
        AgentRequest(
            input="Execute the policy-aware failing tool.",
            session_id="session-policy-failed",
        ),
        tools=AgentToolContext(
            tool_registry,
            definition,
        ),
        llm=AgentLLMContext(
            gateway,
            definition.llm_config,
        ),
        run_id=run_id,
        orchestration_plan=plan,
        agent_run_steps_repository_factory=lambda: repository,
    )

    agent = LLMAgent(definition)

    await agent._start_orchestration_step(context)

    with patch(
        "ai_platform.agents.llm_agent.classify_runtime_failure",
        wraps=__import__(
            "ai_platform.agents.failure_classification",
            fromlist=["classify_runtime_failure"],
        ).classify_runtime_failure,
    ) as classifier:
        with pytest.raises(Exception):
            await agent.run(context)

    classifier.assert_called_once_with(
        ToolExecutionFailureCategory.EXECUTION_ERROR,
        retryable_failure_categories=frozenset({ToolExecutionFailureCategory.EXECUTION_ERROR}),
    )

    step = repository.get(run_id, "iteration-1:execute_tool")

    assert step is not None
    assert step.status is AgentRunStepStatus.FAILED
    assert step.attempt == 1
    assert step.failure_category == ToolExecutionFailureCategory.EXECUTION_ERROR.value
    assert step.error == "RuntimeError: simulated policy failure"
    assert step.completed_at is not None
    assert tool.execution_count == 3
    assert tool.definition.execution_policy.max_retries == 2


@pytest.mark.asyncio
async def test_llm_agent_cancellation_after_failure_commit_prevents_durable_retry() -> None:
    class CancellationAfterFailureRepository(InMemoryAgentRunStepsRepository):
        def __init__(self, cancellation_requested: asyncio.Event) -> None:
            super().__init__()
            self.cancellation_requested = cancellation_requested
            self.retry_calls = 0

        def retry(self, *args, **kwargs):
            self.retry_calls += 1
            return super().retry(*args, **kwargs)

        def transition(self, *args, **kwargs):
            result = super().transition(*args, **kwargs)

            if kwargs.get("status") is AgentRunStepStatus.FAILED and result is not None:
                self.cancellation_requested.set()

            return result

    class FailingTool:
        def __init__(self) -> None:
            self._definition = ToolDefinition(
                name="cancel_before_durable_retry_tool",
                description="A tool that always fails.",
                execution_policy=ToolExecutionPolicy(
                    max_retries=1,
                    retryable_failure_categories=frozenset(
                        {ToolExecutionFailureCategory.EXECUTION_ERROR}
                    ),
                ),
            )
            self.execution_count = 0

        @property
        def definition(self) -> ToolDefinition:
            return self._definition

        async def execute(self, arguments):
            self.execution_count += 1
            raise RuntimeError("simulated durable retry failure")

    definition = AgentDefinition(
        name="cancel-before-durable-retry-agent",
        description="Test cancellation before durable retry.",
        system_prompt="You are a cancellation race test agent.",
        model="gpt-test",
        tool_names=("cancel_before_durable_retry_tool",),
    )

    gateway = FakeToolCallingLLMGateway(
        tool_name="cancel_before_durable_retry_tool",
    )
    tool_registry = InMemoryToolRegistry()
    tool = FailingTool()
    await tool_registry.register(tool)

    plan = OrchestrationPlan(
        steps=(
            OrchestrationStep(
                step_id="execute_tool",
                step_index=0,
                name="Execute cancellation race tool",
                status=OrchestrationStepStatus.PENDING,
                completion_policy=OrchestrationStepCompletionPolicy.ON_TOOL_RESULT,
                metadata={"completion_tool_name": "cancel_before_durable_retry_tool"},
            ),
        )
    )

    cancellation_requested = asyncio.Event()
    repository = CancellationAfterFailureRepository(cancellation_requested)
    observer = FakeAgentExecutionObserver()
    run_id = "run-cancel-before-durable-retry"

    context = AgentExecutionContext(
        AgentRequest(
            input="Execute the cancellation race tool.",
            session_id="session-cancel-before-durable-retry",
        ),
        tools=AgentToolContext(
            tool_registry,
            definition,
        ),
        llm=AgentLLMContext(
            gateway,
            definition.llm_config,
        ),
        run_id=run_id,
        orchestration_plan=plan,
        cancellation_requested=cancellation_requested,
        agent_run_steps_repository_factory=lambda: repository,
    )

    context.orchestration_state = plan.materialize_state()

    agent = LLMAgent(
        definition,
        observer=observer,
        retry_policy=RetryPolicy(
            max_attempts=2,
            retryable_categories={"execution_error"},
            initial_backoff_seconds=0,
            jitter=0,
        ),
    )

    with pytest.raises(asyncio.CancelledError):
        await agent.run(context)

    step = repository.get(run_id, "execute_tool")

    assert step is not None
    assert step.status is AgentRunStepStatus.FAILED
    assert step.attempt == 1
    assert step.failure_category == ToolExecutionFailureCategory.EXECUTION_ERROR.value
    assert tool.execution_count == 2
    assert repository.retry_calls == 0
    assert cancellation_requested.is_set()

    orchestration_events = [
        event
        for event in observer.events
        if event.step_id == "execute_tool"
        and event.event_type
        in {
            AgentExecutionEventType.ORCHESTRATION_STEP_STARTED,
            AgentExecutionEventType.ORCHESTRATION_STEP_FAILED,
            AgentExecutionEventType.ORCHESTRATION_STEP_COMPLETED,
        }
    ]

    assert [(event.event_type, event.attempt) for event in orchestration_events] == [
        (AgentExecutionEventType.ORCHESTRATION_STEP_STARTED, 1),
        (AgentExecutionEventType.ORCHESTRATION_STEP_FAILED, 1),
    ]


@pytest.mark.asyncio
async def test_llm_agent_durable_retries_after_tool_local_retries_are_exhausted() -> None:
    class DurableRetryTool:
        def __init__(self) -> None:
            self._definition = ToolDefinition(
                name="durable_retry_tool",
                description="A tool that fails twice before succeeding.",
                execution_policy=ToolExecutionPolicy(
                    max_retries=1,
                    retryable_failure_categories=frozenset(
                        {ToolExecutionFailureCategory.EXECUTION_ERROR}
                    ),
                ),
            )
            self.execution_count = 0

        @property
        def definition(self) -> ToolDefinition:
            return self._definition

        async def execute(self, arguments):
            self.execution_count += 1

            if self.execution_count <= 2:
                raise RuntimeError(f"simulated retry failure {self.execution_count}")

            return "tool recovered"

    definition = AgentDefinition(
        name="durable-tool-retry-agent",
        description="Test durable retry after tool-local retries.",
        system_prompt="You are a durable retry test agent.",
        model="gpt-test",
        tool_names=("durable_retry_tool",),
    )

    gateway = FakeToolCallingLLMGateway(
        tool_name="durable_retry_tool",
    )
    tool_registry = InMemoryToolRegistry()
    tool = DurableRetryTool()
    await tool_registry.register(tool)

    plan = OrchestrationPlan(
        steps=(
            OrchestrationStep(
                step_id="execute_tool",
                step_index=0,
                name="Execute durable retry tool",
                status=OrchestrationStepStatus.PENDING,
                completion_policy=OrchestrationStepCompletionPolicy.ON_TOOL_RESULT,
                metadata={"completion_tool_name": "durable_retry_tool"},
            ),
            OrchestrationStep(
                step_id="final_response",
                step_index=1,
                name="Generate final response",
                status=OrchestrationStepStatus.PENDING,
                completion_policy=OrchestrationStepCompletionPolicy.ON_AGENT_RESPONSE,
            ),
        )
    )

    repository = InMemoryAgentRunStepsRepository()
    observer = FakeAgentExecutionObserver()
    run_id = "run-durable-tool-retry"

    context = AgentExecutionContext(
        AgentRequest(
            input="Execute the durable retry tool.",
            session_id="session-durable-tool-retry",
        ),
        tools=AgentToolContext(
            tool_registry,
            definition,
        ),
        llm=AgentLLMContext(
            gateway,
            definition.llm_config,
        ),
        run_id=run_id,
        orchestration_plan=plan,
        agent_run_steps_repository_factory=lambda: repository,
    )

    context.orchestration_state = plan.materialize_state()

    agent = LLMAgent(
        definition,
        observer=observer,
        retry_policy=RetryPolicy(
            max_attempts=2,
            retryable_categories={"execution_error"},
            initial_backoff_seconds=0,
            jitter=0,
        ),
    )

    response = await agent.run(context)

    assert response.output
    assert tool.execution_count == 3

    step_id = context.orchestration_state.steps[0].step_id
    step = repository.get(run_id, step_id)

    assert step is not None
    assert step.status is AgentRunStepStatus.COMPLETED
    assert step.attempt == 2
    assert step.failure_category is None

    retry_metadata = step.metadata["retry"]

    assert retry_metadata["category"] == "execution_error"
    assert retry_metadata["attempt"] == 1
    assert retry_metadata["max_attempts"] == 2
    assert retry_metadata["retry_allowed"] is True
    assert retry_metadata["backoff_seconds"] == 0.0

    tool_step_id = context.orchestration_state.steps[0].step_id

    orchestration_events = [
        event
        for event in observer.events
        if event.step_id == tool_step_id
        and event.event_type
        in {
            AgentExecutionEventType.ORCHESTRATION_STEP_STARTED,
            AgentExecutionEventType.ORCHESTRATION_STEP_FAILED,
            AgentExecutionEventType.ORCHESTRATION_STEP_COMPLETED,
        }
    ]

    assert [(event.event_type, event.attempt) for event in orchestration_events] == [
        (AgentExecutionEventType.ORCHESTRATION_STEP_STARTED, 1),
        (AgentExecutionEventType.ORCHESTRATION_STEP_FAILED, 1),
        (AgentExecutionEventType.ORCHESTRATION_STEP_STARTED, 2),
        (AgentExecutionEventType.ORCHESTRATION_STEP_COMPLETED, 2),
    ]

    failed_event = orchestration_events[1]

    assert failed_event.metadata == {
        "retry": {
            "category": "execution_error",
            "disposition": "retryable",
            "attempt": 1,
            "max_attempts": 2,
            "retry_allowed": True,
            "retry_reason": "Category 'execution_error' is retryable on attempt 1/2",
            "backoff_seconds": 0.0,
        },
        "error_type": "ToolExecutionError",
    }


@pytest.mark.asyncio
async def test_llm_agent_classifies_timeout_using_execution_policy() -> None:
    class TimeoutTool:
        def __init__(self) -> None:
            self._definition = ToolDefinition(
                name="timeout_tool",
                description="A timeout test tool.",
                execution_policy=ToolExecutionPolicy(
                    max_retries=0,
                    retryable_failure_categories=frozenset({ToolExecutionFailureCategory.TIMEOUT}),
                ),
            )

        @property
        def definition(self) -> ToolDefinition:
            return self._definition

        async def execute(self, arguments):
            await asyncio.sleep(0.2)

    definition = AgentDefinition(
        name="test-timeout-agent",
        description="Test agent for runtime timeout classification.",
        system_prompt="You are a test agent.",
        model="gpt-test",
        tool_names=("timeout_tool",),
    )

    gateway = FakeToolCallingLLMGateway(tool_name="timeout_tool")
    tool_registry = InMemoryToolRegistry()
    tool = TimeoutTool()
    await tool_registry.register(tool)

    execution_service = ToolExecutionService(
        tool_registry,
        idempotency_store=InMemoryToolExecutionIdempotencyStore(),
        default_timeout_seconds=0.05,
    )

    plan = OrchestrationPlan(
        steps=(
            OrchestrationStep(
                step_id="execute_tool",
                step_index=0,
                name="Execute timeout tool",
                status=OrchestrationStepStatus.PENDING,
                completion_policy=OrchestrationStepCompletionPolicy.ON_TOOL_RESULT,
                metadata={"completion_tool_name": "timeout_tool"},
            ),
        )
    )
    repository = InMemoryAgentRunStepsRepository()
    run_id = "run-timeout-tool-failed"

    context = AgentExecutionContext(
        AgentRequest(
            input="Execute the timeout tool.",
            session_id="session-timeout-failed",
        ),
        tools=AgentToolContext(
            tool_registry,
            definition,
            execution_service=execution_service,
        ),
        llm=AgentLLMContext(
            gateway,
            definition.llm_config,
        ),
        run_id=run_id,
        orchestration_plan=plan,
        agent_run_steps_repository_factory=lambda: repository,
    )

    agent = LLMAgent(definition)

    await agent._start_orchestration_step(context)

    with patch(
        "ai_platform.agents.llm_agent.classify_runtime_failure",
        wraps=__import__(
            "ai_platform.agents.failure_classification",
            fromlist=["classify_runtime_failure"],
        ).classify_runtime_failure,
    ) as classifier:
        with pytest.raises(Exception):
            await agent.run(context)

    classifier.assert_any_call(
        ToolExecutionFailureCategory.TIMEOUT,
        retryable_failure_categories=frozenset({ToolExecutionFailureCategory.TIMEOUT}),
    )

    step = repository.get(run_id, "iteration-1:execute_tool")

    assert step is not None
    assert step.status is AgentRunStepStatus.FAILED
    assert step.failure_category == ToolExecutionFailureCategory.TIMEOUT.value
    assert step.error == "Tool execution timed out after 0.05 seconds: timeout_tool"
    assert step.completed_at is not None
    assert gateway.requests
    assert tool.definition.execution_policy.max_retries == 0


@pytest.mark.asyncio
async def test_llm_agent_persists_duration_limit_failure_for_running_durable_step() -> None:
    definition = AgentDefinition(
        name="test-duration-limit-agent",
        description="Test agent for execution duration limit.",
        system_prompt="You are a test agent.",
        model="gpt-test",
    )

    gateway = FakeLLMGateway()
    tool_registry = InMemoryToolRegistry()

    plan = OrchestrationPlan(
        steps=(
            OrchestrationStep(
                step_id="generate_answer",
                step_index=0,
                name="Generate answer",
                status=OrchestrationStepStatus.PENDING,
                completion_policy=OrchestrationStepCompletionPolicy.ON_AGENT_RESPONSE,
            ),
        )
    )
    repository = InMemoryAgentRunStepsRepository()
    run_id = "run-duration-limit"

    context = AgentExecutionContext(
        AgentRequest(
            input="Generate an answer.",
            session_id="session-duration-limit",
            execution_budget=ExecutionBudget(
                max_duration_seconds=1.0,
            ),
        ),
        tools=AgentToolContext(
            tool_registry,
            definition,
        ),
        llm=AgentLLMContext(
            gateway,
            definition.llm_config,
        ),
        run_id=run_id,
        orchestration_plan=plan,
        agent_run_steps_repository_factory=lambda: repository,
    )

    agent = LLMAgent(definition)

    await agent._start_orchestration_step(context)

    context.execution_budget_state.started_at = monotonic() - 2.0

    with pytest.raises(
        AgentExecutionDurationLimitError,
        match=r"maximum execution duration \(1\.0 seconds\)",
    ):
        await agent.run(context)

    step = repository.get(run_id, "iteration-1:generate_answer")

    assert step is not None
    assert step.status is AgentRunStepStatus.FAILED
    assert step.completed_at is not None
    assert step.error == (
        "AgentExecutionDurationLimitError: "
        "Agent 'test-duration-limit-agent' exceeded the maximum execution "
        "duration (1.0 seconds)."
    )

    assert len(gateway.requests) == 0


@pytest.mark.asyncio
async def test_llm_agent_persists_ambiguous_durable_tool_step() -> None:
    definition = AgentDefinition(
        name="enterprise-rag-analyst",
        description="Enterprise RAG analyst.",
        system_prompt="You are an enterprise RAG analyst.",
        model="gpt-test",
        tool_names=("rag.search",),
    )

    gateway = FakeToolCallingLLMGateway(tool_name="rag.search")
    tool_registry = InMemoryToolRegistry()
    tool = FakeTool(name="rag.search")
    await tool_registry.register(tool)

    plan = build_enterprise_rag_analyst_plan()
    repository = InMemoryAgentRunStepsRepository()
    run_id = "run-rag-tool-ambiguous"

    context = AgentExecutionContext(
        AgentRequest(
            input="Find information about RAG.",
            session_id="session-ambiguous",
        ),
        tools=AgentToolContext(
            tool_registry,
            definition,
        ),
        llm=AgentLLMContext(
            gateway,
            definition.llm_config,
        ),
        run_id=run_id,
        orchestration_plan=plan,
        agent_run_steps_repository_factory=lambda: repository,
    )

    agent = LLMAgent(definition)

    await agent._start_orchestration_step(context)

    tool_calls = (
        AgentToolCall(
            call_id="call-123",
            name="rag.search",
            arguments={"query": "RAG"},
        ),
    )

    async def ambiguous_execute_tool_calls(tool_calls):
        from ai_platform.agents.tool_calls import AgentToolResult

        return [
            AgentToolResult(
                call_id=tool_call.call_id,
                tool_name=tool_call.name,
                error="Tool execution outcome is ambiguous.",
                failure_category=ToolExecutionFailureCategory.EXECUTION_AMBIGUOUS,
            )
            for tool_call in tool_calls
        ]

    context.execute_tool_calls = ambiguous_execute_tool_calls

    await agent._execute_tool_calls_and_append_results(
        [],
        context,
        tool_calls,
        tool_round=1,
    )

    step = repository.get(run_id, "iteration-1:retrieve_evidence")

    assert step is not None
    assert step.status is AgentRunStepStatus.AMBIGUOUS
    assert step.failure_category == ToolExecutionFailureCategory.EXECUTION_AMBIGUOUS.value
    assert step.error == "Tool execution outcome is ambiguous."
    assert step.completed_at is not None


@pytest.mark.asyncio
async def test_llm_agent_keeps_durable_tool_step_running_when_execution_is_in_progress() -> None:
    definition = AgentDefinition(
        name="enterprise-rag-analyst",
        description="Enterprise RAG analyst.",
        system_prompt="You are an enterprise RAG analyst.",
        model="gpt-test",
        tool_names=("rag.search",),
    )

    gateway = FakeToolCallingLLMGateway(tool_name="rag.search")
    tool_registry = InMemoryToolRegistry()
    tool = FakeTool(name="rag.search")
    await tool_registry.register(tool)

    plan = build_enterprise_rag_analyst_plan()
    repository = InMemoryAgentRunStepsRepository()
    run_id = "run-rag-tool-in-progress"

    context = AgentExecutionContext(
        AgentRequest(
            input="Find information about RAG.",
            session_id="session-in-progress",
        ),
        tools=AgentToolContext(
            tool_registry,
            definition,
        ),
        llm=AgentLLMContext(
            gateway,
            definition.llm_config,
        ),
        run_id=run_id,
        orchestration_plan=plan,
        agent_run_steps_repository_factory=lambda: repository,
    )

    agent = LLMAgent(definition)

    await agent._start_orchestration_step(context)

    tool_calls = (
        AgentToolCall(
            call_id="call-123",
            name="rag.search",
            arguments={"query": "RAG"},
        ),
    )

    async def in_progress_execute_tool_calls(tool_calls):
        from ai_platform.agents.tool_calls import AgentToolResult

        return [
            AgentToolResult(
                call_id=tool_call.call_id,
                tool_name=tool_call.name,
                error="Tool execution is already in progress.",
                failure_category=ToolExecutionFailureCategory.EXECUTION_IN_PROGRESS,
            )
            for tool_call in tool_calls
        ]

    context.execute_tool_calls = in_progress_execute_tool_calls

    await agent._execute_tool_calls_and_append_results(
        [],
        context,
        tool_calls,
        tool_round=1,
    )

    step = repository.get(run_id, "iteration-1:retrieve_evidence")

    assert step is not None
    assert step.status is AgentRunStepStatus.RUNNING
    assert step.completed_at is None


@pytest.mark.asyncio
async def test_llm_agent_marks_durable_tool_step_ambiguous_on_ownership_loss() -> None:
    definition = AgentDefinition(
        name="ownership-loss-agent",
        description="Test agent for ownership-loss recovery.",
        system_prompt="You are a test agent.",
        model="gpt-test",
        tool_names=("ownership_losing_tool",),
    )

    gateway = FakeToolCallingLLMGateway(tool_name="ownership_losing_tool")
    tool_registry = InMemoryToolRegistry()
    ownership_lost = asyncio.Event()
    tool = OwnershipLosingTool(ownership_lost)
    await tool_registry.register(tool)

    plan = OrchestrationPlan(
        steps=(
            OrchestrationStep(
                step_id="execute_tool",
                step_index=0,
                name="Execute ownership-losing tool",
                status=OrchestrationStepStatus.PENDING,
                completion_policy=OrchestrationStepCompletionPolicy.ON_TOOL_RESULT,
                metadata={"completion_tool_name": "ownership_losing_tool"},
            ),
        )
    )
    repository = InMemoryAgentRunStepsRepository()
    run_id = "run-tool-ownership-loss"

    context = AgentExecutionContext(
        AgentRequest(
            input="Execute the ownership-losing tool.",
            session_id="session-ownership-loss",
        ),
        tools=AgentToolContext(
            tool_registry,
            definition,
        ),
        llm=AgentLLMContext(
            gateway,
            definition.llm_config,
        ),
        run_id=run_id,
        orchestration_plan=plan,
        execution_ownership_lost=ownership_lost,
        agent_run_steps_repository_factory=lambda: repository,
    )

    agent = LLMAgent(definition)

    await agent._start_orchestration_step(context)

    tool_calls = (
        AgentToolCall(
            call_id="call-ownership-loss",
            name="ownership_losing_tool",
            arguments={"query": "RAG"},
        ),
    )

    with pytest.raises(AgentExecutionOwnershipLostError):
        await agent._execute_tool_calls_and_append_results(
            [],
            context,
            tool_calls,
            tool_round=1,
        )

    step = repository.get(run_id, "iteration-1:execute_tool")

    assert step is not None
    assert step.status is AgentRunStepStatus.AMBIGUOUS
    assert step.failure_category == ToolExecutionFailureCategory.EXECUTION_AMBIGUOUS.value
    assert step.completed_at is not None
    assert tool.execution_count == 1


@pytest.mark.asyncio
async def test_llm_agent_stops_at_rag_orchestration_boundary_before_next_llm_call() -> None:
    definition = AgentDefinition(
        name="enterprise-rag-analyst",
        description="Enterprise RAG analyst.",
        system_prompt="You are an enterprise RAG analyst.",
        model="gpt-test",
        tool_names=("rag.search",),
    )

    gateway = FakeToolCallingLLMGateway(tool_name="rag.search")
    tool_registry = InMemoryToolRegistry()
    tool = FakeRAGTool(name="rag.search")

    await tool_registry.register(tool)

    context = AgentExecutionContext(
        AgentRequest(
            input="Find information about RAG.",
            session_id="session-rag-boundary",
        ),
        tools=AgentToolContext(
            tool_registry,
            definition,
        ),
        llm=AgentLLMContext(
            gateway,
            definition.llm_config,
        ),
        orchestration_plan=build_enterprise_rag_analyst_plan(),
    )

    agent = LLMAgent(definition)

    response = await agent.run(context)

    assert response.output == "RAG retrieves relevant context for generation."

    # Each logical orchestration step gets its own LLM interaction.
    # The retrieval tool result is the boundary between step 0 and step 1;
    # the agent-response boundaries separate the remaining steps.
    assert len(gateway.requests) == 3

    state = context.orchestration_state

    assert state.steps[0].status is OrchestrationStepStatus.COMPLETED
    assert state.steps[0].step_id == "iteration-1:retrieve_evidence"

    assert state.steps[1].status is OrchestrationStepStatus.COMPLETED
    assert state.steps[1].step_id == "iteration-1:analyze_evidence"

    assert state.steps[2].status is OrchestrationStepStatus.COMPLETED
    assert state.steps[2].step_id == "iteration-1:produce_answer"

    assert tool.execute_count == 1

    retrieval_result = state.get_completed_step_result("retrieve_evidence")
    assert retrieval_result.output["retrieved_count"] == 2

    analysis_result = state.get_completed_step_result("analyze_evidence")
    assert analysis_result.output == "RAG retrieves relevant context for generation."

    answer_result = state.get_completed_step_result("produce_answer")
    assert answer_result.output == "RAG retrieves relevant context for generation."


@pytest.mark.asyncio
async def test_llm_agent_cancellation_stops_before_next_rag_orchestration_step() -> None:
    definition = AgentDefinition(
        name="enterprise-rag-analyst",
        description="Enterprise RAG analyst.",
        system_prompt="You are an enterprise RAG analyst.",
        model="gpt-test",
        tool_names=("rag.search",),
    )

    gateway = FakeToolCallingLLMGateway(tool_name="rag.search")
    tool_registry = InMemoryToolRegistry()
    cancellation_requested = asyncio.Event()

    class CancellingRAGTool(FakeRAGTool):
        async def execute(self, arguments):
            result = await super().execute(arguments)
            cancellation_requested.set()
            return result

    tool = CancellingRAGTool(name="rag.search")
    await tool_registry.register(tool)

    context = AgentExecutionContext(
        AgentRequest(
            input="Find information about RAG.",
            session_id="session-rag-cancellation-boundary",
        ),
        tools=AgentToolContext(
            tool_registry,
            definition,
        ),
        llm=AgentLLMContext(
            gateway,
            definition.llm_config,
        ),
        cancellation_requested=cancellation_requested,
        orchestration_plan=build_enterprise_rag_analyst_plan(),
    )

    agent = LLMAgent(definition)

    with pytest.raises(
        asyncio.CancelledError,
        match="Agent execution cancellation requested.",
    ):
        await agent.run(context)

    state = context.orchestration_state

    assert state.steps[0].status is OrchestrationStepStatus.COMPLETED
    assert state.steps[0].step_id == "iteration-1:retrieve_evidence"

    assert state.steps[1].status is OrchestrationStepStatus.PENDING
    assert state.steps[1].step_id == "iteration-1:analyze_evidence"

    assert state.steps[2].status is OrchestrationStepStatus.PENDING
    assert state.steps[2].step_id == "iteration-1:produce_answer"

    assert state.current_step_index == 0
    assert len(gateway.requests) == 1
    assert tool.execute_count == 1


@pytest.mark.asyncio
async def test_llm_agent_rag_tool_event_captures_sanitized_provenance() -> None:
    definition = AgentDefinition(
        name="production-llm-agent",
        description="Production LLM agent",
        system_prompt="You are a production assistant.",
        model="gpt-test",
        tool_names=("rag.search",),
    )

    gateway = FakeToolCallingLLMGateway(tool_name="rag.search")
    tool_registry = InMemoryToolRegistry()

    await tool_registry.register(FakeRAGTool())

    observer = FakeAgentExecutionObserver()

    agent = LLMAgent(
        definition,
        observer=observer,
    )

    context = AgentExecutionContext(
        AgentRequest(
            input="Find information about RAG.",
            session_id="session-rag-123",
        ),
        tools=AgentToolContext(
            tool_registry,
            definition,
        ),
        llm=AgentLLMContext(
            gateway,
            definition.llm_config,
        ),
    )

    await agent.run(context)

    completed = next(
        event
        for event in observer.events
        if event.event_type == AgentExecutionEventType.TOOL_CALL_COMPLETED
    )

    assert completed.tool_name == "rag.search"
    assert completed.call_id == "call-123"
    assert completed.metadata == {
        "execution_provenance": {
            "execution_status": "completed",
        },
        "rag_provenance": {
            "retrieved_count": 2,
            "sources": [
                {
                    "chunk_id": "chunk-rag-001",
                    "document_id": "doc-rag-001",
                    "retrieval_score": 0.72,
                    "reranker_score": 0.91,
                    "score": 0.91,
                },
                {
                    "chunk_id": "chunk-rag-002",
                    "document_id": "doc-rag-002",
                    "retrieval_score": 0.61,
                    "reranker_score": 0.88,
                    "score": 0.83,
                },
            ],
        },
    }

    serialized = str(completed.metadata)

    assert "Sensitive enterprise context." not in serialized
    assert "Another sensitive context." not in serialized
    assert "classification" not in serialized


@pytest.mark.asyncio
async def test_llm_agent_rag_provenance_retains_sources_without_scores() -> None:
    definition = AgentDefinition(
        name="enterprise-rag-analyst",
        description="Enterprise RAG analyst.",
        system_prompt="You are an enterprise RAG analyst.",
        model="gpt-test",
        tool_names=("rag.search",),
    )

    class ScorelessRAGTool(FakeRAGTool):
        def __init__(self) -> None:
            super().__init__("rag.search")

        async def execute(self, arguments):
            self.execute_count += 1
            return {
                "query": arguments["query"],
                "retrieved_count": 1,
                "results": [
                    {
                        "chunk_id": "chunk-scoreless",
                        "document_id": "doc-scoreless",
                        "content": "Context without ranking metadata.",
                        "metadata": {
                            "classification": "internal",
                        },
                    }
                ],
            }

    gateway = FakeToolCallingLLMGateway(tool_name="rag.search")
    tool_registry = InMemoryToolRegistry()
    tool = ScorelessRAGTool()
    await tool_registry.register(tool)

    observer = FakeAgentExecutionObserver()
    agent = LLMAgent(
        definition,
        observer=observer,
    )

    context = AgentExecutionContext(
        AgentRequest(
            input="Find information about RAG.",
            session_id="session-rag-scoreless",
        ),
        tools=AgentToolContext(
            tool_registry,
            definition,
        ),
        llm=AgentLLMContext(
            gateway,
            definition.llm_config,
        ),
    )

    await agent.run(context)

    completed = next(
        event
        for event in observer.events
        if event.event_type == AgentExecutionEventType.TOOL_CALL_COMPLETED
    )

    assert completed.metadata["rag_provenance"] == {
        "retrieved_count": 1,
        "sources": [
            {
                "chunk_id": "chunk-scoreless",
                "document_id": "doc-scoreless",
                "retrieval_score": None,
                "reranker_score": None,
            }
        ],
    }


@pytest.mark.asyncio
async def test_llm_agent_tool_events_do_not_capture_tool_payloads() -> None:
    definition = AgentDefinition(
        name="production-llm-agent",
        description="Production LLM agent",
        system_prompt="You are a production assistant.",
        model="gpt-test",
        tool_names=("search",),
    )

    # Reuse the same gateway/tool setup as the previous test.
    gateway = FakeToolCallingLLMGateway()
    tool_registry = InMemoryToolRegistry()

    observer = FakeAgentExecutionObserver()

    agent = LLMAgent(
        definition,
        observer=observer,
    )

    context = AgentExecutionContext(
        AgentRequest(
            input="Find something.",
            session_id="session-123",
        ),
        tools=AgentToolContext(
            tool_registry,
            definition,
        ),
        llm=AgentLLMContext(
            gateway,
            definition.llm_config,
        ),
    )

    await agent.run(context)

    tool_events = [
        event
        for event in observer.events
        if event.event_type
        in {
            AgentExecutionEventType.TOOL_CALL_REQUESTED,
            AgentExecutionEventType.TOOL_CALL_COMPLETED,
            AgentExecutionEventType.TOOL_CALL_FAILED,
        }
    ]

    for event in tool_events:
        if event.event_type is AgentExecutionEventType.TOOL_CALL_FAILED:
            assert event.metadata == {
                "failure_category": "tool_not_found",
            }
        else:
            assert event.metadata == {}


@pytest.mark.asyncio
async def test_llm_agent_failed_tool_event_includes_failure_category() -> None:
    definition = AgentDefinition(
        name="production-llm-agent",
        description="Production LLM agent",
        system_prompt="You are a production assistant.",
        model="gpt-test",
        tool_names=("failing_tool",),
    )

    gateway = FakeToolCallingLLMGateway(tool_name="failing_tool")
    tool_registry = InMemoryToolRegistry()
    await tool_registry.register(FailingTool())

    observer = FakeAgentExecutionObserver()

    agent = LLMAgent(
        definition,
        observer=observer,
    )

    context = AgentExecutionContext(
        AgentRequest(
            input="Run the failing tool.",
            session_id="session-123",
        ),
        tools=AgentToolContext(
            tool_registry,
            definition,
        ),
        llm=AgentLLMContext(
            gateway,
            definition.llm_config,
        ),
    )

    await agent.run(context)

    failed_events = [
        event
        for event in observer.events
        if event.event_type is AgentExecutionEventType.TOOL_CALL_FAILED
    ]

    assert len(failed_events) == 1
    failed_event = failed_events[0]

    assert failed_event.metadata == {
        "execution_provenance": {
            "execution_status": "failed",
        },
        "failure_category": "execution_error",
    }
    assert "simulated tool failure" not in str(failed_event.metadata)


@pytest.mark.asyncio
async def test_llm_agent_emits_complete_lifecycle_after_tool_execution() -> None:
    definition = AgentDefinition(
        name="production-llm-agent",
        description="Production LLM agent.",
        system_prompt="You are a production assistant.",
        model="gpt-test",
        tool_names=("search",),
    )

    gateway = FakeToolCallingLLMGateway()
    tool_registry = InMemoryToolRegistry()

    await tool_registry.register(
        FakeTool(name="search"),
    )

    observer = FakeAgentExecutionObserver()

    agent = LLMAgent(
        definition,
        observer=observer,
    )

    context = AgentExecutionContext(
        AgentRequest(
            input="Find something.",
            session_id="session-123",
        ),
        tools=AgentToolContext(
            tool_registry,
            definition,
        ),
        llm=AgentLLMContext(
            gateway,
            definition.llm_config,
        ),
    )

    response = await agent.run(context)

    assert response.output == "RAG retrieves relevant context for generation."

    assert [event.event_type for event in observer.events] == [
        AgentExecutionEventType.AGENT_STARTED,
        AgentExecutionEventType.CONTEXT_ASSEMBLY_COMPLETED,
        AgentExecutionEventType.LLM_REQUESTED,
        AgentExecutionEventType.LLM_COMPLETED,
        AgentExecutionEventType.TOOL_CALL_REQUESTED,
        AgentExecutionEventType.TOOL_CALL_COMPLETED,
        AgentExecutionEventType.CONTEXT_ASSEMBLY_COMPLETED,
        AgentExecutionEventType.LLM_REQUESTED,
        AgentExecutionEventType.LLM_COMPLETED,
        AgentExecutionEventType.RUNTIME_DECISION,
        AgentExecutionEventType.AGENT_COMPLETED,
    ]

    first_llm_completed = observer.events[3]
    final_llm_completed = observer.events[8]
    runtime_decision = observer.events[9]
    agent_completed = observer.events[10]

    assert first_llm_completed.tool_round == 0
    assert first_llm_completed.provider == "fake"
    assert first_llm_completed.model == "gpt-test"
    assert first_llm_completed.metadata == {
        "prompt_tokens": 10,
        "completion_tokens": 5,
        "total_tokens": 15,
    }

    assert final_llm_completed.tool_round == 1
    assert final_llm_completed.provider == "fake"
    assert final_llm_completed.model == "gpt-test"
    assert final_llm_completed.metadata == {
        "prompt_tokens": 25,
        "completion_tokens": 10,
        "total_tokens": 35,
    }

    assert runtime_decision.tool_round == 1
    assert runtime_decision.provider == "fake"
    assert runtime_decision.model == "gpt-test"
    assert runtime_decision.step_index == 0
    assert runtime_decision.metadata == {
        "decision": "stop",
        "reason": "iteration_budget_exhausted",
        "iteration": 1,
    }

    assert agent_completed.tool_round == 1
    assert agent_completed.provider == "fake"
    assert agent_completed.model == "gpt-test"


@pytest.mark.asyncio
async def test_llm_agent_executes_real_rag_tool_with_governance_context() -> None:
    class FakeRetriever:
        def __init__(self) -> None:
            self.calls: list[dict[str, Any]] = []

        async def retrieve(
            self,
            query: str,
            top_k: int,
            min_score: float | None = None,
            metadata_filter: dict[str, object] | None = None,
            governance_policy: GovernancePolicy | None = None,
        ) -> list[Any]:
            self.calls.append(
                {
                    "query": query,
                    "top_k": top_k,
                    "min_score": min_score,
                    "metadata_filter": metadata_filter,
                    "governance_policy": governance_policy,
                }
            )

            return [
                RetrievalResult(
                    chunk=DocumentChunk(
                        id="chunk-rag-001",
                        document_id="doc-rag-001",
                        content="Enterprise RAG retrieves relevant knowledge.",
                        metadata={"source": "test"},
                        chunk_index=0,
                    ),
                    score=0.95,
                )
            ]

    definition = AgentDefinition(
        name="production-llm-agent",
        description="Production LLM agent.",
        system_prompt="You are a production assistant.",
        model="gpt-test",
        tool_names=("rag.search",),
    )

    gateway = FakeToolCallingLLMGateway(tool_name="rag.search")
    retriever = FakeRetriever()
    tool_registry = InMemoryToolRegistry()

    await tool_registry.register(
        RAGSearchTool(retriever),
    )

    policy = GovernancePolicy(
        required_metadata={
            "tenant_id": "tenant-a",
        }
    )

    observer = FakeAgentExecutionObserver()

    agent = LLMAgent(
        definition,
        observer=observer,
    )

    context = AgentExecutionContext(
        AgentRequest(
            input="Find information about RAG.",
            session_id="session-rag-123",
            user_id="user-rag-456",
            governance_policy=policy,
            metadata={
                "source": "agent-api",
            },
        ),
        tools=AgentToolContext(
            tool_registry,
            definition,
        ),
        llm=AgentLLMContext(
            gateway,
            definition.llm_config,
        ),
        run_id="run-rag-123",
    )

    response = await agent.run(context)

    assert response.output == "RAG retrieves relevant context for generation."

    assert retriever.calls == [
        {
            "query": "RAG",
            "top_k": 5,
            "min_score": None,
            "metadata_filter": None,
            "governance_policy": policy,
        }
    ]

    completed = next(
        event
        for event in observer.events
        if event.event_type == AgentExecutionEventType.TOOL_CALL_COMPLETED
    )

    assert completed.tool_name == "rag.search"
    assert completed.call_id == "call-123"
    assert completed.run_id == "run-rag-123"
    assert completed.session_id == "session-rag-123"


@pytest.mark.asyncio
async def test_llm_agent_emits_agent_failed_on_llm_failure() -> None:
    definition = AgentDefinition(
        name="production-llm-agent",
        description="Production LLM agent.",
        system_prompt="You are a production assistant.",
        model="gpt-test",
    )

    class FailingLLMGateway:
        async def route_chat(
            self,
            request: dict[str, Any],
        ) -> dict[str, Any]:
            raise RuntimeError("LLM provider failure")

    observer = FakeAgentExecutionObserver()

    agent = LLMAgent(
        definition,
        observer=observer,
    )

    context = AgentExecutionContext(
        AgentRequest(
            input="Explain RAG.",
            session_id="session-failure",
        ),
        tools=AgentToolContext(
            InMemoryToolRegistry(),
            definition,
        ),
        llm=AgentLLMContext(
            FailingLLMGateway(),
            definition.llm_config,
        ),
    )

    with pytest.raises(
        RuntimeError,
        match="LLM provider failure",
    ):
        await agent.run(context)

    assert [event.event_type for event in observer.events] == [
        AgentExecutionEventType.AGENT_STARTED,
        AgentExecutionEventType.CONTEXT_ASSEMBLY_COMPLETED,
        AgentExecutionEventType.LLM_REQUESTED,
        AgentExecutionEventType.AGENT_FAILED,
    ]

    failed = observer.events[-1]

    assert failed.agent_name == definition.name
    assert failed.session_id == "session-failure"
    assert failed.tool_round == 0
    assert failed.metadata == {
        "error_type": "RuntimeError",
    }


@pytest.mark.asyncio
async def test_llm_agent_emits_agent_failed_on_tool_loop_limit() -> None:
    definition = AgentDefinition(
        name="production-llm-agent",
        description="Production LLM agent.",
        system_prompt="You are a production assistant.",
        model="gpt-test",
        tool_names=("search",),
    )

    gateway = FakeMultiRoundToolCallingLLMGateway()
    observer = FakeAgentExecutionObserver()

    agent = LLMAgent(
        definition,
        observer=observer,
    )

    context = AgentExecutionContext(
        AgentRequest(
            input="Find information about RAG.",
            user_id="user-123",
            session_id="session-loop-limit",
        ),
        tools=AgentToolContext(
            InMemoryToolRegistry(),
            definition,
        ),
        llm=AgentLLMContext(
            gateway,
            definition.llm_config,
        ),
    )

    with pytest.raises(
        AgentToolLoopLimitError,
        match="Agent 'production-llm-agent' exceeded the maximum tool-call rounds \\(3\\)",
    ):
        await agent.run(context)

    assert observer.events[-1].event_type == AgentExecutionEventType.AGENT_FAILED
    assert observer.events[-1].agent_name == definition.name
    assert observer.events[-1].session_id == "session-loop-limit"
    assert observer.events[-1].tool_round == 3
    assert observer.events[-1].metadata == {
        "error_type": "AgentToolLoopLimitError",
    }


@pytest.mark.asyncio
async def test_llm_agent_resume_uses_checkpoint_messages_without_replaying_tool() -> None:
    definition = AgentDefinition(
        name="production-llm-agent",
        description="Production LLM agent.",
        system_prompt="You are a production LLM agent.",
        model="mock-gpt",
        tool_names=("search",),
    )

    gateway = FakeLLMGateway()

    llm_context = AgentLLMContext(
        gateway,
        AgentLLMConfig(
            model=definition.model,
            system_prompt=definition.system_prompt,
        ),
    )

    tool_registry = InMemoryToolRegistry()
    tool = FakeRAGTool(name="search")
    await tool_registry.register(tool)

    tools = AgentToolContext(
        tool_registry,
        definition,
    )

    context = AgentExecutionContext(
        AgentRequest(
            input="Find information about RAG.",
            user_id="user-123",
            session_id="session-456",
        ),
        tools=tools,
        llm=llm_context,
        run_id="run-resume-1",
    )

    checkpoint = AgentExecutionCheckpoint(
        schema_version=AgentExecutionCheckpoint.CURRENT_SCHEMA_VERSION,
        run_id="run-resume-1",
        agent_name=definition.name,
        session_id="session-456",
        user_id="user-123",
        messages=(
            system_message("You are a production LLM agent."),
            user_message("Find information about RAG."),
            assistant_tool_call_message(
                tool_calls=(
                    AgentToolCall(
                        call_id="call-completed-1",
                        name="search",
                        arguments={"query": "RAG"},
                    ),
                ),
                content="I searched for the information.",
            ),
            tool_result_message(
                call_id="call-completed-1",
                tool_name="search",
                output={
                    "query": "RAG",
                    "retrieved_count": 2,
                },
            ),
        ),
        tool_round=1,
        position=AgentCheckpointPosition.AFTER_TOOL_EXECUTION,
        metadata={},
    )

    agent = LLMAgent(definition)

    response = await agent.resume(context, checkpoint)

    assert response.output == "Generated answer."
    assert response.metadata["tool_rounds"] == 1
    assert tool.execute_count == 0

    assert len(gateway.requests) == 1
    assert gateway.requests[0]["messages"] == [
        {
            "role": "system",
            "content": "You are a production LLM agent.",
        },
        {
            "role": "user",
            "content": "Find information about RAG.",
        },
        {
            "role": "assistant",
            "content": "I searched for the information.",
            "tool_calls": [
                {
                    "call_id": "call-completed-1",
                    "name": "search",
                    "arguments": {"query": "RAG"},
                }
            ],
        },
        {
            "role": "tool",
            "content": (
                '{"call_id": "call-completed-1", '
                '"output": {"query": "RAG", "retrieved_count": 2}, '
                '"success": true, "tool_name": "search"}'
            ),
            "tool_call_id": "call-completed-1",
            "tool_name": "search",
        },
    ]


@pytest.mark.asyncio
async def test_llm_agent_resume_from_before_tool_checkpoint_executes_saved_tool() -> None:
    definition = AgentDefinition(
        name="production-llm-agent",
        description="Production LLM agent.",
        system_prompt="You are a production LLM agent.",
        model="mock-gpt",
        tool_names=("search",),
    )

    gateway = FakeLLMGateway()

    llm_context = AgentLLMContext(
        gateway,
        AgentLLMConfig(
            model=definition.model,
            system_prompt=definition.system_prompt,
        ),
    )

    tool_registry = InMemoryToolRegistry()
    tool = FakeRAGTool(name="search")
    await tool_registry.register(tool)

    tools = AgentToolContext(
        tool_registry,
        definition,
    )

    checkpoint_handler = FakeAgentCheckpointHandler()

    context = AgentExecutionContext(
        AgentRequest(
            input="Find information about RAG.",
            user_id="user-123",
            session_id="session-456",
        ),
        tools=tools,
        llm=llm_context,
        run_id="run-before-tool-resume-1",
    )

    checkpoint = AgentExecutionCheckpoint(
        schema_version=AgentExecutionCheckpoint.CURRENT_SCHEMA_VERSION,
        run_id="run-before-tool-resume-1",
        agent_name=definition.name,
        session_id="session-456",
        user_id="user-123",
        messages=(
            system_message("You are a production LLM agent."),
            user_message("Find information about RAG."),
            assistant_tool_call_message(
                tool_calls=(
                    AgentToolCall(
                        call_id="call-recover-1",
                        name="search",
                        arguments={"query": "RAG"},
                    ),
                ),
                content="I searched for the information.",
            ),
        ),
        tool_round=1,
        position=AgentCheckpointPosition.BEFORE_TOOL_EXECUTION,
        metadata={},
        execution_budget_state=ExecutionBudgetState(
            llm_calls=1,
            tool_calls=1,
            tool_rounds=1,
        ),
    )

    agent = LLMAgent(
        definition,
        checkpoint_handler=checkpoint_handler,
    )

    response = await agent.resume(context, checkpoint)

    assert response.output == "Generated answer."
    assert response.metadata["tool_rounds"] == 1

    assert tool.execute_count == 1

    # Recovery must execute the saved tool before making the next LLM call.
    assert len(gateway.requests) == 1
    assert gateway.requests[0]["messages"][-1]["role"] == "tool"
    assert gateway.requests[0]["messages"][-1]["tool_call_id"] == "call-recover-1"
    assert gateway.requests[0]["messages"][-1]["tool_name"] == "search"
    assert '"query": "RAG"' in gateway.requests[0]["messages"][-1]["content"]

    # The recovered execution must not consume the original tool budget again.
    assert context.execution_budget_state.llm_calls == 2
    assert context.execution_budget_state.tool_calls == 1
    assert context.execution_budget_state.tool_rounds == 1

    assert len(checkpoint_handler.checkpoints) == 1
    recovered_checkpoint = checkpoint_handler.checkpoints[0]

    assert recovered_checkpoint.run_id == checkpoint.run_id
    assert recovered_checkpoint.agent_name == checkpoint.agent_name
    assert recovered_checkpoint.tool_round == 1
    assert recovered_checkpoint.position is AgentCheckpointPosition.AFTER_TOOL_EXECUTION
    assert recovered_checkpoint.messages[:3] == checkpoint.messages
    assert recovered_checkpoint.messages[3].role is AgentMessageRole.TOOL
    assert '"call_id": "call-recover-1"' in recovered_checkpoint.messages[3].content


@pytest.mark.asyncio
async def test_llm_agent_resume_cancellation_cancels_orchestration_step() -> None:
    definition = AgentDefinition(
        name="enterprise-rag-analyst",
        description="Enterprise RAG analyst.",
        system_prompt="You are an enterprise RAG analyst.",
        model="mock-gpt",
        tool_names=("rag.search",),
    )

    class CancellationBlockingTool(FakeRAGTool):
        def __init__(self) -> None:
            super().__init__(name="rag.search")
            self.started = asyncio.Event()
            self.cancelled = False

        async def execute(self, arguments):
            self.execute_count += 1
            self.started.set()

            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                self.cancelled = True
                raise

    gateway = FakeLLMGateway()

    llm_context = AgentLLMContext(
        gateway,
        AgentLLMConfig(
            model=definition.model,
            system_prompt=definition.system_prompt,
        ),
    )

    tool_registry = InMemoryToolRegistry()
    tool = CancellationBlockingTool()
    await tool_registry.register(tool)

    tools = AgentToolContext(
        tool_registry,
        definition,
    )

    repository = InMemoryAgentRunStepsRepository()
    observer = FakeAgentExecutionObserver()

    context = AgentExecutionContext(
        AgentRequest(
            input="Find information about RAG.",
            user_id="user-123",
            session_id="session-orchestration-cancel",
        ),
        tools=tools,
        llm=llm_context,
        run_id="run-orchestration-cancel-1",
        orchestration_plan=build_enterprise_rag_analyst_plan(),
        agent_run_steps_repository_factory=lambda: repository,
    )

    checkpoint = AgentExecutionCheckpoint(
        schema_version=AgentExecutionCheckpoint.CURRENT_SCHEMA_VERSION,
        run_id="run-orchestration-cancel-1",
        agent_name=definition.name,
        session_id="session-orchestration-cancel",
        user_id="user-123",
        messages=(
            system_message("You are an enterprise RAG analyst."),
            user_message("Find information about RAG."),
            assistant_tool_call_message(
                tool_calls=(
                    AgentToolCall(
                        call_id="call-cancel-1",
                        name="rag.search",
                        arguments={"query": "RAG"},
                    ),
                ),
                content="I searched for the information.",
            ),
        ),
        tool_round=1,
        position=AgentCheckpointPosition.BEFORE_TOOL_EXECUTION,
        metadata={
            "orchestration": {
                "current_step_index": 0,
                "steps": {
                    "retrieve_evidence": {
                        "status": OrchestrationStepStatus.RUNNING.value,
                        "tool_round": None,
                    },
                },
            },
        },
        execution_budget_state=ExecutionBudgetState(
            llm_calls=1,
            tool_calls=1,
            tool_rounds=1,
        ),
    )

    step = context.orchestration_state.steps[0]
    context.orchestration_state.start_step(0)

    repository.create(
        AgentRunStep(
            run_id=context.run_id,
            step_id=step.step_id,
            step_index=step.step_index,
            step_type="tool",
            status=AgentRunStepStatus.RUNNING,
        )
    )

    agent = LLMAgent(
        definition,
        observer=observer,
    )

    execution_task = asyncio.create_task(
        agent.resume(
            context,
            checkpoint,
        )
    )

    await asyncio.wait_for(tool.started.wait(), timeout=1.0)

    execution_task.cancel()

    with pytest.raises(asyncio.CancelledError):
        await execution_task

    assert tool.cancelled is True

    state = context.orchestration_state
    assert state.current_step is not None
    assert state.current_step.status is OrchestrationStepStatus.CANCELLED
    assert state.current_step.step_id == "iteration-1:retrieve_evidence"

    durable_step = repository.get(
        context.run_id,
        "iteration-1:retrieve_evidence",
    )
    assert durable_step is not None
    assert durable_step.status is AgentRunStepStatus.CANCELLED

    cancelled_events = [
        event
        for event in observer.events
        if event.event_type is AgentExecutionEventType.ORCHESTRATION_STEP_CANCELLED
    ]

    assert len(cancelled_events) == 1
    assert cancelled_events[0].step_id == "iteration-1:retrieve_evidence"
    assert cancelled_events[0].step_index == 0

    assert not any(
        event.event_type is AgentExecutionEventType.AGENT_FAILED for event in observer.events
    )


@pytest.mark.asyncio
async def test_llm_agent_resume_completes_current_orchestration_step_from_after_tool_checkpoint() -> (
    None
):
    definition = AgentDefinition(
        name="enterprise-rag-analyst",
        description="Enterprise RAG analyst.",
        system_prompt="You are an enterprise RAG analyst.",
        model="mock-gpt",
        tool_names=("search",),
    )

    gateway = FakeLLMGateway()

    llm_context = AgentLLMContext(
        gateway,
        AgentLLMConfig(
            model=definition.model,
            system_prompt=definition.system_prompt,
        ),
    )

    tool_registry = InMemoryToolRegistry()
    tool = FakeRAGTool(name="rag.search")
    await tool_registry.register(tool)

    tools = AgentToolContext(
        tool_registry,
        definition,
    )

    context = AgentExecutionContext(
        AgentRequest(
            input="Find information about RAG.",
            user_id="user-123",
            session_id="session-orchestration-resume",
        ),
        tools=tools,
        llm=llm_context,
        run_id="run-orchestration-resume-after-1",
        orchestration_plan=build_enterprise_rag_analyst_plan(),
    )

    checkpoint = AgentExecutionCheckpoint(
        schema_version=AgentExecutionCheckpoint.CURRENT_SCHEMA_VERSION,
        run_id="run-orchestration-resume-after-1",
        agent_name=definition.name,
        session_id="session-orchestration-resume",
        user_id="user-123",
        messages=(
            system_message("You are an enterprise RAG analyst."),
            user_message("Find information about RAG."),
            assistant_tool_call_message(
                tool_calls=(
                    AgentToolCall(
                        call_id="call-completed-1",
                        name="rag.search",
                        arguments={"query": "RAG"},
                    ),
                ),
                content="I searched for the information.",
            ),
            tool_result_message(
                call_id="call-completed-1",
                tool_name="rag.search",
                output={
                    "query": "RAG",
                    "retrieved_count": 2,
                },
            ),
        ),
        tool_round=1,
        position=AgentCheckpointPosition.AFTER_TOOL_EXECUTION,
        metadata={
            "orchestration": {
                "current_step_index": 0,
                "steps": {
                    "retrieve_evidence": {
                        "status": OrchestrationStepStatus.COMPLETED.value,
                        "tool_round": 1,
                    },
                },
            },
        },
    )

    agent = LLMAgent(definition)

    response = await agent.resume(context, checkpoint)

    assert response.output == "Generated answer."
    assert response.metadata["tool_rounds"] == 1
    assert tool.execute_count == 0
    assert len(gateway.requests) == 2

    state = context.orchestration_state

    assert state.current_step is None

    assert state.steps[0].step_id == "iteration-1:retrieve_evidence"
    assert state.steps[0].status is OrchestrationStepStatus.COMPLETED
    assert state.steps[0].tool_round == 1
    assert state.get_step_result("retrieve_evidence") is not None

    assert state.steps[1].step_id == "iteration-1:analyze_evidence"
    assert state.steps[1].status is OrchestrationStepStatus.COMPLETED
    assert state.get_step_result("analyze_evidence") is not None

    assert state.steps[2].step_id == "iteration-1:produce_answer"
    assert state.steps[2].status is OrchestrationStepStatus.COMPLETED
    assert state.get_step_result("produce_answer") is not None


@pytest.mark.asyncio
async def test_llm_agent_resume_cancellation_stops_before_recovery_next_step() -> None:
    definition = AgentDefinition(
        name="enterprise-rag-analyst",
        description="Enterprise RAG analyst.",
        system_prompt="You are an enterprise RAG analyst.",
        model="mock-gpt",
        tool_names=("rag.search",),
    )

    cancellation_requested = asyncio.Event()

    class CancellingGateway(FakeLLMGateway):
        async def route_chat(
            self,
            request: dict[str, Any],
        ) -> dict[str, Any]:
            response = await super().route_chat(request)
            if len(self.requests) == 1:
                cancellation_requested.set()
            return response

    gateway = CancellingGateway()

    llm_context = AgentLLMContext(
        gateway,
        AgentLLMConfig(
            model=definition.model,
            system_prompt=definition.system_prompt,
        ),
    )

    tool_registry = InMemoryToolRegistry()
    tool = FakeRAGTool(name="rag.search")
    await tool_registry.register(tool)

    tools = AgentToolContext(
        tool_registry,
        definition,
    )

    context = AgentExecutionContext(
        AgentRequest(
            input="Find information about RAG.",
            user_id="user-123",
            session_id="session-recovery-next-step-cancellation",
        ),
        tools=tools,
        llm=llm_context,
        run_id="run-recovery-next-step-cancellation",
        orchestration_plan=build_enterprise_rag_analyst_plan(),
        cancellation_requested=cancellation_requested,
    )

    checkpoint = AgentExecutionCheckpoint(
        schema_version=AgentExecutionCheckpoint.CURRENT_SCHEMA_VERSION,
        run_id="run-recovery-next-step-cancellation",
        agent_name=definition.name,
        session_id="session-recovery-next-step-cancellation",
        user_id="user-123",
        messages=(
            system_message("You are an enterprise RAG analyst."),
            user_message("Find information about RAG."),
            assistant_tool_call_message(
                tool_calls=(
                    AgentToolCall(
                        call_id="call-recovery-next-step-cancel-1",
                        name="rag.search",
                        arguments={"query": "RAG"},
                    ),
                ),
                content="I searched for the information.",
            ),
            tool_result_message(
                call_id="call-recovery-next-step-cancel-1",
                tool_name="rag.search",
                output={
                    "query": "RAG",
                    "retrieved_count": 2,
                },
            ),
        ),
        tool_round=1,
        position=AgentCheckpointPosition.AFTER_TOOL_EXECUTION,
        metadata={
            "orchestration": {
                "current_step_index": 0,
                "steps": {
                    "retrieve_evidence": {
                        "status": OrchestrationStepStatus.COMPLETED.value,
                        "tool_round": 1,
                    },
                },
            },
        },
    )

    agent = LLMAgent(definition)

    with pytest.raises(
        asyncio.CancelledError,
        match="Agent execution cancellation requested.",
    ):
        await agent.resume(context, checkpoint)

    state = context.orchestration_state

    assert state.steps[0].step_id == "iteration-1:retrieve_evidence"
    assert state.steps[0].status is OrchestrationStepStatus.COMPLETED

    assert state.steps[1].step_id == "iteration-1:analyze_evidence"
    assert state.steps[1].status is OrchestrationStepStatus.COMPLETED

    assert state.steps[2].step_id == "iteration-1:produce_answer"
    assert state.steps[2].status is OrchestrationStepStatus.PENDING

    assert state.current_step_index == 1
    assert tool.execute_count == 0
    assert len(gateway.requests) == 1


@pytest.mark.asyncio
async def test_llm_agent_resume_cancellation_stops_before_runtime_replanned_step() -> None:
    class ReplanningPlanProvider:
        def __init__(self) -> None:
            self.plans: list[OrchestrationPlan] = []

        def build_plan(
            self,
            context: AgentExecutionContext,
        ) -> OrchestrationPlan:
            plan_number = len(self.plans) + 2
            plan = OrchestrationPlan(
                steps=(
                    OrchestrationStep(
                        step_id=f"answer-{plan_number}",
                        step_index=0,
                        name="Produce answer",
                        status=OrchestrationStepStatus.PENDING,
                        completion_policy=(OrchestrationStepCompletionPolicy.ON_AGENT_RESPONSE),
                    ),
                )
            )
            self.plans.append(plan)
            return plan

    class ReplanningDecisionProvider:
        def __init__(self, cancellation_requested: asyncio.Event) -> None:
            self.cancellation_requested = cancellation_requested
            self.results: list[AgentRuntimeDecision] = []

        def evaluate(
            self,
            context: AgentExecutionContext,
            evaluation: AgentRuntimeEvaluationSnapshot,
        ):
            decision = (
                AgentRuntimeDecision.CONTINUE if not self.results else AgentRuntimeDecision.STOP
            )
            self.results.append(decision)

            if decision is AgentRuntimeDecision.CONTINUE:
                self.cancellation_requested.set()

            from ai_platform.agents.decision_provider import (
                AgentRuntimeDecisionReason,
                AgentRuntimeDecisionResult,
            )

            return AgentRuntimeDecisionResult(
                decision=decision,
                reason=AgentRuntimeDecisionReason(
                    "iteration_budget_remaining"
                    if decision is AgentRuntimeDecision.CONTINUE
                    else "iteration_budget_exhausted"
                ),
            )

    definition = AgentDefinition(
        name="enterprise-rag-analyst",
        description="Enterprise RAG analyst.",
        system_prompt="You are an enterprise RAG analyst.",
        model="gpt-test",
        tool_names=("rag.search",),
    )

    cancellation_requested = asyncio.Event()
    plan_provider = ReplanningPlanProvider()
    decision_provider = ReplanningDecisionProvider(cancellation_requested)

    gateway = FakeLLMGateway()

    llm_context = AgentLLMContext(
        gateway,
        AgentLLMConfig(
            model=definition.model,
            system_prompt=definition.system_prompt,
        ),
    )

    tool_registry = InMemoryToolRegistry()
    tool = FakeRAGTool(name="rag.search")
    await tool_registry.register(tool)

    recovery_plan = OrchestrationPlan(
        steps=(
            OrchestrationStep(
                step_id="retrieve_evidence",
                step_index=0,
                name="Retrieve enterprise evidence",
                status=OrchestrationStepStatus.PENDING,
                completion_policy=OrchestrationStepCompletionPolicy.ON_TOOL_RESULT,
                metadata={
                    "phase": "evidence_retrieval",
                    "completion_tool_name": "rag.search",
                },
            ),
            OrchestrationStep(
                step_id="produce_answer",
                step_index=1,
                name="Produce grounded answer",
                status=OrchestrationStepStatus.PENDING,
                completion_policy=OrchestrationStepCompletionPolicy.ON_AGENT_RESPONSE,
                metadata={"phase": "response_generation"},
            ),
        )
    )

    context = AgentExecutionContext(
        AgentRequest(
            input="Find information about RAG.",
            user_id="user-123",
            session_id="session-recovery-replanning-cancellation",
        ),
        tools=AgentToolContext(
            tool_registry,
            definition,
        ),
        llm=llm_context,
        run_id="run-recovery-replanning-cancellation",
        orchestration_plan=recovery_plan,
        cancellation_requested=cancellation_requested,
    )

    context.plan_provider = plan_provider
    context.decision_provider = decision_provider

    checkpoint = AgentExecutionCheckpoint(
        schema_version=AgentExecutionCheckpoint.CURRENT_SCHEMA_VERSION,
        run_id="run-recovery-replanning-cancellation",
        agent_name=definition.name,
        session_id="session-recovery-replanning-cancellation",
        user_id="user-123",
        messages=(
            system_message("You are an enterprise RAG analyst."),
            user_message("Find information about RAG."),
            assistant_tool_call_message(
                tool_calls=(
                    AgentToolCall(
                        call_id="call-recovery-replanning-cancel-1",
                        name="rag.search",
                        arguments={"query": "RAG"},
                    ),
                ),
                content="I searched for the information.",
            ),
            tool_result_message(
                call_id="call-recovery-replanning-cancel-1",
                tool_name="rag.search",
                output={
                    "query": "RAG",
                    "retrieved_count": 2,
                },
            ),
        ),
        tool_round=1,
        position=AgentCheckpointPosition.AFTER_TOOL_EXECUTION,
        metadata={
            "orchestration": {
                "current_step_index": 0,
                "steps": {
                    "retrieve_evidence": {
                        "status": OrchestrationStepStatus.COMPLETED.value,
                        "tool_round": 1,
                    },
                },
            },
        },
    )

    agent = LLMAgent(definition)

    with pytest.raises(
        asyncio.CancelledError,
        match="Agent execution cancellation requested.",
    ):
        await agent.resume(context, checkpoint)

    assert decision_provider.results == [AgentRuntimeDecision.CONTINUE]
    assert plan_provider.plans == []

    state = context.orchestration_state

    assert state.steps[0].step_id == "iteration-1:retrieve_evidence"
    assert state.steps[0].status is OrchestrationStepStatus.COMPLETED

    assert state.steps[1].step_id == "iteration-1:produce_answer"
    assert state.steps[1].status is OrchestrationStepStatus.COMPLETED

    assert state.current_step is None
    assert len(gateway.requests) == 1
    assert tool.execute_count == 0


@pytest.mark.asyncio
async def test_llm_agent_resume_cancellation_stops_before_next_orchestration_step() -> None:
    definition = AgentDefinition(
        name="enterprise-rag-analyst",
        description="Enterprise RAG analyst.",
        system_prompt="You are an enterprise RAG analyst.",
        model="mock-gpt",
        tool_names=("rag.search",),
    )

    gateway = FakeLLMGateway()

    llm_context = AgentLLMContext(
        gateway,
        AgentLLMConfig(
            model=definition.model,
            system_prompt=definition.system_prompt,
        ),
    )

    tool_registry = InMemoryToolRegistry()
    tool = FakeRAGTool(name="rag.search")
    await tool_registry.register(tool)

    tools = AgentToolContext(
        tool_registry,
        definition,
    )

    cancellation_requested = asyncio.Event()

    context = AgentExecutionContext(
        AgentRequest(
            input="Find information about RAG.",
            user_id="user-123",
            session_id="session-orchestration-resume-cancellation",
        ),
        tools=tools,
        llm=llm_context,
        run_id="run-orchestration-resume-cancellation",
        orchestration_plan=build_enterprise_rag_analyst_plan(),
        cancellation_requested=cancellation_requested,
    )

    checkpoint = AgentExecutionCheckpoint(
        schema_version=AgentExecutionCheckpoint.CURRENT_SCHEMA_VERSION,
        run_id="run-orchestration-resume-cancellation",
        agent_name=definition.name,
        session_id="session-orchestration-resume-cancellation",
        user_id="user-123",
        messages=(
            system_message("You are an enterprise RAG analyst."),
            user_message("Find information about RAG."),
            assistant_tool_call_message(
                tool_calls=(
                    AgentToolCall(
                        call_id="call-resume-cancel-1",
                        name="rag.search",
                        arguments={"query": "RAG"},
                    ),
                ),
                content="I searched for the information.",
            ),
            tool_result_message(
                call_id="call-resume-cancel-1",
                tool_name="rag.search",
                output={
                    "query": "RAG",
                    "retrieved_count": 2,
                },
            ),
        ),
        tool_round=1,
        position=AgentCheckpointPosition.AFTER_TOOL_EXECUTION,
        metadata={
            "orchestration": {
                "current_step_index": 0,
                "steps": {
                    "retrieve_evidence": {
                        "status": OrchestrationStepStatus.COMPLETED.value,
                        "tool_round": 1,
                    },
                },
            },
        },
    )

    cancellation_requested.set()

    agent = LLMAgent(definition)

    with pytest.raises(
        asyncio.CancelledError,
        match="Agent execution cancellation requested.",
    ):
        await agent.resume(context, checkpoint)

    state = context.orchestration_state

    assert state.steps[0].step_id == "iteration-1:retrieve_evidence"
    assert state.steps[0].status is OrchestrationStepStatus.COMPLETED

    assert state.steps[1].step_id == "iteration-1:analyze_evidence"
    assert state.steps[1].status is OrchestrationStepStatus.PENDING

    assert state.steps[2].step_id == "iteration-1:produce_answer"
    assert state.steps[2].status is OrchestrationStepStatus.PENDING

    assert state.current_step_index == 0
    assert tool.execute_count == 0
    assert len(gateway.requests) == 0


@pytest.mark.asyncio
async def test_llm_agent_resume_completes_current_orchestration_step_from_before_tool_checkpoint() -> (
    None
):
    definition = AgentDefinition(
        name="enterprise-rag-analyst",
        description="Enterprise RAG analyst.",
        system_prompt="You are an enterprise RAG analyst.",
        model="mock-gpt",
        tool_names=("rag.search",),
    )

    gateway = FakeLLMGateway()

    llm_context = AgentLLMContext(
        gateway,
        AgentLLMConfig(
            model=definition.model,
            system_prompt=definition.system_prompt,
        ),
    )

    tool_registry = InMemoryToolRegistry()
    tool = FakeRAGTool(name="rag.search")
    await tool_registry.register(tool)

    tools = AgentToolContext(
        tool_registry,
        definition,
    )

    context = AgentExecutionContext(
        AgentRequest(
            input="Find information about RAG.",
            user_id="user-123",
            session_id="session-orchestration-resume",
        ),
        tools=tools,
        llm=llm_context,
        run_id="run-orchestration-resume-before-1",
        orchestration_plan=build_enterprise_rag_analyst_plan(),
    )

    checkpoint = AgentExecutionCheckpoint(
        schema_version=AgentExecutionCheckpoint.CURRENT_SCHEMA_VERSION,
        run_id="run-orchestration-resume-before-1",
        agent_name=definition.name,
        session_id="session-orchestration-resume",
        user_id="user-123",
        messages=(
            system_message("You are an enterprise RAG analyst."),
            user_message("Find information about RAG."),
            assistant_tool_call_message(
                tool_calls=(
                    AgentToolCall(
                        call_id="call-recover-1",
                        name="rag.search",
                        arguments={"query": "RAG"},
                    ),
                ),
                content="I searched for the information.",
            ),
        ),
        tool_round=1,
        position=AgentCheckpointPosition.BEFORE_TOOL_EXECUTION,
        metadata={
            "orchestration": {
                "current_step_index": 0,
                "steps": {
                    "retrieve_evidence": {
                        "status": OrchestrationStepStatus.RUNNING.value,
                        "tool_round": None,
                    },
                },
            },
        },
        execution_budget_state=ExecutionBudgetState(
            llm_calls=1,
            tool_calls=1,
            tool_rounds=1,
        ),
    )

    agent = LLMAgent(definition)

    response = await agent.resume(context, checkpoint)

    assert response.output == "Generated answer."
    assert response.metadata["tool_rounds"] == 1
    assert tool.execute_count == 1
    assert len(gateway.requests) == 2

    state = context.orchestration_state

    assert state.current_step is None

    assert state.steps[0].step_id == "iteration-1:retrieve_evidence"
    assert state.steps[0].status is OrchestrationStepStatus.COMPLETED
    assert state.steps[0].tool_round == 1
    assert state.get_step_result("retrieve_evidence") is not None

    assert state.steps[1].step_id == "iteration-1:analyze_evidence"
    assert state.steps[1].status is OrchestrationStepStatus.COMPLETED
    assert state.get_step_result("analyze_evidence") is not None

    assert state.steps[2].step_id == "iteration-1:produce_answer"
    assert state.steps[2].status is OrchestrationStepStatus.COMPLETED
    assert state.get_step_result("produce_answer") is not None


@pytest.mark.asyncio
async def test_llm_agent_resume_rejects_non_running_current_orchestration_step() -> None:
    definition = AgentDefinition(
        name="enterprise-rag-analyst",
        description="Enterprise RAG analyst.",
        system_prompt="You are an enterprise RAG analyst.",
        model="mock-gpt",
        tool_names=("rag.search",),
    )

    gateway = FakeLLMGateway()

    llm_context = AgentLLMContext(
        gateway,
        AgentLLMConfig(
            model=definition.model,
            system_prompt=definition.system_prompt,
        ),
    )

    tool_registry = InMemoryToolRegistry()
    tool = FakeRAGTool(name="rag.search")
    await tool_registry.register(tool)

    tools = AgentToolContext(
        tool_registry,
        definition,
    )

    context = AgentExecutionContext(
        AgentRequest(
            input="Find information about RAG.",
            user_id="user-123",
            session_id="session-orchestration-invalid",
        ),
        tools=tools,
        llm=llm_context,
        run_id="run-orchestration-invalid-1",
        orchestration_plan=build_enterprise_rag_analyst_plan(),
    )

    checkpoint = AgentExecutionCheckpoint(
        schema_version=AgentExecutionCheckpoint.CURRENT_SCHEMA_VERSION,
        run_id="run-orchestration-invalid-1",
        agent_name=definition.name,
        session_id="session-orchestration-invalid",
        user_id="user-123",
        messages=(
            system_message("You are an enterprise RAG analyst."),
            user_message("Find information about RAG."),
        ),
        tool_round=0,
        position=AgentCheckpointPosition.BEFORE_TOOL_EXECUTION,
        metadata={
            "orchestration": {
                "current_step_index": 0,
                "steps": {
                    "retrieve_evidence": {
                        "status": OrchestrationStepStatus.COMPLETED.value,
                        "tool_round": 1,
                    },
                },
            },
        },
    )

    agent = LLMAgent(definition)

    with pytest.raises(
        ValueError,
        match="current step must be RUNNING before tool execution",
    ):
        await agent.resume(context, checkpoint)


@pytest.mark.asyncio
async def test_llm_agent_resume_rejects_completed_step_after_current_step() -> None:
    definition = AgentDefinition(
        name="enterprise-rag-analyst",
        description="Enterprise RAG analyst.",
        system_prompt="You are an enterprise RAG analyst.",
        model="mock-gpt",
        tool_names=("rag.search",),
    )

    gateway = FakeLLMGateway()

    llm_context = AgentLLMContext(
        gateway,
        AgentLLMConfig(
            model=definition.model,
            system_prompt=definition.system_prompt,
        ),
    )

    tool_registry = InMemoryToolRegistry()
    tool = FakeRAGTool(name="rag.search")
    await tool_registry.register(tool)

    tools = AgentToolContext(
        tool_registry,
        definition,
    )

    context = AgentExecutionContext(
        AgentRequest(
            input="Find information about RAG.",
            user_id="user-123",
            session_id="session-orchestration-invalid",
        ),
        tools=tools,
        llm=llm_context,
        run_id="run-orchestration-invalid-2",
        orchestration_plan=build_enterprise_rag_analyst_plan(),
    )

    checkpoint = AgentExecutionCheckpoint(
        schema_version=AgentExecutionCheckpoint.CURRENT_SCHEMA_VERSION,
        run_id="run-orchestration-invalid-2",
        agent_name=definition.name,
        session_id="session-orchestration-invalid",
        user_id="user-123",
        messages=(
            system_message("You are an enterprise RAG analyst."),
            user_message("Find information about RAG."),
        ),
        tool_round=0,
        position=AgentCheckpointPosition.BEFORE_TOOL_EXECUTION,
        metadata={
            "orchestration": {
                "current_step_index": 0,
                "steps": {
                    "retrieve_evidence": {
                        "status": OrchestrationStepStatus.RUNNING.value,
                        "tool_round": None,
                    },
                    "analyze_evidence": {
                        "status": OrchestrationStepStatus.COMPLETED.value,
                        "tool_round": 1,
                    },
                },
            },
        },
    )

    agent = LLMAgent(definition)

    with pytest.raises(
        ValueError,
        match="steps after the current step must be PENDING",
    ):
        await agent.resume(context, checkpoint)


@pytest.mark.asyncio
async def test_llm_agent_uses_pinned_model_governance_decision() -> None:
    definition = AgentDefinition(
        name="production-llm-agent",
        description="Production LLM agent.",
        system_prompt="You are a production LLM agent.",
        model="agent-config-model",
    )

    gateway = FakeLLMGateway()

    llm_context = AgentLLMContext(
        gateway,
        AgentLLMConfig(
            model=definition.model,
            system_prompt=definition.system_prompt,
        ),
    )

    tools = AgentToolContext(
        InMemoryToolRegistry(),
        definition,
    )

    decision = ModelGovernanceDecision(
        effective_model="governed-model",
        effective_provider="governed-provider",
        policy_id="policy-enterprise-models",
        policy_version="v7",
    )

    context = AgentExecutionContext(
        AgentRequest(
            input="Use the governed model.",
            model_governance=decision,
        ),
        tools=tools,
        llm=llm_context,
        run_id="run-model-governance-1",
    )

    agent = LLMAgent(definition)

    response = await agent.run(context)

    assert response.output == "Generated answer."
    assert len(gateway.requests) == 1
    assert gateway.requests[0]["model"] == "governed-model"
    assert gateway.requests[0]["provider"] == "governed-provider"
    assert gateway.requests[0]["model"] != definition.model


@pytest.mark.asyncio
async def test_llm_agent_records_provider_reported_token_usage() -> None:
    definition = AgentDefinition(
        name="production-llm-agent",
        description="Production LLM agent.",
        system_prompt="You are a production LLM agent.",
        model="mock-gpt",
    )

    gateway = FakeLLMGateway()

    llm_context = AgentLLMContext(
        gateway,
        AgentLLMConfig(
            model=definition.model,
            system_prompt=definition.system_prompt,
        ),
    )

    tools = AgentToolContext(
        InMemoryToolRegistry(),
        definition,
    )

    context = AgentExecutionContext(
        AgentRequest(
            input="Track my token usage.",
            execution_budget=ExecutionBudget(max_tokens_per_run=100),
        ),
        tools=tools,
        llm=llm_context,
        run_id="run-token-usage-1",
    )

    agent = LLMAgent(definition)

    response = await agent.run(context)

    assert response.output == "Generated answer."
    assert context.execution_budget_state.total_tokens == 15
    assert len(gateway.requests) == 1


@pytest.mark.asyncio
async def test_llm_agent_rejects_exhausted_token_budget_before_provider_call() -> None:
    definition = AgentDefinition(
        name="production-llm-agent",
        description="Production LLM agent.",
        system_prompt="You are a production LLM agent.",
        model="mock-gpt",
    )

    gateway = FakeLLMGateway()

    llm_context = AgentLLMContext(
        gateway,
        AgentLLMConfig(
            model=definition.model,
            system_prompt=definition.system_prompt,
        ),
    )

    tools = AgentToolContext(
        InMemoryToolRegistry(),
        definition,
    )

    context = AgentExecutionContext(
        AgentRequest(
            input="Continue execution.",
            execution_budget=ExecutionBudget(max_tokens_per_run=100),
        ),
        tools=tools,
        llm=llm_context,
        run_id="run-token-exhausted-1",
    )

    context.execution_budget_state.total_tokens = 100

    agent = LLMAgent(definition)

    with pytest.raises(
        AgentTokenLimitError,
        match=r"maximum token usage \(100; actual: 100\)",
    ):
        await agent.run(context)

    assert len(gateway.requests) == 0
    assert context.execution_budget_state.total_tokens == 100


@pytest.mark.asyncio
async def test_llm_agent_passes_remaining_token_budget_to_gateway() -> None:
    definition = AgentDefinition(
        name="production-llm-agent",
        description="Production LLM agent.",
        system_prompt="You are a production LLM agent.",
        model="mock-gpt",
    )

    gateway = FakeLLMGateway()

    llm_context = AgentLLMContext(
        gateway,
        AgentLLMConfig(
            model=definition.model,
            system_prompt=definition.system_prompt,
        ),
    )

    tools = AgentToolContext(
        InMemoryToolRegistry(),
        definition,
    )

    context = AgentExecutionContext(
        AgentRequest(
            input="Use the remaining budget.",
            execution_budget=ExecutionBudget(max_tokens_per_run=50),
        ),
        tools=tools,
        llm=llm_context,
        run_id="run-token-cap-1",
    )

    context.execution_budget_state.total_tokens = 20

    agent = LLMAgent(definition)

    await agent.run(context)

    assert len(gateway.requests) == 1
    assert gateway.requests[0]["max_tokens"] == 30
    assert context.execution_budget_state.total_tokens == 35


@pytest.mark.asyncio
async def test_llm_agent_respects_lower_configured_max_tokens() -> None:
    definition = AgentDefinition(
        name="production-llm-agent",
        description="Production LLM agent.",
        system_prompt="You are a production LLM agent.",
        model="mock-gpt",
    )

    gateway = FakeLLMGateway()

    llm_context = AgentLLMContext(
        gateway,
        AgentLLMConfig(
            model=definition.model,
            system_prompt=definition.system_prompt,
            max_tokens=10,
        ),
    )

    tools = AgentToolContext(
        InMemoryToolRegistry(),
        definition,
    )

    context = AgentExecutionContext(
        AgentRequest(
            input="Respect the configured completion limit.",
            execution_budget=ExecutionBudget(max_tokens_per_run=50),
        ),
        tools=tools,
        llm=llm_context,
        run_id="run-token-configured-cap-1",
    )

    agent = LLMAgent(definition)

    await agent.run(context)

    assert len(gateway.requests) == 1
    assert gateway.requests[0]["max_tokens"] == 10
    assert context.execution_budget_state.total_tokens == 15


@pytest.mark.asyncio
async def test_llm_agent_rejects_provider_usage_that_exceeds_token_budget() -> None:
    definition = AgentDefinition(
        name="production-llm-agent",
        description="Production LLM agent.",
        system_prompt="You are a production LLM agent.",
        model="mock-gpt",
    )

    gateway = FakeLLMGateway()

    llm_context = AgentLLMContext(
        gateway,
        AgentLLMConfig(
            model=definition.model,
            system_prompt=definition.system_prompt,
        ),
    )

    tools = AgentToolContext(
        InMemoryToolRegistry(),
        definition,
    )

    context = AgentExecutionContext(
        AgentRequest(
            input="Use the remaining token budget.",
            execution_budget=ExecutionBudget(max_tokens_per_run=20),
        ),
        tools=tools,
        llm=llm_context,
        run_id="run-token-overage-1",
    )

    context.execution_budget_state.total_tokens = 10

    agent = LLMAgent(definition)

    with pytest.raises(
        AgentTokenLimitError,
        match=r"maximum token usage \(20; actual: 25\)",
    ):
        await agent.run(context)

    # The provider did execute, so its authoritative usage remains recorded.
    assert context.execution_budget_state.total_tokens == 25

    # The remaining budget was passed as the completion cap.
    assert len(gateway.requests) == 1
    assert gateway.requests[0]["max_tokens"] == 10


@pytest.mark.asyncio
async def test_llm_agent_resume_restores_execution_budget_state() -> None:
    definition = AgentDefinition(
        name="production-llm-agent",
        description="Production LLM agent.",
        system_prompt="You are a production LLM agent.",
        model="mock-gpt",
        tool_names=(),
    )

    gateway = FakeLLMGateway()

    llm_context = AgentLLMContext(
        gateway,
        AgentLLMConfig(
            model=definition.model,
            system_prompt=definition.system_prompt,
        ),
    )

    tools = AgentToolContext(
        InMemoryToolRegistry(),
        definition,
    )

    context = AgentExecutionContext(
        AgentRequest(
            input="Continue the execution.",
            user_id="user-123",
            session_id="session-456",
            execution_budget=ExecutionBudget(
                max_llm_calls=1,
                max_tool_calls=5,
                max_tool_rounds=3,
                max_duration_seconds=300.0,
            ),
        ),
        tools=tools,
        llm=llm_context,
        run_id="run-budget-resume-1",
    )

    checkpoint = AgentExecutionCheckpoint(
        schema_version=AgentExecutionCheckpoint.CURRENT_SCHEMA_VERSION,
        run_id="run-budget-resume-1",
        agent_name=definition.name,
        session_id="session-456",
        user_id="user-123",
        messages=(
            system_message("You are a production LLM agent."),
            user_message("Continue the execution."),
        ),
        tool_round=0,
        position=AgentCheckpointPosition.AFTER_TOOL_EXECUTION,
        metadata={},
        execution_budget_state=ExecutionBudgetState(
            llm_calls=1,
            tool_calls=0,
            tool_rounds=0,
        ),
    )

    agent = LLMAgent(definition)

    with pytest.raises(
        AgentLLMCallLimitError,
        match="Agent 'production-llm-agent' exceeded the maximum LLM calls \\(1\\)",
    ):
        await agent.resume(context, checkpoint)

    assert len(gateway.requests) == 0
    assert context.execution_budget_state.llm_calls == 1
    assert context.execution_budget_state.tool_calls == 0
    assert context.execution_budget_state.tool_rounds == 0


@pytest.mark.asyncio
async def test_llm_agent_resume_continues_with_new_tool_call_and_checkpoint() -> None:
    definition = AgentDefinition(
        name="production-llm-agent",
        description="Production LLM agent.",
        system_prompt="You are a production LLM agent.",
        model="mock-gpt",
        tool_names=("search",),
    )

    gateway = FakeResumeContinuationLLMGateway()
    checkpoint_handler = FakeAgentCheckpointHandler()

    llm_context = AgentLLMContext(
        gateway,
        AgentLLMConfig(
            model=definition.model,
            system_prompt=definition.system_prompt,
        ),
    )

    tool_registry = InMemoryToolRegistry()
    tool = FakeRAGTool(name="search")
    await tool_registry.register(tool)

    tools = AgentToolContext(
        tool_registry,
        definition,
    )

    context = AgentExecutionContext(
        AgentRequest(
            input="Find information about RAG.",
            user_id="user-123",
            session_id="session-456",
            metadata={"request_id": "request-789"},
        ),
        tools=tools,
        llm=llm_context,
        run_id="run-resume-2",
    )

    original_checkpoint = AgentExecutionCheckpoint(
        schema_version=AgentExecutionCheckpoint.CURRENT_SCHEMA_VERSION,
        run_id="run-resume-2",
        agent_name=definition.name,
        session_id="session-456",
        user_id="user-123",
        messages=(
            system_message("You are a production LLM agent."),
            user_message("Find information about RAG."),
            assistant_tool_call_message(
                tool_calls=(
                    AgentToolCall(
                        call_id="call-1",
                        name="search",
                        arguments={"query": "RAG"},
                    ),
                ),
                content="I searched for the initial information.",
            ),
            tool_result_message(
                call_id="call-1",
                tool_name="search",
                output={
                    "query": "RAG",
                    "retrieved_count": 2,
                },
            ),
        ),
        tool_round=1,
        position=AgentCheckpointPosition.AFTER_TOOL_EXECUTION,
        metadata={"request_id": "request-789"},
    )

    agent = LLMAgent(
        definition,
        checkpoint_handler=checkpoint_handler,
    )

    response = await agent.resume(context, original_checkpoint)

    assert response.output == "Resumed execution completed successfully."
    assert response.metadata["tool_rounds"] == 2

    assert tool.execute_count == 1
    assert len(gateway.requests) == 2

    assert gateway.requests[0]["messages"][-1] == {
        "role": "tool",
        "content": (
            '{"call_id": "call-1", '
            '"output": {"query": "RAG", "retrieved_count": 2}, '
            '"success": true, "tool_name": "search"}'
        ),
        "tool_call_id": "call-1",
        "tool_name": "search",
    }

    resumed_tool_result = gateway.requests[1]["messages"][-1]

    assert resumed_tool_result["role"] == "tool"
    assert resumed_tool_result["tool_call_id"] == "call-2"
    assert resumed_tool_result["tool_name"] == "search"
    assert '"call_id": "call-2"' in resumed_tool_result["content"]
    assert '"query": "follow-up"' in resumed_tool_result["content"]
    assert '"success": true' in resumed_tool_result["content"]

    assert len(checkpoint_handler.checkpoints) == 2

    before_checkpoint = checkpoint_handler.checkpoints[0]

    assert before_checkpoint.run_id == original_checkpoint.run_id
    assert before_checkpoint.agent_name == original_checkpoint.agent_name
    assert before_checkpoint.tool_round == 2
    assert before_checkpoint.position is AgentCheckpointPosition.BEFORE_TOOL_EXECUTION
    assert before_checkpoint.metadata["request_id"] == "request-789"
    assert before_checkpoint.metadata["runtime"]["iteration"] == 1
    assert len(before_checkpoint.messages) == 5
    assert before_checkpoint.messages[:4] == original_checkpoint.messages
    assert before_checkpoint.messages[4] == assistant_tool_call_message(
        tool_calls=(
            AgentToolCall(
                call_id="call-2",
                name="search",
                arguments={"query": "follow-up"},
            ),
        ),
        content="",
    )

    new_checkpoint = checkpoint_handler.checkpoints[1]

    assert new_checkpoint.run_id == original_checkpoint.run_id
    assert new_checkpoint.agent_name == original_checkpoint.agent_name
    assert new_checkpoint.tool_round == 2
    assert new_checkpoint.position is AgentCheckpointPosition.AFTER_TOOL_EXECUTION
    assert new_checkpoint.metadata["request_id"] == "request-789"
    assert new_checkpoint.metadata["runtime"]["iteration"] == 1
    assert len(new_checkpoint.messages) == 6

    assert new_checkpoint.messages[:4] == original_checkpoint.messages
    assert new_checkpoint.messages[4] == assistant_tool_call_message(
        tool_calls=(
            AgentToolCall(
                call_id="call-2",
                name="search",
                arguments={"query": "follow-up"},
            ),
        ),
        content="",
    )
    assert new_checkpoint.messages[5].role is AgentMessageRole.TOOL
    assert '"call_id": "call-2"' in new_checkpoint.messages[5].content
    assert '"query": "follow-up"' in new_checkpoint.messages[5].content


@pytest.mark.asyncio
async def test_llm_agent_resume_rejects_mismatched_agent() -> None:
    context, _ = make_context()
    context = AgentExecutionContext(
        context.request,
        tools=context.tools,
        llm=context.llm,
        run_id="run-1",
    )

    checkpoint = AgentExecutionCheckpoint(
        schema_version=1,
        run_id="run-1",
        agent_name="different-agent",
        session_id=None,
        user_id=None,
        messages=(system_message("System."),),
        tool_round=1,
        position=AgentCheckpointPosition.AFTER_TOOL_EXECUTION,
        metadata={},
    )

    agent = LLMAgent(
        AgentDefinition(
            name="test-llm-agent",
            description="Test agent.",
            system_prompt="System.",
            model="mock-gpt",
        )
    )

    with pytest.raises(
        ValueError,
        match="agent_name does not match",
    ):
        await agent.resume(context, checkpoint)


@pytest.mark.asyncio
async def test_llm_agent_resume_rejects_mismatched_run_id() -> None:
    context, _ = make_context()
    context = AgentExecutionContext(
        context.request,
        tools=context.tools,
        llm=context.llm,
        run_id="run-context",
    )

    checkpoint = AgentExecutionCheckpoint(
        schema_version=1,
        run_id="run-checkpoint",
        agent_name="test-llm-agent",
        session_id=None,
        user_id=None,
        messages=(system_message("System."),),
        tool_round=1,
        position=AgentCheckpointPosition.AFTER_TOOL_EXECUTION,
        metadata={},
    )

    agent = LLMAgent(
        AgentDefinition(
            name="test-llm-agent",
            description="Test agent.",
            system_prompt="System.",
            model="mock-gpt",
        )
    )

    with pytest.raises(
        ValueError,
        match="run_id does not match",
    ):
        await agent.resume(context, checkpoint)
