from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from unittest.mock import MagicMock

import pytest

from ai_platform.agents.exceptions import AgentExecutionOwnershipLostError
from ai_platform.agents.execution import AgentExecutionContext
from ai_platform.agents.models import AgentDefinition, AgentRequest, AgentResponse
from ai_platform.agents.registry.in_memory import InMemoryAgentRegistry
from ai_platform.agents.runtime import AgentRuntime
from ai_platform.agents.tool_calls import AgentToolCall
from app.control_plane.agent_run_events.tool_authorization_observer import (
    ToolAuthorizationAuditObserver,
)
from ai_platform.agents.observability import AgentExecutionEventType
from data_platform.vehicle.service import VehicleDataService
from tools.authorization.in_memory import InMemoryToolAuthorizer
from tools.authorization.service import ToolAuthorizationService
from tools.execution.context import ToolExecutionContext
from tools.execution.idempotency import (
    InMemoryToolExecutionIdempotencyStore,
    ToolExecutionIdempotencyKey,
    ToolIdempotencyClaimStatus,
)
from tools.execution.service import ToolExecutionService
from tools.models import ToolExecutionFailureCategory
from tools.registry.in_memory import InMemoryToolRegistry
from tools.vehicle.data_query import VehicleDataQueryTool


class _FakeSparkColumn:
    def __init__(self, name: str) -> None:
        self.name = name

    def __eq__(self, other):
        return ("eq", self.name, other)

    def __ge__(self, other):
        return ("ge", self.name, other)

    def __le__(self, other):
        return ("le", self.name, other)

    def asc(self):
        return ("asc", self.name)


class FakeAgentExecutionObserver:
    def __init__(self) -> None:
        self.events = []

    async def record(self, event) -> None:
        self.events.append(event)


class VehicleCallingAgent:
    def __init__(self, definition: AgentDefinition) -> None:
        self._definition = definition
        self.last_tool_results = None

    @property
    def definition(self) -> AgentDefinition:
        return self._definition

    async def run(
        self,
        context: AgentExecutionContext,
    ) -> AgentResponse:
        self.last_tool_results = await context.execute_tool_calls(
            (
                AgentToolCall(
                    call_id="vehicle-call-1",
                    name="vehicle.data.query",
                    arguments={
                        "vehicle_id": "veh-123",
                        "start_time": "2026-01-01T00:00:00+00:00",
                        "end_time": "2026-01-02T00:00:00+00:00",
                        "limit": 10,
                    },
                ),
            )
        )

        return AgentResponse(
            agent_name=self.definition.name,
            output=self.last_tool_results,
            session_id=context.request.session_id,
        )


def _build_vehicle_service():
    reader = MagicMock()
    spark = MagicMock()
    dataframe = MagicMock()

    reader.read.return_value = dataframe

    dataframe.vehicle_id = _FakeSparkColumn("vehicle_id")
    dataframe.event_time = _FakeSparkColumn("event_time")

    dataframe.select.return_value = dataframe
    dataframe.where.return_value = dataframe
    dataframe.orderBy.return_value = dataframe
    dataframe.limit.return_value = dataframe

    event_time = datetime(
        2026,
        1,
        1,
        12,
        0,
        tzinfo=timezone.utc,
    )

    dataframe.collect.return_value = [
        {
            "vehicle_id": "veh-123",
            "event_time": event_time,
            "speed": 52.5,
        }
    ]

    return (
        VehicleDataService(
            reader=reader,
            spark=spark,
        ),
        reader,
        spark,
        dataframe,
    )


class OwnershipLosingVehicleDataQueryTool(VehicleDataQueryTool):
    def __init__(
        self,
        service: VehicleDataService,
        ownership_lost: asyncio.Event,
    ) -> None:
        super().__init__(service)
        self._ownership_lost = ownership_lost
        self.execution_count = 0

    async def execute_with_context(self, arguments, context):
        self.execution_count += 1

        result = await super().execute_with_context(arguments, context)

        # Simulate the enterprise data operation completing before the
        # worker discovers that durable run ownership has been lost.
        self._ownership_lost.set()

        return result


async def _build_runtime(observer=None):
    vehicle_service, reader, spark, dataframe = _build_vehicle_service()

    vehicle_tool = VehicleDataQueryTool(vehicle_service)

    tool_registry = InMemoryToolRegistry()
    await tool_registry.register(vehicle_tool)

    authorizer = InMemoryToolAuthorizer()
    await authorizer.allow(
        "enterprise-demo-user",
        "vehicle.data.query",
    )

    authorization_service = ToolAuthorizationService(authorizer)

    idempotency_store = InMemoryToolExecutionIdempotencyStore()

    execution_service = ToolExecutionService(
        tool_registry,
        authorization_service=authorization_service,
        audit_sink=(ToolAuthorizationAuditObserver(observer) if observer is not None else None),
        idempotency_store=idempotency_store,
    )

    agent_definition = AgentDefinition(
        name="enterprise-rag-analyst",
        description="Enterprise agent for governed vehicle telemetry evidence.",
        system_prompt=(
            "Use vehicle.data.query when structured vehicle telemetry " "evidence is required."
        ),
        model="mock-gpt",
        tool_names=("vehicle.data.query",),
    )

    agent = VehicleCallingAgent(agent_definition)

    agent_registry = InMemoryAgentRegistry()
    await agent_registry.register(agent)

    runtime = AgentRuntime(
        agent_registry,
        tool_execution_service=execution_service,
    )

    return (
        runtime,
        agent,
        reader,
        spark,
        dataframe,
    )


@pytest.mark.asyncio
async def test_agent_runtime_executes_vehicle_tool_with_authorization_and_idempotency():
    observer = FakeAgentExecutionObserver()

    (
        runtime,
        agent,
        reader,
        spark,
        dataframe,
    ) = await _build_runtime(observer=observer)

    request = AgentRequest(
        input="Show vehicle veh-123 telemetry evidence.",
        session_id="vehicle-session-1",
        user_id="enterprise-demo-user",
        metadata={
            "classification": "internal",
            "tenant": "deldai",
        },
    )

    first_response = await runtime.run(
        "enterprise-rag-analyst",
        request,
        run_id="vehicle-run-1",
    )

    assert first_response.agent_name == "enterprise-rag-analyst"
    assert first_response.session_id == "vehicle-session-1"

    assert agent.last_tool_results is not None
    assert len(agent.last_tool_results) == 1

    first_result = agent.last_tool_results[0]

    assert first_result.success is True

    authorization_events = [
        event
        for event in observer.events
        if event.event_type is AgentExecutionEventType.TOOL_AUTHORIZATION_DECISION
    ]

    assert len(authorization_events) == 1

    authorization = authorization_events[0]

    assert authorization.agent_name == "enterprise-rag-analyst"
    assert authorization.session_id == "vehicle-session-1"
    assert authorization.tool_name == "vehicle.data.query"
    assert authorization.call_id == "vehicle-call-1"
    assert authorization.metadata == {
        "allowed": True,
        "reason": "Tool is authorized.",
    }

    # Authorization audit records must not expose the principal.
    assert "principal" not in authorization.metadata
    assert "enterprise-demo-user" not in str(authorization.metadata)

    assert first_result.tool_name == "vehicle.data.query"
    assert first_result.output == {
        "source": "silver.vehicle_events",
        "filters": {
            "vehicle_id": "veh-123",
            "start_time": "2026-01-01T00:00:00+00:00",
            "end_time": "2026-01-02T00:00:00+00:00",
            "limit": 10,
        },
        "count": 1,
        "records": [
            {
                "vehicle_id": "veh-123",
                "event_time": "2026-01-01T12:00:00+00:00",
                "speed": 52.5,
            }
        ],
    }

    reader.read.assert_called_once_with(spark)
    dataframe.select.assert_called_once_with(
        "vehicle_id",
        "event_time",
        "speed",
    )

    second_response = await runtime.run(
        "enterprise-rag-analyst",
        request,
        run_id="vehicle-run-1",
    )

    assert second_response.agent_name == "enterprise-rag-analyst"
    assert agent.last_tool_results is not None
    assert len(agent.last_tool_results) == 1

    second_result = agent.last_tool_results[0]

    assert second_result.success is True
    assert second_result.tool_name == "vehicle.data.query"
    assert second_result.output == first_result.output

    # Same run_id + call_id + tool_name must replay the completed result.
    # The underlying enterprise data service must not execute twice.
    reader.read.assert_called_once_with(spark)
    dataframe.collect.assert_called_once()


@pytest.mark.asyncio
async def test_agent_runtime_marks_vehicle_tool_outcome_ambiguous_when_ownership_is_lost():
    ownership_lost = asyncio.Event()

    vehicle_service, reader, spark, dataframe = _build_vehicle_service()

    vehicle_tool = OwnershipLosingVehicleDataQueryTool(
        vehicle_service,
        ownership_lost,
    )

    tool_registry = InMemoryToolRegistry()
    await tool_registry.register(vehicle_tool)

    authorizer = InMemoryToolAuthorizer()
    await authorizer.allow(
        "enterprise-demo-user",
        "vehicle.data.query",
    )

    authorization_service = ToolAuthorizationService(authorizer)
    idempotency_store = InMemoryToolExecutionIdempotencyStore()

    execution_service = ToolExecutionService(
        tool_registry,
        authorization_service=authorization_service,
        idempotency_store=idempotency_store,
    )

    agent_definition = AgentDefinition(
        name="enterprise-rag-analyst",
        description="Enterprise agent for governed vehicle telemetry evidence.",
        system_prompt=(
            "Use vehicle.data.query when structured vehicle telemetry " "evidence is required."
        ),
        model="mock-gpt",
        tool_names=("vehicle.data.query",),
    )

    agent = VehicleCallingAgent(agent_definition)

    agent_registry = InMemoryAgentRegistry()
    await agent_registry.register(agent)

    runtime = AgentRuntime(
        agent_registry,
        tool_execution_service=execution_service,
    )

    request = AgentRequest(
        input="Show vehicle veh-123 telemetry evidence.",
        session_id="vehicle-session-ownership-loss",
        user_id="enterprise-demo-user",
        metadata={
            "classification": "internal",
            "tenant": "deldai",
        },
    )

    with pytest.raises(AgentExecutionOwnershipLostError):
        await runtime.run(
            "enterprise-rag-analyst",
            request,
            run_id="vehicle-run-ownership-loss",
            execution_ownership_lost=ownership_lost,
        )

    assert ownership_lost.is_set()
    assert vehicle_tool.execution_count == 1

    # The enterprise data operation completed once, but durable ownership
    # was lost before the result could safely be committed as completed.
    reader.read.assert_called_once_with(spark)
    dataframe.collect.assert_called_once()

    key = ToolExecutionIdempotencyKey(
        run_id="vehicle-run-ownership-loss",
        call_id="vehicle-call-1",
        tool_name="vehicle.data.query",
    )

    claim = await idempotency_store.claim(key)

    assert claim.status is ToolIdempotencyClaimStatus.AMBIGUOUS

    # A retry of the same logical tool call must not access enterprise
    # vehicle data again.
    retry_result = await execution_service.execute(
        "vehicle.data.query",
        {
            "vehicle_id": "veh-123",
            "start_time": "2026-01-01T00:00:00+00:00",
            "end_time": "2026-01-02T00:00:00+00:00",
            "limit": 10,
        },
        principal="enterprise-demo-user",
        execution_context=ToolExecutionContext(
            run_id="vehicle-run-ownership-loss",
            call_id="vehicle-call-1",
            agent_name="enterprise-rag-analyst",
            session_id="vehicle-session-ownership-loss",
            user_id="enterprise-demo-user",
            execution_ownership_lost=asyncio.Event(),
        ),
    )

    assert retry_result.success is False
    assert retry_result.failure_category is ToolExecutionFailureCategory.EXECUTION_AMBIGUOUS
    assert retry_result.error == (
        "Tool execution has an ambiguous external outcome and must not be retried automatically."
    )

    assert vehicle_tool.execution_count == 1
    reader.read.assert_called_once_with(spark)
    dataframe.collect.assert_called_once()


@pytest.mark.asyncio
async def test_agent_runtime_denies_vehicle_tool_when_principal_is_not_authorized():
    vehicle_service, _, _, _ = _build_vehicle_service()
    vehicle_tool = VehicleDataQueryTool(vehicle_service)

    tool_registry = InMemoryToolRegistry()
    await tool_registry.register(vehicle_tool)

    authorizer = InMemoryToolAuthorizer()
    authorization_service = ToolAuthorizationService(authorizer)
    observer = FakeAgentExecutionObserver()

    execution_service = ToolExecutionService(
        tool_registry,
        authorization_service=authorization_service,
        audit_sink=ToolAuthorizationAuditObserver(observer),
    )

    agent_definition = AgentDefinition(
        name="enterprise-rag-analyst",
        description="Enterprise agent for governed vehicle telemetry evidence.",
        system_prompt="Use vehicle.data.query for vehicle telemetry.",
        model="mock-gpt",
        tool_names=("vehicle.data.query",),
    )

    agent = VehicleCallingAgent(agent_definition)

    agent_registry = InMemoryAgentRegistry()
    await agent_registry.register(agent)

    runtime = AgentRuntime(
        agent_registry,
        tool_execution_service=execution_service,
    )

    response = await runtime.run(
        "enterprise-rag-analyst",
        AgentRequest(
            input="Show vehicle telemetry.",
            session_id="vehicle-denied-session",
            user_id="unauthorized-user",
        ),
        run_id="vehicle-denied-run",
    )

    assert response.agent_name == "enterprise-rag-analyst"
    assert agent.last_tool_results is not None
    assert len(agent.last_tool_results) == 1

    result = agent.last_tool_results[0]

    assert result.success is False
    assert result.tool_name == "vehicle.data.query"
    assert result.error == "Tool is not authorized for this principal."

    authorization_events = [
        event
        for event in observer.events
        if event.event_type is AgentExecutionEventType.TOOL_AUTHORIZATION_DECISION
    ]

    assert len(authorization_events) == 1

    authorization = authorization_events[0]

    assert authorization.agent_name == "enterprise-rag-analyst"
    assert authorization.session_id == "vehicle-denied-session"
    assert authorization.tool_name == "vehicle.data.query"
    assert authorization.call_id == "vehicle-call-1"
    assert authorization.metadata == {
        "allowed": False,
        "reason": "Tool is not authorized for this principal.",
    }

    # Authorization audit records must not expose the principal.
    assert "principal" not in authorization.metadata
    assert "unauthorized-user" not in str(authorization.metadata)
