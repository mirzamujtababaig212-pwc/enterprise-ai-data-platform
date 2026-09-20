import asyncio

import pytest

from tools.authorization.in_memory import (
    InMemoryToolAuthorizer,
)
from tools.authorization.service import (
    ToolAuthorizationService,
)
from tools.execution.service import ToolExecutionService
from tools.models import (
    ToolDefinition,
    ToolExecutionFailureCategory,
)
from tools.registry.in_memory import InMemoryToolRegistry
from tools.authorization.models import ToolAuthorizationResult
from tools.execution.context import ToolExecutionContext


class RecordingToolAuthorizer:
    def __init__(self) -> None:
        self.requests = []

    async def authorize(self, request):
        self.requests.append(request)

        return ToolAuthorizationResult(
            principal=request.principal,
            tool_name=request.tool_name,
            allowed=True,
        )


class FakeTool:
    def __init__(
        self,
        name: str = "test_tool",
        enabled: bool = True,
    ):
        self._definition = ToolDefinition(
            name=name,
            description="A test tool.",
            enabled=enabled,
        )

        self.execution_count = 0

    @property
    def definition(self) -> ToolDefinition:
        return self._definition

    async def execute(self, arguments):
        self.execution_count += 1

        return {
            "status": "success",
            "arguments": arguments,
        }


class ContextAwareTool:
    def __init__(self, name: str = "context_tool"):
        self._definition = ToolDefinition(
            name=name,
            description="A context-aware test tool.",
        )
        self.received_arguments = None
        self.received_context = None

    @property
    def definition(self) -> ToolDefinition:
        return self._definition

    async def execute(self, arguments):
        raise AssertionError("execute() should not be called for a context-aware tool.")

    async def execute_with_context(self, arguments, context):
        self.received_arguments = arguments
        self.received_context = context

        return {
            "status": "context_success",
            "arguments": arguments,
            "context": context,
        }


class FailingTool:
    def __init__(self):
        self._definition = ToolDefinition(
            name="failing_tool",
            description="A failing test tool.",
        )

    @property
    def definition(self) -> ToolDefinition:
        return self._definition

    async def execute(self, arguments):
        raise RuntimeError("simulated tool failure")


class SlowTool:
    def __init__(self):
        self._definition = ToolDefinition(
            name="slow_tool",
            description="A slow test tool.",
        )

    @property
    def definition(self) -> ToolDefinition:
        return self._definition

    async def execute(self, arguments):
        await asyncio.sleep(0.2)
        return {"status": "completed"}


@pytest.mark.asyncio
async def test_execute_calls_registered_tool():
    registry = InMemoryToolRegistry()
    tool = FakeTool()

    await registry.register(tool)

    service = ToolExecutionService(registry)

    result = await service.execute(
        "test_tool",
        {"value": 42},
    )

    assert result.success is True
    assert result.tool_name == "test_tool"
    assert result.output == {
        "status": "success",
        "arguments": {"value": 42},
    }
    assert result.error is None
    assert result.failure_category is None
    assert tool.execution_count == 1


@pytest.mark.asyncio
async def test_execute_unknown_tool_returns_failure():
    registry = InMemoryToolRegistry()
    service = ToolExecutionService(registry)

    result = await service.execute(
        "missing_tool",
        {},
    )

    assert result.success is False
    assert result.tool_name == "missing_tool"
    assert result.output is None
    assert result.error == "Tool not found: missing_tool"
    assert result.failure_category == ToolExecutionFailureCategory.TOOL_NOT_FOUND


@pytest.mark.asyncio
async def test_execute_disabled_tool_returns_failure():
    registry = InMemoryToolRegistry()

    await registry.register(
        FakeTool(
            name="disabled_tool",
            enabled=False,
        )
    )

    service = ToolExecutionService(registry)

    result = await service.execute(
        "disabled_tool",
        {},
    )

    assert result.success is False
    assert result.tool_name == "disabled_tool"
    assert result.error == "Tool is disabled: disabled_tool"
    assert result.failure_category == ToolExecutionFailureCategory.TOOL_DISABLED


@pytest.mark.asyncio
async def test_execute_empty_tool_name_is_rejected():
    registry = InMemoryToolRegistry()
    service = ToolExecutionService(registry)

    with pytest.raises(
        ValueError,
        match="Tool name must not be empty",
    ):
        await service.execute(
            "",
            {},
        )


@pytest.mark.asyncio
async def test_execute_tool_failure_is_isolated():
    registry = InMemoryToolRegistry()

    await registry.register(FailingTool())

    service = ToolExecutionService(registry)

    result = await service.execute(
        "failing_tool",
        {},
    )

    assert result.success is False
    assert result.tool_name == "failing_tool"
    assert result.output is None
    assert result.error == "RuntimeError: simulated tool failure"
    assert result.failure_category == ToolExecutionFailureCategory.EXECUTION_ERROR


@pytest.mark.asyncio
async def test_execute_timeout_is_handled():
    registry = InMemoryToolRegistry()

    await registry.register(SlowTool())

    service = ToolExecutionService(
        registry,
        default_timeout_seconds=0.05,
    )

    result = await service.execute(
        "slow_tool",
        {},
    )

    assert result.success is False
    assert result.tool_name == "slow_tool"
    assert result.output is None
    assert "timed out" in result.error
    assert result.failure_category == ToolExecutionFailureCategory.TIMEOUT


@pytest.mark.asyncio
async def test_execute_custom_timeout_is_used():
    registry = InMemoryToolRegistry()

    await registry.register(SlowTool())

    service = ToolExecutionService(
        registry,
        default_timeout_seconds=1.0,
    )

    result = await service.execute(
        "slow_tool",
        {},
        timeout_seconds=0.05,
    )

    assert result.success is False
    assert "timed out" in result.error
    assert result.failure_category == ToolExecutionFailureCategory.TIMEOUT


def test_invalid_default_timeout_is_rejected():
    registry = InMemoryToolRegistry()

    with pytest.raises(
        ValueError,
        match="default_timeout_seconds must be greater than zero",
    ):
        ToolExecutionService(
            registry,
            default_timeout_seconds=0,
        )


@pytest.mark.asyncio
async def test_invalid_custom_timeout_is_rejected():
    registry = InMemoryToolRegistry()

    service = ToolExecutionService(registry)

    with pytest.raises(
        ValueError,
        match="timeout_seconds must be greater than zero",
    ):
        await service.execute(
            "test_tool",
            {},
            timeout_seconds=0,
        )


@pytest.mark.asyncio
async def test_authorized_principal_can_execute_tool():
    registry = InMemoryToolRegistry()
    authorizer = InMemoryToolAuthorizer()

    tool = FakeTool()

    await registry.register(tool)

    await authorizer.allow(
        "agent:research",
        "test_tool",
    )

    authorization_service = ToolAuthorizationService(authorizer)

    service = ToolExecutionService(
        registry,
        authorization_service=authorization_service,
    )

    result = await service.execute(
        "test_tool",
        {"value": 42},
        principal="agent:research",
    )

    assert result.success is True
    assert result.output == {
        "status": "success",
        "arguments": {"value": 42},
    }
    assert tool.execution_count == 1


@pytest.mark.asyncio
async def test_unauthorized_principal_cannot_execute_tool():
    registry = InMemoryToolRegistry()
    authorizer = InMemoryToolAuthorizer()

    tool = FakeTool()

    await registry.register(tool)

    authorization_service = ToolAuthorizationService(authorizer)

    service = ToolExecutionService(
        registry,
        authorization_service=authorization_service,
    )

    result = await service.execute(
        "test_tool",
        {},
        principal="agent:restricted",
    )

    assert result.success is False
    assert result.tool_name == "test_tool"
    assert result.error == "Tool is not authorized for this principal."
    assert result.failure_category == ToolExecutionFailureCategory.AUTHORIZATION
    assert tool.execution_count == 0


@pytest.mark.asyncio
async def test_authorization_requires_principal():
    registry = InMemoryToolRegistry()
    authorizer = InMemoryToolAuthorizer()

    await registry.register(FakeTool())

    authorization_service = ToolAuthorizationService(authorizer)

    service = ToolExecutionService(
        registry,
        authorization_service=authorization_service,
    )

    result = await service.execute(
        "test_tool",
        {},
    )

    assert result.success is False
    assert result.error == ("Principal is required when " "tool authorization is enabled.")
    assert result.failure_category == ToolExecutionFailureCategory.MISSING_PRINCIPAL


@pytest.mark.asyncio
async def test_authorization_happens_before_tool_execution():
    registry = InMemoryToolRegistry()
    authorizer = InMemoryToolAuthorizer()

    tool = FakeTool()

    await registry.register(tool)

    authorization_service = ToolAuthorizationService(authorizer)

    service = ToolExecutionService(
        registry,
        authorization_service=authorization_service,
    )

    result = await service.execute(
        "test_tool",
        {},
        principal="agent:restricted",
    )

    assert result.success is False
    assert tool.execution_count == 0


@pytest.mark.asyncio
async def test_authorized_tool_still_respects_timeout():
    registry = InMemoryToolRegistry()
    authorizer = InMemoryToolAuthorizer()

    tool = SlowTool()

    await registry.register(tool)

    await authorizer.allow(
        "agent:research",
        "slow_tool",
    )

    authorization_service = ToolAuthorizationService(authorizer)

    service = ToolExecutionService(
        registry,
        authorization_service=authorization_service,
        default_timeout_seconds=0.05,
    )

    result = await service.execute(
        "slow_tool",
        {},
        principal="agent:research",
    )

    assert result.success is False
    assert "timed out" in result.error


@pytest.mark.asyncio
async def test_execute_passes_execution_context_to_context_aware_tool():
    registry = InMemoryToolRegistry()
    tool = ContextAwareTool()

    await registry.register(tool)

    service = ToolExecutionService(registry)

    execution_context = ToolExecutionContext(
        agent_name="research-agent",
        session_id="session-123",
        user_id="user-456",
    )

    result = await service.execute(
        "context_tool",
        {"query": "RAG"},
        execution_context=execution_context,
    )

    assert result.success is True
    assert result.output == {
        "status": "context_success",
        "arguments": {"query": "RAG"},
        "context": execution_context,
    }

    assert tool.received_arguments == {"query": "RAG"}
    assert tool.received_context == execution_context


@pytest.mark.asyncio
async def test_execute_passes_none_context_when_none_is_provided():
    registry = InMemoryToolRegistry()
    tool = ContextAwareTool()

    await registry.register(tool)

    service = ToolExecutionService(registry)

    result = await service.execute(
        "context_tool",
        {},
    )

    assert result.success is True
    assert tool.received_context is None


@pytest.mark.asyncio
async def test_execute_preserves_legacy_tool_execution_without_context():
    registry = InMemoryToolRegistry()
    tool = FakeTool()

    await registry.register(tool)

    service = ToolExecutionService(registry)

    result = await service.execute(
        "test_tool",
        {"value": 42},
        execution_context={
            "agent_name": "legacy-compatible-agent",
        },
    )

    assert result.success is True
    assert result.output == {
        "status": "success",
        "arguments": {"value": 42},
    }
    assert tool.execution_count == 1


@pytest.mark.asyncio
async def test_authorization_happens_before_contextual_tool_execution():
    registry = InMemoryToolRegistry()
    authorizer = InMemoryToolAuthorizer()
    tool = ContextAwareTool()

    await registry.register(tool)

    authorization_service = ToolAuthorizationService(authorizer)

    service = ToolExecutionService(
        registry,
        authorization_service=authorization_service,
    )

    result = await service.execute(
        "context_tool",
        {},
        principal="agent:restricted",
        execution_context=ToolExecutionContext(),
    )

    assert result.success is False
    assert result.error == "Tool is not authorized for this principal."
    assert result.failure_category == ToolExecutionFailureCategory.AUTHORIZATION
    assert tool.received_arguments is None
    assert tool.received_context is None


@pytest.mark.asyncio
async def test_execution_authorization_receives_tool_metadata():
    registry = InMemoryToolRegistry()
    authorizer = RecordingToolAuthorizer()

    tool = FakeTool()
    await registry.register(tool)

    authorization_service = ToolAuthorizationService(authorizer)

    service = ToolExecutionService(
        registry,
        authorization_service=authorization_service,
    )

    result = await service.execute(
        "test_tool",
        {"value": 42},
        principal="agent:research",
    )

    assert result.success is True
    assert len(authorizer.requests) == 1
    assert authorizer.requests[0].principal == "agent:research"
    assert authorizer.requests[0].tool_name == "test_tool"
    assert authorizer.requests[0].metadata == tool.definition.metadata


class RecordingAuthorizationAuditSink:
    def __init__(self) -> None:
        self.records = []

    async def record(self, record) -> None:
        self.records.append(record)


class FailingAuthorizationAuditSink:
    async def record(self, record) -> None:
        raise RuntimeError("simulated audit failure")


@pytest.mark.asyncio
async def test_authorization_decision_is_audited_for_allowed_tool() -> None:
    registry = InMemoryToolRegistry()
    authorizer = InMemoryToolAuthorizer()
    audit_sink = RecordingAuthorizationAuditSink()
    tool = FakeTool()

    await registry.register(tool)
    await authorizer.allow("agent:research", "test_tool")

    authorization_service = ToolAuthorizationService(authorizer)

    service = ToolExecutionService(
        registry,
        authorization_service=authorization_service,
        audit_sink=audit_sink,
    )

    result = await service.execute(
        "test_tool",
        {"value": 42},
        principal="agent:research",
        execution_context={
            "run_id": "run-123",
            "call_id": "call-456",
            "agent_name": "research-agent",
            "session_id": "session-789",
        },
    )

    assert result.success is True
    assert tool.execution_count == 1
    assert len(audit_sink.records) == 1

    record = audit_sink.records[0]
    assert record.principal == "agent:research"
    assert record.tool_name == "test_tool"
    assert record.allowed is True
    assert record.reason == "Tool is authorized."
    assert record.run_id == "run-123"
    assert record.call_id == "call-456"
    assert record.agent_name == "research-agent"
    assert record.session_id == "session-789"


@pytest.mark.asyncio
async def test_authorization_decision_is_audited_for_denied_tool() -> None:
    registry = InMemoryToolRegistry()
    authorizer = InMemoryToolAuthorizer()
    audit_sink = RecordingAuthorizationAuditSink()
    tool = FakeTool()

    await registry.register(tool)

    authorization_service = ToolAuthorizationService(authorizer)

    service = ToolExecutionService(
        registry,
        authorization_service=authorization_service,
        audit_sink=audit_sink,
    )

    result = await service.execute(
        "test_tool",
        {},
        principal="agent:restricted",
        execution_context={
            "run_id": "run-denied",
            "call_id": "call-denied",
            "agent_name": "restricted-agent",
            "session_id": "session-denied",
        },
    )

    assert result.success is False
    assert result.error == "Tool is not authorized for this principal."
    assert result.failure_category == ToolExecutionFailureCategory.AUTHORIZATION
    assert tool.execution_count == 0
    assert len(audit_sink.records) == 1

    record = audit_sink.records[0]
    assert record.principal == "agent:restricted"
    assert record.tool_name == "test_tool"
    assert record.allowed is False
    assert record.reason == "Tool is not authorized for this principal."
    assert record.run_id == "run-denied"
    assert record.call_id == "call-denied"
    assert record.agent_name == "restricted-agent"
    assert record.session_id == "session-denied"


@pytest.mark.asyncio
async def test_authorization_audit_failure_does_not_break_tool_execution() -> None:
    registry = InMemoryToolRegistry()
    authorizer = InMemoryToolAuthorizer()
    tool = FakeTool()

    await registry.register(tool)
    await authorizer.allow("agent:research", "test_tool")

    authorization_service = ToolAuthorizationService(authorizer)

    service = ToolExecutionService(
        registry,
        authorization_service=authorization_service,
        audit_sink=FailingAuthorizationAuditSink(),
    )

    result = await service.execute(
        "test_tool",
        {"value": 42},
        principal="agent:research",
    )

    assert result.success is True
    assert result.output == {
        "status": "success",
        "arguments": {"value": 42},
    }
    assert tool.execution_count == 1


class SchemaValidatedTool:
    def __init__(self, schema):
        self._definition = ToolDefinition(
            name="schema_tool",
            description="A schema-validated test tool.",
            input_schema=schema,
        )
        self.execution_count = 0

    @property
    def definition(self) -> ToolDefinition:
        return self._definition

    async def execute(self, arguments):
        self.execution_count += 1
        return arguments


@pytest.mark.asyncio
async def test_execute_validates_tool_arguments_against_input_schema():
    registry = InMemoryToolRegistry()

    tool = SchemaValidatedTool(
        {
            "type": "object",
            "properties": {
                "query": {"type": "string"},
                "top_k": {"type": "integer", "minimum": 1},
            },
            "required": ["query"],
            "additionalProperties": False,
        }
    )

    await registry.register(tool)

    service = ToolExecutionService(registry)

    result = await service.execute(
        "schema_tool",
        {
            "query": "enterprise AI",
            "top_k": 5,
        },
    )

    assert result.success is True
    assert result.output == {
        "query": "enterprise AI",
        "top_k": 5,
    }
    assert tool.execution_count == 1


@pytest.mark.asyncio
async def test_execute_rejects_invalid_tool_arguments_before_execution():
    registry = InMemoryToolRegistry()

    tool = SchemaValidatedTool(
        {
            "type": "object",
            "properties": {
                "query": {"type": "string"},
            },
            "required": ["query"],
        }
    )

    await registry.register(tool)

    service = ToolExecutionService(registry)

    result = await service.execute(
        "schema_tool",
        {},
    )

    assert result.success is False
    assert result.tool_name == "schema_tool"
    assert "schema validation" in result.error
    assert result.failure_category == ToolExecutionFailureCategory.SCHEMA_VALIDATION
    assert "query" in result.error
    assert tool.execution_count == 0


@pytest.mark.asyncio
async def test_execute_rejects_invalid_argument_type_before_execution():
    registry = InMemoryToolRegistry()

    tool = SchemaValidatedTool(
        {
            "type": "object",
            "properties": {
                "top_k": {"type": "integer"},
            },
        }
    )

    await registry.register(tool)

    service = ToolExecutionService(registry)

    result = await service.execute(
        "schema_tool",
        {"top_k": "5"},
    )

    assert result.success is False
    assert result.tool_name == "schema_tool"
    assert "schema validation" in result.error
    assert result.failure_category == ToolExecutionFailureCategory.SCHEMA_VALIDATION
    assert tool.execution_count == 0


@pytest.mark.asyncio
async def test_execute_allows_empty_schema():
    registry = InMemoryToolRegistry()

    tool = SchemaValidatedTool({})

    await registry.register(tool)

    service = ToolExecutionService(registry)

    result = await service.execute(
        "schema_tool",
        {"anything": "goes"},
    )

    assert result.success is True
    assert result.output == {"anything": "goes"}
    assert tool.execution_count == 1


@pytest.mark.asyncio
async def test_execute_rejects_invalid_tool_schema():
    registry = InMemoryToolRegistry()

    tool = SchemaValidatedTool(
        {
            "type": "not-a-real-json-schema-type",
        }
    )

    await registry.register(tool)

    service = ToolExecutionService(registry)

    result = await service.execute(
        "schema_tool",
        {},
    )

    assert result.success is False
    assert result.tool_name == "schema_tool"
    assert "Tool input schema is invalid" in result.error
    assert result.failure_category == ToolExecutionFailureCategory.INVALID_SCHEMA
    assert tool.execution_count == 0


@pytest.mark.asyncio
async def test_argument_validation_happens_before_authorization():
    registry = InMemoryToolRegistry()
    authorizer = InMemoryToolAuthorizer()

    tool = SchemaValidatedTool(
        {
            "type": "object",
            "required": ["query"],
        }
    )

    await registry.register(tool)

    authorization_service = ToolAuthorizationService(authorizer)

    service = ToolExecutionService(
        registry,
        authorization_service=authorization_service,
    )

    result = await service.execute(
        "schema_tool",
        {},
        principal="agent:research",
    )

    assert result.success is False
    assert "schema validation" in result.error
    assert result.failure_category == ToolExecutionFailureCategory.SCHEMA_VALIDATION
    assert tool.execution_count == 0
