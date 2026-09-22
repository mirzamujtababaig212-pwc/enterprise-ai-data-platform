import pytest

from tools.authorization.audit import ToolAuthorizationAuditRecord


class RecordingAuditSink:
    def __init__(self) -> None:
        self.records = []

    async def record(
        self,
        record: ToolAuthorizationAuditRecord,
    ) -> None:
        self.records.append(record)


def test_authorization_audit_record_is_constructed() -> None:
    record = ToolAuthorizationAuditRecord(
        principal="user-123",
        tool_name="rag.search",
        allowed=True,
        reason="Tool is authorized.",
        policy_id="metadata_policy",
        policy_version="1.0",
        run_id="run-123",
        call_id="call-456",
        agent_name="enterprise-rag-analyst",
        session_id="session-789",
    )

    assert record.principal == "user-123"
    assert record.tool_name == "rag.search"
    assert record.allowed is True
    assert record.reason == "Tool is authorized."
    assert record.policy_id == "metadata_policy"
    assert record.policy_version == "1.0"
    assert record.run_id == "run-123"
    assert record.call_id == "call-456"
    assert record.agent_name == "enterprise-rag-analyst"
    assert record.session_id == "session-789"


@pytest.mark.asyncio
async def test_authorization_audit_sink_records_decision() -> None:
    sink = RecordingAuditSink()

    record = ToolAuthorizationAuditRecord(
        principal="user-123",
        tool_name="rag.search",
        allowed=False,
        reason="Tool is not authorized for this principal.",
        run_id="run-123",
        call_id="call-456",
        agent_name="enterprise-rag-analyst",
        session_id="session-789",
    )

    await sink.record(record)

    assert sink.records == [record]


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("principal", "   ", "Principal must not be empty."),
        ("tool_name", "   ", "Tool name must not be empty."),
        ("reason", "   ", "reason must not be empty when provided."),
        ("policy_id", "   ", "policy_id must not be empty when provided."),
        ("policy_version", "   ", "policy_version must not be empty when provided."),
        ("run_id", "   ", "run_id must not be empty when provided."),
        ("call_id", "   ", "call_id must not be empty when provided."),
        ("agent_name", "   ", "agent_name must not be empty when provided."),
        ("session_id", "   ", "session_id must not be empty when provided."),
    ],
)
def test_authorization_audit_record_rejects_empty_fields(
    field: str,
    value: str,
    message: str,
) -> None:
    kwargs = {
        "principal": "user-123",
        "tool_name": "rag.search",
        "allowed": True,
        field: value,
    }

    with pytest.raises(ValueError, match=message):
        ToolAuthorizationAuditRecord(**kwargs)
