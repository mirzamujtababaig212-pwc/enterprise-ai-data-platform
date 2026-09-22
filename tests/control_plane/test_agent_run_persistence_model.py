from datetime import UTC, datetime

from sqlalchemy import inspect

from app.control_plane.persistence.models import AgentRunRecord, Base


def test_agent_run_record_table_name() -> None:
    assert AgentRunRecord.__tablename__ == "agent_runs"


def test_agent_run_record_columns() -> None:
    columns = {column.name: column for column in inspect(AgentRunRecord).columns}

    assert set(columns) == {
        "run_id",
        "agent_name",
        "session_id",
        "user_id",
        "idempotency_key",
        "status",
        "started_at",
        "completed_at",
        "error_type",
        "error_message",
        "output",
        "metadata",
        "request_snapshot",
        "lease_id",
        "lease_expires_at",
        "cancellation_requested",
        "cancellation_requested_at",
        "recovery_attempts",
    }


def test_agent_run_record_primary_key() -> None:
    primary_key_columns = {column.name for column in inspect(AgentRunRecord).primary_key}

    assert primary_key_columns == {"run_id"}


def test_agent_run_record_indexes() -> None:
    indexes = {
        index.name: {column.name for column in index.columns}
        for index in AgentRunRecord.__table__.indexes
    }

    assert indexes["ix_agent_runs_agent_name"] == {"agent_name"}
    assert indexes["ix_agent_runs_session_id"] == {"session_id"}
    assert indexes["ix_agent_runs_user_id"] == {"user_id"}
    assert indexes["ix_agent_runs_status"] == {"status"}
    assert indexes["ix_agent_runs_started_at"] == {"started_at"}


def test_agent_run_record_json_fields() -> None:
    columns = {column.name: column for column in inspect(AgentRunRecord).columns}

    assert columns["output"].type.__class__.__name__ == "JSON"
    assert columns["metadata"].type.__class__.__name__ == "JSON"


def test_agent_run_record_accepts_domain_shape() -> None:
    started_at = datetime.now(UTC)
    completed_at = datetime.now(UTC)

    record = AgentRunRecord(
        run_id="550e8400-e29b-41d4-a716-446655440000",
        agent_name="enterprise-analyst",
        session_id="session-1",
        user_id="user-1",
        status="completed",
        started_at=started_at,
        completed_at=completed_at,
        output={
            "answer": "completed",
            "sources": ["doc-1"],
        },
        run_metadata={
            "source": "control_plane",
        },
    )

    assert record.run_id == "550e8400-e29b-41d4-a716-446655440000"
    assert record.agent_name == "enterprise-analyst"
    assert record.status == "completed"
    assert record.output == {
        "answer": "completed",
        "sources": ["doc-1"],
    }
    assert record.run_metadata == {"source": "control_plane"}


def test_agent_run_record_is_registered_in_base_metadata() -> None:
    assert "agent_runs" in Base.metadata.tables


def test_agent_run_checkpoint_record_table_name() -> None:
    from app.control_plane.persistence.models import AgentRunCheckpointRecord

    assert AgentRunCheckpointRecord.__tablename__ == "agent_run_checkpoints"


def test_agent_run_checkpoint_record_columns() -> None:
    from app.control_plane.persistence.models import AgentRunCheckpointRecord

    columns = {column.name: column for column in inspect(AgentRunCheckpointRecord).columns}

    assert set(columns) == {
        "id",
        "run_id",
        "agent_name",
        "session_id",
        "user_id",
        "schema_version",
        "position",
        "tool_round",
        "checkpoint_payload",
        "created_at",
    }


def test_agent_run_checkpoint_record_primary_key() -> None:
    from app.control_plane.persistence.models import AgentRunCheckpointRecord

    primary_key_columns = {column.name for column in inspect(AgentRunCheckpointRecord).primary_key}

    assert primary_key_columns == {"id"}


def test_agent_run_checkpoint_record_indexes() -> None:
    from app.control_plane.persistence.models import AgentRunCheckpointRecord

    indexes = {
        index.name: {column.name for column in index.columns}
        for index in AgentRunCheckpointRecord.__table__.indexes
    }

    assert indexes["ix_agent_run_checkpoints_run_id"] == {"run_id"}
    assert indexes["ix_agent_run_checkpoints_agent_name"] == {"agent_name"}
    assert indexes["ix_agent_run_checkpoints_session_id"] == {"session_id"}
    assert indexes["ix_agent_run_checkpoints_user_id"] == {"user_id"}
    assert indexes["ix_agent_run_checkpoints_position"] == {"position"}
    assert indexes["ix_agent_run_checkpoints_created_at"] == {"created_at"}
    assert indexes["ix_agent_run_checkpoints_run_created_at"] == {
        "run_id",
        "created_at",
    }


def test_agent_run_checkpoint_record_json_field() -> None:
    from app.control_plane.persistence.models import AgentRunCheckpointRecord

    columns = {column.name: column for column in inspect(AgentRunCheckpointRecord).columns}

    assert columns["checkpoint_payload"].type.__class__.__name__ == "JSON"


def test_agent_run_checkpoint_record_is_registered_in_base_metadata() -> None:
    assert "agent_run_checkpoints" in Base.metadata.tables
