from __future__ import annotations

from datetime import datetime

from sqlalchemy import (
    Boolean,
    CheckConstraint,
    DateTime,
    Float,
    ForeignKey,
    ForeignKeyConstraint,
    Index,
    Integer,
    JSON,
    PrimaryKeyConstraint,
    String,
    Text,
    UniqueConstraint,
    func,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship
from pgvector.sqlalchemy import Vector


class Base(DeclarativeBase):
    pass


class MCPServerRecord(Base):
    __tablename__ = "mcp_servers"

    server_id: Mapped[str] = mapped_column(
        String(255),
        primary_key=True,
    )

    tenant_id: Mapped[str] = mapped_column(
        String(255),
        nullable=False,
        index=True,
    )

    name: Mapped[str] = mapped_column(
        String(255),
        nullable=False,
        unique=True,
    )

    transport: Mapped[str] = mapped_column(
        String(50),
        nullable=False,
    )

    desired_state: Mapped[str] = mapped_column(
        String(50),
        nullable=False,
        index=True,
    )

    configuration: Mapped[dict] = mapped_column(
        JSON().with_variant(JSONB, "postgresql"),
        nullable=False,
    )

    secret_references: Mapped[dict] = mapped_column(
        JSON().with_variant(JSONB, "postgresql"),
        nullable=False,
        default=dict,
    )

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
    )

    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
        onupdate=func.now(),
    )


class RetrievalEvaluationRunRecord(Base):
    __tablename__ = "retrieval_evaluation_runs"

    run_id: Mapped[str] = mapped_column(String(255), primary_key=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        index=True,
    )
    dataset_name: Mapped[str] = mapped_column(
        String(255),
        nullable=False,
        index=True,
    )
    dataset_version: Mapped[str] = mapped_column(
        String(255),
        nullable=False,
        index=True,
    )
    release_passed: Mapped[bool] = mapped_column(
        nullable=False,
        index=True,
    )
    lineage: Mapped[dict] = mapped_column(
        JSON().with_variant(JSONB, "postgresql"),
        nullable=False,
    )
    evaluation: Mapped[dict] = mapped_column(
        JSON().with_variant(JSONB, "postgresql"),
        nullable=False,
    )
    quality_gate: Mapped[dict] = mapped_column(
        JSON().with_variant(JSONB, "postgresql"),
        nullable=False,
    )
    regression: Mapped[dict | None] = mapped_column(
        JSON().with_variant(JSONB, "postgresql"),
        nullable=True,
    )
    external_evaluations: Mapped[list[dict]] = mapped_column(
        JSON().with_variant(JSONB, "postgresql"),
        nullable=False,
        default=list,
    )
    external_quality_gate: Mapped[dict | None] = mapped_column(
        JSON().with_variant(JSONB, "postgresql"),
        nullable=True,
    )


class RetrievalEvaluationReleaseDecisionRecord(Base):
    __tablename__ = "retrieval_evaluation_release_decisions"

    run_id: Mapped[str] = mapped_column(
        String(255),
        ForeignKey("retrieval_evaluation_runs.run_id", ondelete="CASCADE"),
        primary_key=True,
    )
    passed: Mapped[bool] = mapped_column(
        nullable=False,
        index=True,
    )
    errors: Mapped[list] = mapped_column(
        JSON().with_variant(JSONB, "postgresql"),
        nullable=False,
        default=list,
    )
    native: Mapped[dict] = mapped_column(
        JSON().with_variant(JSONB, "postgresql"),
        nullable=False,
    )
    external: Mapped[dict] = mapped_column(
        JSON().with_variant(JSONB, "postgresql"),
        nullable=False,
    )


class AgentEvaluationRunRecord(Base):
    __tablename__ = "agent_evaluation_runs"

    evaluation_run_id: Mapped[str] = mapped_column(
        String(255),
        primary_key=True,
    )

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        index=True,
    )

    evaluated_run_id: Mapped[str] = mapped_column(
        String(36),
        ForeignKey("agent_runs.run_id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )

    agent_name: Mapped[str] = mapped_column(
        String(255),
        nullable=False,
        index=True,
    )

    agent_version: Mapped[str | None] = mapped_column(
        String(255),
        nullable=True,
    )

    tenant_id: Mapped[str | None] = mapped_column(
        String(255),
        nullable=True,
        index=True,
    )

    passed: Mapped[bool] = mapped_column(
        nullable=False,
        index=True,
    )

    lineage: Mapped[dict] = mapped_column(
        JSON().with_variant(JSONB, "postgresql"),
        nullable=False,
    )

    metrics: Mapped[dict] = mapped_column(
        JSON().with_variant(JSONB, "postgresql"),
        nullable=False,
    )

    policy: Mapped[dict] = mapped_column(
        JSON().with_variant(JSONB, "postgresql"),
        nullable=False,
    )

    quality_gate: Mapped[dict] = mapped_column(
        JSON().with_variant(JSONB, "postgresql"),
        nullable=False,
    )

    answer_evaluation: Mapped[dict | None] = mapped_column(
        JSON().with_variant(JSONB, "postgresql"),
        nullable=True,
    )

    context_quality: Mapped[dict | None] = mapped_column(
        JSON().with_variant(JSONB, "postgresql"),
        nullable=True,
    )


class AgentRunRecord(Base):
    __tablename__ = "agent_runs"
    __table_args__ = (
        UniqueConstraint(
            "user_id",
            "idempotency_key",
            name="uq_agent_runs_user_id_idempotency_key",
        ),
        ForeignKeyConstraint(
            ["root_run_id"],
            ["agent_runs.run_id"],
            name="fk_agent_runs_root_run_id",
            ondelete="RESTRICT",
        ),
        ForeignKeyConstraint(
            ["parent_run_id"],
            ["agent_runs.run_id"],
            name="fk_agent_runs_parent_run_id",
            ondelete="RESTRICT",
        ),
        ForeignKeyConstraint(
            ["parent_run_id", "parent_step_id"],
            ["agent_run_steps.run_id", "agent_run_steps.step_id"],
            name="fk_agent_runs_parent_step",
            ondelete="RESTRICT",
        ),
    )

    run_id: Mapped[str] = mapped_column(
        String(36),
        primary_key=True,
    )

    root_run_id: Mapped[str] = mapped_column(
        String(36),
        nullable=False,
        index=True,
    )

    parent_run_id: Mapped[str | None] = mapped_column(
        String(36),
        nullable=True,
        index=True,
    )

    parent_step_id: Mapped[str | None] = mapped_column(
        String(255),
        nullable=True,
    )

    causation_id: Mapped[str | None] = mapped_column(
        String(255),
        nullable=True,
        index=True,
    )

    agent_name: Mapped[str] = mapped_column(
        String(255),
        nullable=False,
        index=True,
    )

    session_id: Mapped[str | None] = mapped_column(
        String(255),
        nullable=True,
        index=True,
    )

    user_id: Mapped[str | None] = mapped_column(
        String(255),
        nullable=True,
        index=True,
    )

    principal: Mapped[str | None] = mapped_column(String(255), nullable=True, index=True)

    tenant_id: Mapped[str | None] = mapped_column(
        String(255),
        nullable=True,
        index=True,
    )

    idempotency_key: Mapped[str | None] = mapped_column(
        String(255),
        nullable=True,
        index=True,
    )

    lease_id: Mapped[str | None] = mapped_column(
        String(36),
        nullable=True,
        index=True,
    )
    recovery_attempts: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        default=0,
        server_default="0",
    )

    lease_expires_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
        index=True,
    )

    cancellation_requested: Mapped[bool] = mapped_column(
        Boolean,
        nullable=False,
        server_default="false",
    )

    cancellation_requested_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
    )

    status: Mapped[str] = mapped_column(
        String(50),
        nullable=False,
        index=True,
    )

    started_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
        index=True,
    )

    completed_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
    )

    error_type: Mapped[str | None] = mapped_column(
        String(255),
        nullable=True,
    )

    error_message: Mapped[str | None] = mapped_column(
        String,
        nullable=True,
    )

    output: Mapped[object | None] = mapped_column(
        JSON().with_variant(JSONB, "postgresql"),
        nullable=True,
    )

    run_metadata: Mapped[dict] = mapped_column(
        "metadata",
        JSON().with_variant(JSONB, "postgresql"),
        nullable=False,
        default=dict,
    )

    request_snapshot: Mapped[dict | None] = mapped_column(
        JSON().with_variant(JSONB, "postgresql"),
        nullable=True,
    )


class AgentRunStepRecord(Base):
    __tablename__ = "agent_run_steps"

    run_id: Mapped[str] = mapped_column(
        String(36),
        ForeignKey("agent_runs.run_id", ondelete="CASCADE"),
        nullable=False,
    )

    step_id: Mapped[str] = mapped_column(
        String(255),
        nullable=False,
    )

    step_index: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
    )

    step_type: Mapped[str] = mapped_column(
        String(100),
        nullable=False,
    )

    status: Mapped[str] = mapped_column(
        String(50),
        nullable=False,
        index=True,
    )

    attempt: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        default=1,
    )

    tool_name: Mapped[str | None] = mapped_column(
        String(255),
        nullable=True,
    )

    call_id: Mapped[str | None] = mapped_column(
        String(255),
        nullable=True,
    )

    input: Mapped[object | None] = mapped_column(
        JSON().with_variant(JSONB, "postgresql"),
        nullable=True,
    )

    output: Mapped[object | None] = mapped_column(
        JSON().with_variant(JSONB, "postgresql"),
        nullable=True,
    )

    error: Mapped[str | None] = mapped_column(
        String,
        nullable=True,
    )

    failure_category: Mapped[str | None] = mapped_column(
        String(100),
        nullable=True,
    )

    started_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
    )

    completed_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
    )

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
    )

    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
        onupdate=func.now(),
    )

    step_metadata: Mapped[dict] = mapped_column(
        "metadata",
        JSON().with_variant(JSONB, "postgresql"),
        nullable=False,
        default=dict,
    )

    __table_args__ = (
        PrimaryKeyConstraint(
            "run_id",
            "step_id",
            name="pk_agent_run_steps",
        ),
        Index(
            "ix_agent_run_steps_run_step_index",
            "run_id",
            "step_index",
        ),
        Index(
            "ix_agent_run_steps_run_status",
            "run_id",
            "status",
        ),
    )


class ApprovalOverrideRecord(Base):
    __tablename__ = "approval_overrides"

    override_id: Mapped[str] = mapped_column(
        String(255),
        primary_key=True,
    )

    approval_id: Mapped[str] = mapped_column(
        String(255),
        ForeignKey("approval_requests.approval_id", ondelete="CASCADE"),
        nullable=False,
        unique=True,
        index=True,
    )

    run_id: Mapped[str] = mapped_column(
        String(36),
        ForeignKey("agent_runs.run_id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )

    actor: Mapped[str] = mapped_column(
        String(255),
        nullable=False,
    )

    reason: Mapped[str] = mapped_column(
        String,
        nullable=False,
    )

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
    )


class ApprovalRequestRecord(Base):
    __tablename__ = "approval_requests"

    approval_id: Mapped[str] = mapped_column(
        String(255),
        primary_key=True,
    )

    run_id: Mapped[str] = mapped_column(
        String(36),
        ForeignKey("agent_runs.run_id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )

    step_id: Mapped[str] = mapped_column(
        String(255),
        nullable=False,
    )

    call_id: Mapped[str] = mapped_column(
        String(255),
        nullable=False,
    )

    tool_name: Mapped[str] = mapped_column(
        String(255),
        nullable=False,
    )

    idempotency_key: Mapped[str] = mapped_column(
        String(255),
        nullable=False,
    )

    status: Mapped[str] = mapped_column(
        String(50),
        nullable=False,
        index=True,
    )

    policy_name: Mapped[str] = mapped_column(
        String(255),
        nullable=False,
    )

    policy_version: Mapped[str] = mapped_column(
        String(100),
        nullable=False,
    )

    risk_tier: Mapped[str] = mapped_column(
        String(50),
        nullable=False,
    )

    requested_action: Mapped[str] = mapped_column(
        String,
        nullable=False,
    )

    policy_metadata: Mapped[dict] = mapped_column(
        JSON().with_variant(JSONB, "postgresql"),
        nullable=False,
        default=dict,
    )

    resolved_by: Mapped[str | None] = mapped_column(
        String(255),
        nullable=True,
    )

    resolution_reason: Mapped[str | None] = mapped_column(
        String,
        nullable=True,
    )

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
    )

    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
        onupdate=func.now(),
    )

    resolved_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
    )

    __table_args__ = (
        UniqueConstraint(
            "run_id",
            "step_id",
            "call_id",
            name="uq_approval_requests_run_step_call",
        ),
        Index(
            "ix_approval_requests_run_step",
            "run_id",
            "step_id",
        ),
    )


class ToolExecutionIdempotencyRecord(Base):
    __tablename__ = "tool_execution_idempotency"

    run_id: Mapped[str] = mapped_column(
        String(36),
        nullable=False,
    )

    call_id: Mapped[str] = mapped_column(
        String(255),
        nullable=False,
    )

    tool_name: Mapped[str] = mapped_column(
        String(255),
        nullable=False,
    )

    status: Mapped[str] = mapped_column(
        String(50),
        nullable=False,
        index=True,
    )
    claim_token: Mapped[str | None] = mapped_column(
        String(36),
        nullable=True,
        index=True,
    )

    success: Mapped[bool] = mapped_column(
        nullable=False,
    )

    output: Mapped[object | None] = mapped_column(
        JSON().with_variant(JSONB, "postgresql"),
        nullable=True,
    )

    error: Mapped[str | None] = mapped_column(
        String,
        nullable=True,
    )

    failure_category: Mapped[str | None] = mapped_column(
        String(100),
        nullable=True,
    )

    execution_metadata: Mapped[dict] = mapped_column(
        "metadata",
        JSON().with_variant(JSONB, "postgresql"),
        nullable=False,
        default=dict,
    )

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
    )

    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
        onupdate=func.now(),
    )

    __table_args__ = (
        PrimaryKeyConstraint(
            "run_id",
            "call_id",
            "tool_name",
            name="pk_tool_execution_idempotency",
        ),
    )


class AgentRunEventRecord(Base):
    __tablename__ = "agent_run_events"

    id: Mapped[int] = mapped_column(
        Integer,
        primary_key=True,
        autoincrement=True,
    )

    event_id: Mapped[str] = mapped_column(
        String(36),
        nullable=False,
        unique=True,
        index=True,
    )

    run_id: Mapped[str] = mapped_column(
        String(36),
        ForeignKey("agent_runs.run_id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )

    event_type: Mapped[str] = mapped_column(
        String(100),
        nullable=False,
        index=True,
    )

    agent_name: Mapped[str] = mapped_column(
        String(255),
        nullable=False,
        index=True,
    )

    session_id: Mapped[str | None] = mapped_column(
        String(255),
        nullable=True,
        index=True,
    )

    user_id: Mapped[str | None] = mapped_column(
        String(255),
        nullable=True,
        index=True,
    )

    principal: Mapped[str | None] = mapped_column(String(255), nullable=True, index=True)

    tool_round: Mapped[int | None] = mapped_column(
        Integer,
        nullable=True,
    )

    tool_name: Mapped[str | None] = mapped_column(
        String(255),
        nullable=True,
    )

    call_id: Mapped[str | None] = mapped_column(
        String(255),
        nullable=True,
    )

    step_id: Mapped[str | None] = mapped_column(
        String(255),
        nullable=True,
    )

    step_index: Mapped[int | None] = mapped_column(
        Integer,
        nullable=True,
    )

    step_name: Mapped[str | None] = mapped_column(
        String(255),
        nullable=True,
    )

    provider: Mapped[str | None] = mapped_column(
        String(255),
        nullable=True,
    )

    model: Mapped[str | None] = mapped_column(
        String(255),
        nullable=True,
    )

    event_metadata: Mapped[dict] = mapped_column(
        "metadata",
        JSON().with_variant(JSONB, "postgresql"),
        nullable=False,
        default=dict,
    )

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
        index=True,
    )

    __table_args__ = (
        Index(
            "ix_agent_run_events_run_created_at",
            "run_id",
            "created_at",
        ),
        Index(
            "ix_agent_run_events_run_id_created_at_id",
            "run_id",
            "created_at",
            "id",
        ),
    )


class MemoryItemRecord(Base):
    __tablename__ = "memory_items"

    id: Mapped[str] = mapped_column(
        String(255),
        primary_key=True,
    )

    memory_type: Mapped[str] = mapped_column(
        String(50),
        nullable=False,
        index=True,
    )

    content: Mapped[str] = mapped_column(
        String,
        nullable=False,
    )

    namespace: Mapped[str] = mapped_column(
        String(255),
        nullable=False,
        index=True,
    )

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        index=True,
    )

    expires_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
        index=True,
    )

    memory_metadata: Mapped[dict] = mapped_column(
        "metadata",
        JSON().with_variant(JSONB, "postgresql"),
        nullable=False,
        default=dict,
    )


class MemoryEmbeddingRecord(Base):
    __tablename__ = "memory_embeddings"

    memory_id: Mapped[str] = mapped_column(
        String(255),
        ForeignKey("memory_items.id", ondelete="CASCADE"),
        primary_key=True,
    )

    embedding: Mapped[list[float]] = mapped_column(
        Vector(),
        nullable=False,
    )

    embedding_dimension: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
    )

    embedding_requested_provider: Mapped[str | None] = mapped_column(
        String(100),
        nullable=True,
    )

    embedding_requested_model: Mapped[str] = mapped_column(
        String(255),
        nullable=False,
    )

    embedding_resolved_provider: Mapped[str] = mapped_column(
        String(100),
        nullable=False,
    )

    embedding_resolved_model: Mapped[str] = mapped_column(
        String(255),
        nullable=False,
    )

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
    )


class UsageEventRecord(Base):
    __tablename__ = "usage_events"

    id: Mapped[int] = mapped_column(
        Integer,
        primary_key=True,
        autoincrement=True,
    )

    event_id: Mapped[str] = mapped_column(
        String(36),
        unique=True,
        nullable=False,
        index=True,
    )

    request_id: Mapped[str] = mapped_column(
        String(255),
        nullable=False,
        index=True,
    )

    timestamp: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
        index=True,
    )

    capability: Mapped[str] = mapped_column(
        String(255),
        nullable=False,
        index=True,
    )

    provider: Mapped[str | None] = mapped_column(
        String(100),
        nullable=True,
    )

    model: Mapped[str | None] = mapped_column(
        String(255),
        nullable=True,
    )

    tokens_in: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        default=0,
    )

    tokens_out: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        default=0,
    )

    estimated_cost: Mapped[float] = mapped_column(
        Float,
        nullable=False,
        default=0.0,
    )

    latency_ms: Mapped[float] = mapped_column(
        Float,
        nullable=False,
        default=0.0,
    )

    status: Mapped[str] = mapped_column(
        String(50),
        nullable=False,
        index=True,
    )


class RAGDocumentRecord(Base):
    __tablename__ = "rag_documents"

    id: Mapped[int] = mapped_column(
        Integer,
        primary_key=True,
        autoincrement=True,
    )

    document_id: Mapped[str] = mapped_column(
        String(255),
        unique=True,
        nullable=False,
        index=True,
    )

    content: Mapped[str] = mapped_column(
        String,
        nullable=False,
    )

    document_metadata: Mapped[dict] = mapped_column(
        "metadata",
        JSON().with_variant(JSONB, "postgresql"),
        nullable=False,
        default=dict,
    )

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
    )

    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
        onupdate=func.now(),
    )

    chunks: Mapped[list["RAGChunkRecord"]] = relationship(
        back_populates="document",
        cascade="all, delete-orphan",
    )


class RAGChunkRecord(Base):
    __tablename__ = "rag_chunks"

    id: Mapped[int] = mapped_column(
        Integer,
        primary_key=True,
        autoincrement=True,
    )

    chunk_id: Mapped[str] = mapped_column(
        String(255),
        unique=True,
        nullable=False,
        index=True,
    )

    document_id: Mapped[str] = mapped_column(
        String(255),
        ForeignKey("rag_documents.document_id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )

    chunk_index: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
    )

    content: Mapped[str] = mapped_column(
        String,
        nullable=False,
    )

    chunk_metadata: Mapped[dict] = mapped_column(
        "metadata",
        JSON().with_variant(JSONB, "postgresql"),
        nullable=False,
        default=dict,
    )

    embedding: Mapped[list[float] | None] = mapped_column(
        Vector(),
        nullable=True,
    )

    embedding_model: Mapped[str | None] = mapped_column(
        String(255),
        nullable=True,
    )

    embedding_dimension: Mapped[int | None] = mapped_column(
        Integer,
        nullable=True,
    )

    embedding_requested_provider: Mapped[str | None] = mapped_column(
        String(100),
        nullable=True,
    )

    embedding_requested_model: Mapped[str | None] = mapped_column(
        String(255),
        nullable=True,
    )

    embedding_resolved_provider: Mapped[str | None] = mapped_column(
        String(100),
        nullable=True,
    )

    embedding_resolved_model: Mapped[str | None] = mapped_column(
        String(255),
        nullable=True,
    )

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
    )

    document: Mapped[RAGDocumentRecord] = relationship(
        back_populates="chunks",
    )


class AgentRunCheckpointRecord(Base):
    __tablename__ = "agent_run_checkpoints"

    id: Mapped[int] = mapped_column(
        Integer,
        primary_key=True,
        autoincrement=True,
    )

    run_id: Mapped[str] = mapped_column(
        String(36),
        ForeignKey("agent_runs.run_id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )

    agent_name: Mapped[str] = mapped_column(
        String(255),
        nullable=False,
        index=True,
    )

    session_id: Mapped[str | None] = mapped_column(
        String(255),
        nullable=True,
        index=True,
    )

    user_id: Mapped[str | None] = mapped_column(
        String(255),
        nullable=True,
        index=True,
    )

    principal: Mapped[str | None] = mapped_column(String(255), nullable=True, index=True)

    schema_version: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
    )

    position: Mapped[str] = mapped_column(
        String(100),
        nullable=False,
        index=True,
    )

    tool_round: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
    )

    checkpoint_payload: Mapped[dict] = mapped_column(
        JSON().with_variant(JSONB, "postgresql"),
        nullable=False,
    )

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
        index=True,
    )

    __table_args__ = (
        Index(
            "ix_agent_run_checkpoints_run_created_at",
            "run_id",
            "created_at",
        ),
    )


class VoiceCallRecord(Base):
    """Tenant-owned call state, excluding transcripts and raw provider payloads."""

    __tablename__ = "voice_calls"
    __table_args__ = (
        UniqueConstraint(
            "provider_key",
            "provider_call_id",
            name="uq_voice_calls_provider_call",
        ),
        CheckConstraint(
            "status IN ('received', 'in_progress', 'completed', 'failed')",
            name="ck_voice_calls_status",
        ),
        Index(
            "ix_voice_calls_tenant_created_at",
            "tenant_id",
            "created_at",
        ),
        Index(
            "ix_voice_calls_tenant_status",
            "tenant_id",
            "status",
        ),
    )

    call_id: Mapped[str] = mapped_column(String(36), primary_key=True)
    tenant_id: Mapped[str] = mapped_column(String(255), nullable=False)
    voice_line_id: Mapped[int] = mapped_column(
        ForeignKey("voice_line_configs.id", ondelete="RESTRICT"),
        nullable=False,
    )
    provider_key: Mapped[str] = mapped_column(String(100), nullable=False)
    provider_call_id: Mapped[str] = mapped_column(String(255), nullable=False)
    business_line_e164: Mapped[str] = mapped_column(String(16), nullable=False)
    status: Mapped[str] = mapped_column(
        String(24), nullable=False, default="received", server_default="received"
    )
    started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    completed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    last_event_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
        onupdate=func.now(),
    )


class VoiceProviderEventRecord(Base):
    """Deduplicated provider event metadata; deliberately stores no raw payload."""

    __tablename__ = "voice_provider_events"
    __table_args__ = (
        UniqueConstraint(
            "provider_key",
            "provider_event_id",
            name="uq_voice_provider_events_provider_event",
        ),
        Index(
            "ix_voice_provider_events_tenant_received_at",
            "tenant_id",
            "received_at",
        ),
        Index(
            "ix_voice_provider_events_call_occurred_at",
            "call_id",
            "occurred_at",
        ),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    call_id: Mapped[str] = mapped_column(
        ForeignKey("voice_calls.call_id", ondelete="RESTRICT"),
        nullable=False,
    )
    tenant_id: Mapped[str] = mapped_column(String(255), nullable=False)
    provider_key: Mapped[str] = mapped_column(String(100), nullable=False)
    provider_event_id: Mapped[str] = mapped_column(String(255), nullable=False)
    provider_call_id: Mapped[str] = mapped_column(String(255), nullable=False)
    event_type: Mapped[str] = mapped_column(String(32), nullable=False)
    occurred_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    received_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )


class VoiceLineConfigRecord(Base):
    """Tenant-scoped voice line and agent configuration."""

    __tablename__ = "voice_line_configs"
    __table_args__ = (
        UniqueConstraint(
            "configured_phone_number",
            name="uq_voice_line_configs_configured_phone_number",
        ),
        CheckConstraint(
            "line_status IN ('active', 'paused', 'offline')",
            name="ck_voice_line_configs_line_status",
        ),
        CheckConstraint(
            "provider_sync_status IN ('not_configured', 'pending', 'synced', 'error')",
            name="ck_voice_line_configs_provider_sync_status",
        ),
        CheckConstraint(
            "conversational_pause_pacing >= 0 AND conversational_pause_pacing <= 10",
            name="ck_voice_line_configs_pause_pacing",
        ),
        CheckConstraint(
            "barge_in_sensitivity BETWEEN 1 AND 10",
            name="ck_voice_line_configs_barge_in_sensitivity",
        ),
        CheckConstraint(
            "max_call_duration_minutes BETWEEN 1 AND 240",
            name="ck_voice_line_configs_max_call_duration",
        ),
        Index(
            "ix_voice_line_configs_tenant_updated_at",
            "tenant_id",
            "updated_at",
        ),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    tenant_id: Mapped[str] = mapped_column(String(255), nullable=False)
    configured_phone_number: Mapped[str] = mapped_column(String(32), nullable=False)
    carrier_trunk_line: Mapped[str] = mapped_column(String(255), nullable=False)
    call_forwarding_code: Mapped[str | None] = mapped_column(String(128), nullable=True)

    # A saved configuration is not proof that calls are routing.
    line_status: Mapped[str] = mapped_column(
        String(16), nullable=False, default="paused", server_default="paused"
    )

    inbound_opening_greeting: Mapped[str] = mapped_column(Text, nullable=False)
    inbound_system_prompt_override: Mapped[str | None] = mapped_column(Text, nullable=True)
    outbound_opening_message: Mapped[str] = mapped_column(Text, nullable=False)
    outbound_system_prompt_override: Mapped[str | None] = mapped_column(Text, nullable=True)

    provider_key: Mapped[str | None] = mapped_column(String(100), nullable=True)
    agent_uuid: Mapped[str | None] = mapped_column(String(255), nullable=True)
    provider_sync_status: Mapped[str] = mapped_column(
        String(24),
        nullable=False,
        default="not_configured",
        server_default="not_configured",
    )
    last_provider_sync_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    # Store a sanitized code, not credentials or raw provider responses.
    last_provider_sync_error_code: Mapped[str | None] = mapped_column(String(100), nullable=True)

    voice_synthesizer: Mapped[str] = mapped_column(
        String(255),
        nullable=False,
        default="polly-kajal",
        server_default="polly-kajal",
    )
    spoken_language: Mapped[str] = mapped_column(
        String(20), nullable=False, default="hi-IN", server_default="hi-IN"
    )
    conversational_pause_pacing: Mapped[float] = mapped_column(
        Float, nullable=False, default=1.0, server_default="1.0"
    )
    barge_in_sensitivity: Mapped[int] = mapped_column(
        Integer, nullable=False, default=2, server_default="2"
    )
    llm_reasoning_model: Mapped[str] = mapped_column(
        String(255),
        nullable=False,
        default="openai/gpt-4o-mini",
        server_default="openai/gpt-4o-mini",
    )
    max_call_duration_minutes: Mapped[int] = mapped_column(
        Integer, nullable=False, default=10, server_default="10"
    )

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
        onupdate=func.now(),
    )


class VoiceCallTurnRecord(Base):
    """Durable state for one inbound voice interaction turn."""

    __tablename__ = "voice_call_turns"

    turn_id: Mapped[str] = mapped_column(
        String(36),
        primary_key=True,
    )

    tenant_id: Mapped[str] = mapped_column(
        String(255),
        nullable=False,
    )

    call_id: Mapped[str] = mapped_column(
        String(36),
        ForeignKey("voice_calls.call_id", ondelete="RESTRICT"),
        nullable=False,
    )

    turn_number: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
    )

    callback_token: Mapped[str] = mapped_column(
        String(128),
        nullable=False,
    )

    status: Mapped[str] = mapped_column(
        String(24),
        nullable=False,
        server_default="awaiting_input",
    )

    # SHA-256 fingerprint of normalized recognized input; never the transcript.
    request_fingerprint: Mapped[str | None] = mapped_column(
        String(64),
        nullable=True,
    )

    # Opaque ownership token; never derived from recognized speech.
    claim_token: Mapped[str | None] = mapped_column(
        String(36),
        nullable=True,
    )

    agent_run_id: Mapped[str | None] = mapped_column(
        String(36),
        nullable=True,
    )

    # Persist only the bounded response needed for safe callback replay.
    response_text: Mapped[str | None] = mapped_column(
        Text,
        nullable=True,
    )

    failure_code: Mapped[str | None] = mapped_column(
        String(100),
        nullable=True,
    )

    started_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
    )

    completed_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
    )

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
    )

    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
        onupdate=func.now(),
    )

    __table_args__ = (
        UniqueConstraint(
            "tenant_id",
            "call_id",
            "turn_number",
            name="uq_voice_call_turns_tenant_call_turn",
        ),
        UniqueConstraint(
            "callback_token",
            name="uq_voice_call_turns_callback_token",
        ),
        CheckConstraint(
            "turn_number >= 1",
            name="ck_voice_call_turns_positive_turn_number",
        ),
        CheckConstraint(
            "status IN ('awaiting_input', 'processing', 'completed', " "'failed', 'no_input')",
            name="ck_voice_call_turns_status",
        ),
        CheckConstraint(
            "request_fingerprint IS NULL OR length(request_fingerprint) = 64",
            name="ck_voice_call_turns_fingerprint_length",
        ),
        CheckConstraint(
            "response_text IS NULL OR length(response_text) <= 8000",
            name="ck_voice_call_turns_response_length",
        ),
        Index(
            "ix_voice_call_turns_tenant_status",
            "tenant_id",
            "status",
        ),
        Index(
            "ix_voice_call_turns_call_turn_number",
            "call_id",
            "turn_number",
        ),
    )
