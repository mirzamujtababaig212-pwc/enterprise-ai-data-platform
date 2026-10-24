"""Create persisted voice calls and deduplicated provider events.

Revision ID: 20261018020000
Revises: 20261018010000
Create Date: 2026-10-18
"""

from collections.abc import Sequence

import sqlalchemy as sa

from alembic import op

revision: str = "20261018020000"
down_revision: str | Sequence[str] | None = "20261018010000"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "voice_calls",
        sa.Column("call_id", sa.String(length=36), nullable=False),
        sa.Column("tenant_id", sa.String(length=255), nullable=False),
        sa.Column("voice_line_id", sa.Integer(), nullable=False),
        sa.Column("provider_key", sa.String(length=100), nullable=False),
        sa.Column("provider_call_id", sa.String(length=255), nullable=False),
        sa.Column("business_line_e164", sa.String(length=16), nullable=False),
        sa.Column(
            "status",
            sa.String(length=24),
            server_default="received",
            nullable=False,
        ),
        sa.Column("started_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("completed_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("last_event_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
        sa.CheckConstraint(
            "status IN ('received', 'in_progress', 'completed', 'failed')",
            name="ck_voice_calls_status",
        ),
        sa.ForeignKeyConstraint(
            ["voice_line_id"],
            ["voice_line_configs.id"],
            ondelete="RESTRICT",
        ),
        sa.PrimaryKeyConstraint("call_id"),
        sa.UniqueConstraint(
            "provider_key",
            "provider_call_id",
            name="uq_voice_calls_provider_call",
        ),
    )
    op.create_index(
        "ix_voice_calls_tenant_created_at",
        "voice_calls",
        ["tenant_id", "created_at"],
        unique=False,
    )
    op.create_index(
        "ix_voice_calls_tenant_status",
        "voice_calls",
        ["tenant_id", "status"],
        unique=False,
    )

    op.create_table(
        "voice_provider_events",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("call_id", sa.String(length=36), nullable=False),
        sa.Column("tenant_id", sa.String(length=255), nullable=False),
        sa.Column("provider_key", sa.String(length=100), nullable=False),
        sa.Column("provider_event_id", sa.String(length=255), nullable=False),
        sa.Column("provider_call_id", sa.String(length=255), nullable=False),
        sa.Column("event_type", sa.String(length=32), nullable=False),
        sa.Column("occurred_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column(
            "received_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
        sa.ForeignKeyConstraint(
            ["call_id"],
            ["voice_calls.call_id"],
            ondelete="RESTRICT",
        ),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint(
            "provider_key",
            "provider_event_id",
            name="uq_voice_provider_events_provider_event",
        ),
    )
    op.create_index(
        "ix_voice_provider_events_tenant_received_at",
        "voice_provider_events",
        ["tenant_id", "received_at"],
        unique=False,
    )
    op.create_index(
        "ix_voice_provider_events_call_occurred_at",
        "voice_provider_events",
        ["call_id", "occurred_at"],
        unique=False,
    )


def downgrade() -> None:
    op.drop_index(
        "ix_voice_provider_events_call_occurred_at",
        table_name="voice_provider_events",
    )
    op.drop_index(
        "ix_voice_provider_events_tenant_received_at",
        table_name="voice_provider_events",
    )
    op.drop_table("voice_provider_events")

    op.drop_index(
        "ix_voice_calls_tenant_status",
        table_name="voice_calls",
    )
    op.drop_index(
        "ix_voice_calls_tenant_created_at",
        table_name="voice_calls",
    )
    op.drop_table("voice_calls")
