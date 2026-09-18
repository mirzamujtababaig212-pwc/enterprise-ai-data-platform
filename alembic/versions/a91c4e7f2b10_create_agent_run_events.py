"""create agent run events

Revision ID: a91c4e7f2b10
Revises: 77e72db32a00
Create Date: 2026-09-18
"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql


revision: str = "a91c4e7f2b10"
down_revision: Union[str, Sequence[str], None] = "77e72db32a00"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "agent_run_events",
        sa.Column(
            "id",
            sa.Integer(),
            autoincrement=True,
            nullable=False,
        ),
        sa.Column(
            "event_id",
            sa.String(length=36),
            nullable=False,
        ),
        sa.Column(
            "run_id",
            sa.String(length=36),
            nullable=False,
        ),
        sa.Column(
            "event_type",
            sa.String(length=100),
            nullable=False,
        ),
        sa.Column(
            "agent_name",
            sa.String(length=255),
            nullable=False,
        ),
        sa.Column(
            "session_id",
            sa.String(length=255),
            nullable=True,
        ),
        sa.Column(
            "tool_round",
            sa.Integer(),
            nullable=True,
        ),
        sa.Column(
            "tool_name",
            sa.String(length=255),
            nullable=True,
        ),
        sa.Column(
            "call_id",
            sa.String(length=255),
            nullable=True,
        ),
        sa.Column(
            "provider",
            sa.String(length=255),
            nullable=True,
        ),
        sa.Column(
            "model",
            sa.String(length=255),
            nullable=True,
        ),
        sa.Column(
            "metadata",
            sa.JSON().with_variant(postgresql.JSONB(), "postgresql"),
            nullable=False,
        ),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
        sa.ForeignKeyConstraint(
            ["run_id"],
            ["agent_runs.run_id"],
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("event_id"),
    )

    op.create_index(
        "ix_agent_run_events_event_id",
        "agent_run_events",
        ["event_id"],
    )
    op.create_index(
        "ix_agent_run_events_run_id",
        "agent_run_events",
        ["run_id"],
    )
    op.create_index(
        "ix_agent_run_events_event_type",
        "agent_run_events",
        ["event_type"],
    )
    op.create_index(
        "ix_agent_run_events_agent_name",
        "agent_run_events",
        ["agent_name"],
    )
    op.create_index(
        "ix_agent_run_events_session_id",
        "agent_run_events",
        ["session_id"],
    )
    op.create_index(
        "ix_agent_run_events_created_at",
        "agent_run_events",
        ["created_at"],
    )
    op.create_index(
        "ix_agent_run_events_run_created_at",
        "agent_run_events",
        ["run_id", "created_at"],
    )


def downgrade() -> None:
    op.drop_index(
        "ix_agent_run_events_run_created_at",
        table_name="agent_run_events",
    )
    op.drop_index(
        "ix_agent_run_events_created_at",
        table_name="agent_run_events",
    )
    op.drop_index(
        "ix_agent_run_events_session_id",
        table_name="agent_run_events",
    )
    op.drop_index(
        "ix_agent_run_events_agent_name",
        table_name="agent_run_events",
    )
    op.drop_index(
        "ix_agent_run_events_event_type",
        table_name="agent_run_events",
    )
    op.drop_index(
        "ix_agent_run_events_run_id",
        table_name="agent_run_events",
    )
    op.drop_index(
        "ix_agent_run_events_event_id",
        table_name="agent_run_events",
    )
    op.drop_table("agent_run_events")
