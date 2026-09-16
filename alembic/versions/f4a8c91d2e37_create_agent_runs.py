"""create agent runs

Revision ID: f4a8c91d2e37
Revises: d826184cc317
Create Date: 2026-09-17
"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

revision: str = "f4a8c91d2e37"
down_revision: Union[str, Sequence[str], None] = "d826184cc317"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "agent_runs",
        sa.Column("run_id", sa.String(length=36), nullable=False),
        sa.Column("agent_name", sa.String(length=255), nullable=False),
        sa.Column("session_id", sa.String(length=255), nullable=True),
        sa.Column("user_id", sa.String(length=255), nullable=True),
        sa.Column("status", sa.String(length=50), nullable=False),
        sa.Column(
            "started_at",
            sa.DateTime(timezone=True),
            nullable=True,
        ),
        sa.Column(
            "completed_at",
            sa.DateTime(timezone=True),
            nullable=True,
        ),
        sa.Column("error_type", sa.String(length=255), nullable=True),
        sa.Column("error_message", sa.String(), nullable=True),
        sa.Column(
            "output",
            sa.JSON().with_variant(postgresql.JSONB(), "postgresql"),
            nullable=True,
        ),
        sa.Column(
            "metadata",
            sa.JSON().with_variant(postgresql.JSONB(), "postgresql"),
            nullable=False,
        ),
        sa.PrimaryKeyConstraint("run_id"),
    )

    op.create_index(
        "ix_agent_runs_agent_name",
        "agent_runs",
        ["agent_name"],
    )
    op.create_index(
        "ix_agent_runs_session_id",
        "agent_runs",
        ["session_id"],
    )
    op.create_index(
        "ix_agent_runs_user_id",
        "agent_runs",
        ["user_id"],
    )
    op.create_index(
        "ix_agent_runs_status",
        "agent_runs",
        ["status"],
    )
    op.create_index(
        "ix_agent_runs_started_at",
        "agent_runs",
        ["started_at"],
    )


def downgrade() -> None:
    op.drop_index(
        "ix_agent_runs_started_at",
        table_name="agent_runs",
    )
    op.drop_index(
        "ix_agent_runs_status",
        table_name="agent_runs",
    )
    op.drop_index(
        "ix_agent_runs_user_id",
        table_name="agent_runs",
    )
    op.drop_index(
        "ix_agent_runs_session_id",
        table_name="agent_runs",
    )
    op.drop_index(
        "ix_agent_runs_agent_name",
        table_name="agent_runs",
    )
    op.drop_table("agent_runs")
