"""create agent run checkpoints

Revision ID: b7c3d91e4f20
Revises: a91c4e7f2b10
Create Date: 2026-09-19
"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql


revision: str = "b7c3d91e4f20"
down_revision: Union[str, Sequence[str], None] = "a91c4e7f2b10"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "agent_run_checkpoints",
        sa.Column(
            "id",
            sa.Integer(),
            autoincrement=True,
            nullable=False,
        ),
        sa.Column(
            "run_id",
            sa.String(length=36),
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
            "user_id",
            sa.String(length=255),
            nullable=True,
        ),
        sa.Column(
            "schema_version",
            sa.Integer(),
            nullable=False,
        ),
        sa.Column(
            "position",
            sa.String(length=100),
            nullable=False,
        ),
        sa.Column(
            "tool_round",
            sa.Integer(),
            nullable=False,
        ),
        sa.Column(
            "checkpoint_payload",
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
    )

    op.create_index(
        "ix_agent_run_checkpoints_run_id",
        "agent_run_checkpoints",
        ["run_id"],
    )
    op.create_index(
        "ix_agent_run_checkpoints_agent_name",
        "agent_run_checkpoints",
        ["agent_name"],
    )
    op.create_index(
        "ix_agent_run_checkpoints_session_id",
        "agent_run_checkpoints",
        ["session_id"],
    )
    op.create_index(
        "ix_agent_run_checkpoints_user_id",
        "agent_run_checkpoints",
        ["user_id"],
    )
    op.create_index(
        "ix_agent_run_checkpoints_position",
        "agent_run_checkpoints",
        ["position"],
    )
    op.create_index(
        "ix_agent_run_checkpoints_created_at",
        "agent_run_checkpoints",
        ["created_at"],
    )
    op.create_index(
        "ix_agent_run_checkpoints_run_created_at",
        "agent_run_checkpoints",
        ["run_id", "created_at"],
    )


def downgrade() -> None:
    op.drop_index(
        "ix_agent_run_checkpoints_run_created_at",
        table_name="agent_run_checkpoints",
    )
    op.drop_index(
        "ix_agent_run_checkpoints_created_at",
        table_name="agent_run_checkpoints",
    )
    op.drop_index(
        "ix_agent_run_checkpoints_position",
        table_name="agent_run_checkpoints",
    )
    op.drop_index(
        "ix_agent_run_checkpoints_user_id",
        table_name="agent_run_checkpoints",
    )
    op.drop_index(
        "ix_agent_run_checkpoints_session_id",
        table_name="agent_run_checkpoints",
    )
    op.drop_index(
        "ix_agent_run_checkpoints_agent_name",
        table_name="agent_run_checkpoints",
    )
    op.drop_index(
        "ix_agent_run_checkpoints_run_id",
        table_name="agent_run_checkpoints",
    )
    op.drop_table("agent_run_checkpoints")
