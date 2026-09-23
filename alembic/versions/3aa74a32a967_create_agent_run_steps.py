"""create agent run steps

Revision ID: 3aa74a32a967
Revises: 5e8b1c4d7a90
Create Date: 2026-09-23
"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql


# revision identifiers, used by Alembic.
revision: str = "3aa74a32a967"
down_revision: Union[str, Sequence[str], None] = "5e8b1c4d7a90"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "agent_run_steps",
        sa.Column(
            "run_id",
            sa.String(length=36),
            nullable=False,
        ),
        sa.Column(
            "step_id",
            sa.String(length=255),
            nullable=False,
        ),
        sa.Column(
            "step_index",
            sa.Integer(),
            nullable=False,
        ),
        sa.Column(
            "step_type",
            sa.String(length=100),
            nullable=False,
        ),
        sa.Column(
            "status",
            sa.String(length=50),
            nullable=False,
        ),
        sa.Column(
            "attempt",
            sa.Integer(),
            nullable=False,
            server_default="1",
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
            "input",
            postgresql.JSONB(astext_type=sa.Text()),
            nullable=True,
        ),
        sa.Column(
            "output",
            postgresql.JSONB(astext_type=sa.Text()),
            nullable=True,
        ),
        sa.Column(
            "error",
            sa.Text(),
            nullable=True,
        ),
        sa.Column(
            "failure_category",
            sa.String(length=100),
            nullable=True,
        ),
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
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.func.now(),
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.func.now(),
        ),
        sa.Column(
            "metadata",
            postgresql.JSONB(astext_type=sa.Text()),
            nullable=False,
            server_default=sa.text("'{}'::jsonb"),
        ),
        sa.ForeignKeyConstraint(
            ["run_id"],
            ["agent_runs.run_id"],
            name="fk_agent_run_steps_run_id_agent_runs",
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint(
            "run_id",
            "step_id",
            name="pk_agent_run_steps",
        ),
    )

    op.create_index(
        "ix_agent_run_steps_run_step_index",
        "agent_run_steps",
        ["run_id", "step_index"],
        unique=False,
    )

    op.create_index(
        "ix_agent_run_steps_run_status",
        "agent_run_steps",
        ["run_id", "status"],
        unique=False,
    )


def downgrade() -> None:
    op.drop_index(
        "ix_agent_run_steps_run_status",
        table_name="agent_run_steps",
    )
    op.drop_index(
        "ix_agent_run_steps_run_step_index",
        table_name="agent_run_steps",
    )
    op.drop_table("agent_run_steps")
