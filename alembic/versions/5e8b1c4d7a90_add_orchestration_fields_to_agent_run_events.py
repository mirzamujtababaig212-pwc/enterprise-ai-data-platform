"""add orchestration fields to agent run events

Revision ID: 5e8b1c4d7a90
Revises: 4030ddbb4224
Create Date: 2026-09-23
"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa

revision: str = "5e8b1c4d7a90"
down_revision: Union[str, Sequence[str], None] = "4030ddbb4224"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column(
        "agent_run_events",
        sa.Column("step_id", sa.String(length=255), nullable=True),
    )
    op.add_column(
        "agent_run_events",
        sa.Column("step_index", sa.Integer(), nullable=True),
    )
    op.add_column(
        "agent_run_events",
        sa.Column("step_name", sa.String(length=255), nullable=True),
    )


def downgrade() -> None:
    op.drop_column("agent_run_events", "step_name")
    op.drop_column("agent_run_events", "step_index")
    op.drop_column("agent_run_events", "step_id")
