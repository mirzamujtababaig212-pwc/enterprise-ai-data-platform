"""add agent run events cursor index

Revision ID: 20261002010000
Revises: 20260930020000
Create Date: 2026-10-02
"""

from typing import Sequence, Union

from alembic import op


revision: str = "20261002010000"
down_revision: Union[str, Sequence[str], None] = "20260930020000"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_index(
        "ix_agent_run_events_run_id_created_at_id",
        "agent_run_events",
        ["run_id", "created_at", "id"],
        unique=False,
    )


def downgrade() -> None:
    op.drop_index(
        "ix_agent_run_events_run_id_created_at_id",
        table_name="agent_run_events",
    )
