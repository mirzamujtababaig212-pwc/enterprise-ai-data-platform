"""add agent run recovery attempts

Revision ID: 51056e01ef80
Revises: f2a7c9d81b45
Create Date: 2026-09-21
"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


revision: str = "51056e01ef80"
down_revision: Union[str, Sequence[str], None] = "f2a7c9d81b45"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column(
        "agent_runs",
        sa.Column(
            "recovery_attempts",
            sa.Integer(),
            nullable=False,
            server_default="0",
        ),
    )
    op.alter_column(
        "agent_runs",
        "recovery_attempts",
        server_default=None,
    )


def downgrade() -> None:
    op.drop_column("agent_runs", "recovery_attempts")
