"""add agent run cancellation intent

Revision ID: f2a7c9d81b45
Revises: e91f7a2c6b34
Create Date: 2026-09-20
"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


revision: str = "f2a7c9d81b45"
down_revision: Union[str, Sequence[str], None] = "e91f7a2c6b34"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column(
        "agent_runs",
        sa.Column(
            "cancellation_requested",
            sa.Boolean(),
            nullable=False,
            server_default="false",
        ),
    )

    op.add_column(
        "agent_runs",
        sa.Column(
            "cancellation_requested_at",
            sa.DateTime(timezone=True),
            nullable=True,
        ),
    )


def downgrade() -> None:
    op.drop_column(
        "agent_runs",
        "cancellation_requested_at",
    )

    op.drop_column(
        "agent_runs",
        "cancellation_requested",
    )
