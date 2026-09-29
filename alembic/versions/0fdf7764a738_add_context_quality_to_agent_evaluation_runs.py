"""add context quality to agent evaluation runs

Revision ID: 0fdf7764a738
Revises: 20260928000000
Create Date: 2026-09-29
"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql


revision: str = "0fdf7764a738"
down_revision: Union[str, Sequence[str], None] = "20260928000000"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column(
        "agent_evaluation_runs",
        sa.Column(
            "context_quality",
            sa.JSON().with_variant(
                postgresql.JSONB(),
                "postgresql",
            ),
            nullable=True,
        ),
    )


def downgrade() -> None:
    op.drop_column(
        "agent_evaluation_runs",
        "context_quality",
    )
