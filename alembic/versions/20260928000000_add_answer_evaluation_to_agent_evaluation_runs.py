"""add answer evaluation to agent evaluation runs

Revision ID: 20260928000000
Revises: 20260927approval_override
Create Date: 2026-09-28
"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql


revision: str = "20260928000000"
down_revision: Union[str, Sequence[str], None] = "20260927approval_override"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column(
        "agent_evaluation_runs",
        sa.Column(
            "answer_evaluation",
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
        "answer_evaluation",
    )
