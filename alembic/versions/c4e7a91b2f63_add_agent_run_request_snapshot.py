"""add agent run request snapshot

Revision ID: c4e7a91b2f63
Revises: b7c3d91e4f20
Create Date: 2026-09-19
"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql


revision: str = "c4e7a91b2f63"
down_revision: Union[str, Sequence[str], None] = "b7c3d91e4f20"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column(
        "agent_runs",
        sa.Column(
            "request_snapshot",
            sa.JSON().with_variant(postgresql.JSONB(), "postgresql"),
            nullable=True,
        ),
    )


def downgrade() -> None:
    op.drop_column(
        "agent_runs",
        "request_snapshot",
    )
