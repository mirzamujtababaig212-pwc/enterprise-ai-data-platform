"""add metadata to tool execution idempotency

Revision ID: b4e7c91d2f63
Revises: 8a1d4f6c9b27
Create Date: 2026-09-24
"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql


revision: str = "b4e7c91d2f63"
down_revision: Union[str, Sequence[str], None] = "8a1d4f6c9b27"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column(
        "tool_execution_idempotency",
        sa.Column(
            "metadata",
            sa.JSON().with_variant(postgresql.JSONB(), "postgresql"),
            nullable=True,
        ),
    )

    op.execute(
        sa.text(
            "UPDATE tool_execution_idempotency "
            "SET metadata = :empty_metadata "
            "WHERE metadata IS NULL"
        ).bindparams(
            empty_metadata="{}",
        )
    )

    op.alter_column(
        "tool_execution_idempotency",
        "metadata",
        nullable=False,
    )


def downgrade() -> None:
    op.drop_column(
        "tool_execution_idempotency",
        "metadata",
    )
