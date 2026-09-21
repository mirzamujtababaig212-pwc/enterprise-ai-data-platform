"""add claim token to tool execution idempotency

Revision ID: d39f49619987
Revises: 51056e01ef80
Create Date: 2026-09-21
"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa

revision: str = "d39f49619987"
down_revision: Union[str, Sequence[str], None] = "51056e01ef80"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column(
        "tool_execution_idempotency",
        sa.Column("claim_token", sa.String(length=36), nullable=True),
    )

    op.create_index(
        "ix_tool_execution_idempotency_claim_token",
        "tool_execution_idempotency",
        ["claim_token"],
    )


def downgrade() -> None:
    op.drop_index(
        "ix_tool_execution_idempotency_claim_token",
        table_name="tool_execution_idempotency",
    )
    op.drop_column(
        "tool_execution_idempotency",
        "claim_token",
    )
