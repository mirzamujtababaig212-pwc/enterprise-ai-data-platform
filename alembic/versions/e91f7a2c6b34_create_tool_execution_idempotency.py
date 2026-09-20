"""create tool execution idempotency

Revision ID: e91f7a2c6b34
Revises: db2ca1d37aec
Create Date: 2026-09-20
"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

revision: str = "e91f7a2c6b34"
down_revision: Union[str, Sequence[str], None] = "db2ca1d37aec"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "tool_execution_idempotency",
        sa.Column("run_id", sa.String(length=36), nullable=False),
        sa.Column("call_id", sa.String(length=255), nullable=False),
        sa.Column("tool_name", sa.String(length=255), nullable=False),
        sa.Column("status", sa.String(length=50), nullable=False),
        sa.Column("success", sa.Boolean(), nullable=False),
        sa.Column(
            "output",
            sa.JSON().with_variant(postgresql.JSONB(), "postgresql"),
            nullable=True,
        ),
        sa.Column("error", sa.String(), nullable=True),
        sa.Column("failure_category", sa.String(length=100), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
        sa.PrimaryKeyConstraint(
            "run_id",
            "call_id",
            "tool_name",
            name="pk_tool_execution_idempotency",
        ),
    )

    op.create_index(
        "ix_tool_execution_idempotency_status",
        "tool_execution_idempotency",
        ["status"],
    )


def downgrade() -> None:
    op.drop_index(
        "ix_tool_execution_idempotency_status",
        table_name="tool_execution_idempotency",
    )
    op.drop_table("tool_execution_idempotency")
