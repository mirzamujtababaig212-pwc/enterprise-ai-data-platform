"""create memory items

Revision ID: cff669198c0d
Revises: f4a8c91d2e37
Create Date: 2026-09-17
"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

revision: str = "cff669198c0d"
down_revision: Union[str, Sequence[str], None] = "f4a8c91d2e37"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "memory_items",
        sa.Column("id", sa.String(length=255), nullable=False),
        sa.Column("memory_type", sa.String(length=50), nullable=False),
        sa.Column("content", sa.Text(), nullable=False),
        sa.Column("namespace", sa.String(length=255), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
        ),
        sa.Column(
            "expires_at",
            sa.DateTime(timezone=True),
            nullable=True,
        ),
        sa.Column(
            "metadata",
            sa.JSON().with_variant(postgresql.JSONB(astext_type=sa.Text()), "postgresql"),
            nullable=False,
        ),
        sa.PrimaryKeyConstraint("id"),
    )

    op.create_index(
        "ix_memory_items_namespace",
        "memory_items",
        ["namespace"],
    )
    op.create_index(
        "ix_memory_items_memory_type",
        "memory_items",
        ["memory_type"],
    )
    op.create_index(
        "ix_memory_items_created_at",
        "memory_items",
        ["created_at"],
    )
    op.create_index(
        "ix_memory_items_expires_at",
        "memory_items",
        ["expires_at"],
    )


def downgrade() -> None:
    op.drop_index("ix_memory_items_expires_at", table_name="memory_items")
    op.drop_index("ix_memory_items_created_at", table_name="memory_items")
    op.drop_index("ix_memory_items_memory_type", table_name="memory_items")
    op.drop_index("ix_memory_items_namespace", table_name="memory_items")
    op.drop_table("memory_items")
