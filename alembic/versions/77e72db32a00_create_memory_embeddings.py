"""create memory embeddings

Revision ID: 77e72db32a00
Revises: 9b97f6f62cf7
Create Date: 2026-09-17 10:11:18.175214

"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from pgvector.sqlalchemy import Vector

# revision identifiers, used by Alembic.
revision: str = "77e72db32a00"
down_revision: Union[str, Sequence[str], None] = "9b97f6f62cf7"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    op.create_table(
        "memory_embeddings",
        sa.Column(
            "memory_id",
            sa.String(length=255),
            sa.ForeignKey("memory_items.id", ondelete="CASCADE"),
            primary_key=True,
        ),
        sa.Column(
            "embedding",
            Vector(),
            nullable=False,
        ),
        sa.Column(
            "embedding_dimension",
            sa.Integer(),
            nullable=False,
        ),
        sa.Column(
            "embedding_requested_provider",
            sa.String(length=100),
            nullable=True,
        ),
        sa.Column(
            "embedding_requested_model",
            sa.String(length=255),
            nullable=False,
        ),
        sa.Column(
            "embedding_resolved_provider",
            sa.String(length=100),
            nullable=False,
        ),
        sa.Column(
            "embedding_resolved_model",
            sa.String(length=255),
            nullable=False,
        ),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.func.now(),
        ),
    )


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_table("memory_embeddings")
