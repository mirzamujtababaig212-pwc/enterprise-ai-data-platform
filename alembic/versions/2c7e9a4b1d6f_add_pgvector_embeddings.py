"""add pgvector embeddings to rag chunks

Revision ID: 2c7e9a4b1d6f
Revises: 8f2c6a1d4b7e
Create Date: 2026-09-13 00:00:00.000000
"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from pgvector.sqlalchemy import Vector


revision: str = "2c7e9a4b1d6f"
down_revision: Union[str, Sequence[str], None] = "8f2c6a1d4b7e"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    op.execute("CREATE EXTENSION IF NOT EXISTS vector")

    op.add_column(
        "rag_chunks",
        sa.Column(
            "embedding",
            Vector(),
            nullable=True,
        ),
    )


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_column("rag_chunks", "embedding")
