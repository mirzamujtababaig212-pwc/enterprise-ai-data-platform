"""add rag embedding provenance

Revision ID: 37d9ce8bfb9e
Revises: ec114008a881
Create Date: 2026-09-11

"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa

revision: str = "37d9ce8bfb9e"
down_revision: Union[str, Sequence[str], None] = "ec114008a881"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Add embedding request/resolution provenance to RAG chunks."""
    op.add_column(
        "rag_chunks",
        sa.Column(
            "embedding_requested_provider",
            sa.String(length=100),
            nullable=True,
        ),
    )
    op.add_column(
        "rag_chunks",
        sa.Column(
            "embedding_requested_model",
            sa.String(length=255),
            nullable=True,
        ),
    )
    op.add_column(
        "rag_chunks",
        sa.Column(
            "embedding_resolved_provider",
            sa.String(length=100),
            nullable=True,
        ),
    )
    op.add_column(
        "rag_chunks",
        sa.Column(
            "embedding_resolved_model",
            sa.String(length=255),
            nullable=True,
        ),
    )


def downgrade() -> None:
    """Remove RAG embedding request/resolution provenance."""
    op.drop_column("rag_chunks", "embedding_resolved_model")
    op.drop_column("rag_chunks", "embedding_resolved_provider")
    op.drop_column("rag_chunks", "embedding_requested_model")
    op.drop_column("rag_chunks", "embedding_requested_provider")
