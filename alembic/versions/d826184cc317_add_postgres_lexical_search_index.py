"""add postgres lexical search index

Revision ID: d826184cc317
Revises: 2c7e9a4b1d6f
Create Date: 2026-09-16 16:00:51.301014

"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa

# revision identifiers, used by Alembic.
revision: str = "d826184cc317"
down_revision: Union[str, Sequence[str], None] = "2c7e9a4b1d6f"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Add PostgreSQL full-text search support for RAG chunks."""
    op.add_column(
        "rag_chunks",
        sa.Column(
            "content_tsv",
            sa.dialects.postgresql.TSVECTOR(),
            sa.Computed(
                "to_tsvector('simple'::regconfig, content)",
                persisted=True,
            ),
            nullable=True,
        ),
    )

    op.create_index(
        "ix_rag_chunks_content_tsv_gin",
        "rag_chunks",
        ["content_tsv"],
        unique=False,
        postgresql_using="gin",
    )


def downgrade() -> None:
    """Remove PostgreSQL full-text search support for RAG chunks."""
    op.drop_index(
        "ix_rag_chunks_content_tsv_gin",
        table_name="rag_chunks",
    )
    op.drop_column("rag_chunks", "content_tsv")
