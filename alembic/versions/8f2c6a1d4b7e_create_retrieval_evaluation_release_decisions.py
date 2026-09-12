"""create retrieval evaluation release decisions

Revision ID: 8f2c6a1d4b7e
Revises: d3a7f5c1e842
Create Date: 2026-09-12 00:00:00.000000
"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql


revision: str = "8f2c6a1d4b7e"
down_revision: Union[str, Sequence[str], None] = "d3a7f5c1e842"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    op.create_table(
        "retrieval_evaluation_release_decisions",
        sa.Column(
            "run_id",
            sa.String(length=255),
            nullable=False,
        ),
        sa.Column(
            "passed",
            sa.Boolean(),
            nullable=False,
        ),
        sa.Column(
            "errors",
            sa.JSON().with_variant(
                postgresql.JSONB(astext_type=sa.Text()),
                "postgresql",
            ),
            nullable=False,
        ),
        sa.Column(
            "native",
            sa.JSON().with_variant(
                postgresql.JSONB(astext_type=sa.Text()),
                "postgresql",
            ),
            nullable=False,
        ),
        sa.Column(
            "external",
            sa.JSON().with_variant(
                postgresql.JSONB(astext_type=sa.Text()),
                "postgresql",
            ),
            nullable=False,
        ),
        sa.ForeignKeyConstraint(
            ["run_id"],
            ["retrieval_evaluation_runs.run_id"],
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("run_id"),
    )

    op.create_index(
        "ix_retrieval_evaluation_release_decisions_passed",
        "retrieval_evaluation_release_decisions",
        ["passed"],
        unique=False,
    )


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_index(
        "ix_retrieval_evaluation_release_decisions_passed",
        table_name="retrieval_evaluation_release_decisions",
    )
    op.drop_table("retrieval_evaluation_release_decisions")
