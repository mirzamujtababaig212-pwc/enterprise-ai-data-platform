"""create retrieval evaluation runs

Revision ID: 65be8ea0448a
Revises: 37d9ce8bfb9e
Create Date: 2026-09-11
"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql


revision: str = "65be8ea0448a"
down_revision: Union[str, Sequence[str], None] = "37d9ce8bfb9e"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "retrieval_evaluation_runs",
        sa.Column("run_id", sa.String(length=255), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
        ),
        sa.Column("dataset_name", sa.String(length=255), nullable=False),
        sa.Column("dataset_version", sa.String(length=255), nullable=False),
        sa.Column("release_passed", sa.Boolean(), nullable=False),
        sa.Column(
            "lineage",
            sa.JSON().with_variant(postgresql.JSONB(), "postgresql"),
            nullable=False,
        ),
        sa.Column(
            "evaluation",
            sa.JSON().with_variant(postgresql.JSONB(), "postgresql"),
            nullable=False,
        ),
        sa.Column(
            "quality_gate",
            sa.JSON().with_variant(postgresql.JSONB(), "postgresql"),
            nullable=False,
        ),
        sa.Column(
            "regression",
            sa.JSON().with_variant(postgresql.JSONB(), "postgresql"),
            nullable=True,
        ),
        sa.PrimaryKeyConstraint("run_id"),
    )

    op.create_index(
        "ix_retrieval_evaluation_runs_created_at",
        "retrieval_evaluation_runs",
        ["created_at"],
    )
    op.create_index(
        "ix_retrieval_evaluation_runs_dataset_name",
        "retrieval_evaluation_runs",
        ["dataset_name"],
    )
    op.create_index(
        "ix_retrieval_evaluation_runs_dataset_version",
        "retrieval_evaluation_runs",
        ["dataset_version"],
    )
    op.create_index(
        "ix_retrieval_evaluation_runs_release_passed",
        "retrieval_evaluation_runs",
        ["release_passed"],
    )


def downgrade() -> None:
    op.drop_index(
        "ix_retrieval_evaluation_runs_release_passed",
        table_name="retrieval_evaluation_runs",
    )
    op.drop_index(
        "ix_retrieval_evaluation_runs_dataset_version",
        table_name="retrieval_evaluation_runs",
    )
    op.drop_index(
        "ix_retrieval_evaluation_runs_dataset_name",
        table_name="retrieval_evaluation_runs",
    )
    op.drop_index(
        "ix_retrieval_evaluation_runs_created_at",
        table_name="retrieval_evaluation_runs",
    )
    op.drop_table("retrieval_evaluation_runs")
