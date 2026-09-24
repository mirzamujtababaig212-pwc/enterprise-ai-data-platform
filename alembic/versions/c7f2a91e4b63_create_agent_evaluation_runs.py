"""create agent evaluation runs

Revision ID: c7f2a91e4b63
Revises: b4e7c91d2f63
Create Date: 2026-09-24
"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

revision: str = "c7f2a91e4b63"
down_revision: Union[str, Sequence[str], None] = "b4e7c91d2f63"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "agent_evaluation_runs",
        sa.Column(
            "evaluation_run_id",
            sa.String(length=255),
            nullable=False,
        ),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
        ),
        sa.Column(
            "evaluated_run_id",
            sa.String(length=36),
            nullable=False,
        ),
        sa.Column(
            "agent_name",
            sa.String(length=255),
            nullable=False,
        ),
        sa.Column(
            "agent_version",
            sa.String(length=255),
            nullable=True,
        ),
        sa.Column(
            "tenant_id",
            sa.String(length=255),
            nullable=True,
        ),
        sa.Column(
            "passed",
            sa.Boolean(),
            nullable=False,
        ),
        sa.Column(
            "lineage",
            sa.JSON().with_variant(
                postgresql.JSONB(),
                "postgresql",
            ),
            nullable=False,
        ),
        sa.Column(
            "metrics",
            sa.JSON().with_variant(
                postgresql.JSONB(),
                "postgresql",
            ),
            nullable=False,
        ),
        sa.Column(
            "policy",
            sa.JSON().with_variant(
                postgresql.JSONB(),
                "postgresql",
            ),
            nullable=False,
        ),
        sa.Column(
            "quality_gate",
            sa.JSON().with_variant(
                postgresql.JSONB(),
                "postgresql",
            ),
            nullable=False,
        ),
        sa.ForeignKeyConstraint(
            ["evaluated_run_id"],
            ["agent_runs.run_id"],
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("evaluation_run_id"),
    )

    op.create_index(
        "ix_agent_evaluation_runs_created_at",
        "agent_evaluation_runs",
        ["created_at"],
    )

    op.create_index(
        "ix_agent_evaluation_runs_evaluated_run_id",
        "agent_evaluation_runs",
        ["evaluated_run_id"],
    )

    op.create_index(
        "ix_agent_evaluation_runs_agent_name",
        "agent_evaluation_runs",
        ["agent_name"],
    )

    op.create_index(
        "ix_agent_evaluation_runs_tenant_id",
        "agent_evaluation_runs",
        ["tenant_id"],
    )

    op.create_index(
        "ix_agent_evaluation_runs_passed",
        "agent_evaluation_runs",
        ["passed"],
    )


def downgrade() -> None:
    op.drop_index(
        "ix_agent_evaluation_runs_passed",
        table_name="agent_evaluation_runs",
    )

    op.drop_index(
        "ix_agent_evaluation_runs_tenant_id",
        table_name="agent_evaluation_runs",
    )

    op.drop_index(
        "ix_agent_evaluation_runs_agent_name",
        table_name="agent_evaluation_runs",
    )

    op.drop_index(
        "ix_agent_evaluation_runs_evaluated_run_id",
        table_name="agent_evaluation_runs",
    )

    op.drop_index(
        "ix_agent_evaluation_runs_created_at",
        table_name="agent_evaluation_runs",
    )

    op.drop_table("agent_evaluation_runs")
