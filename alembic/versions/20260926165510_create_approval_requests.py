"""create approval requests

Revision ID: 20260926approval
Revises: c7f2a91e4b63
Create Date: 2026-09-26
"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql


revision: str = "20260926approval"
down_revision: Union[str, Sequence[str], None] = "c7f2a91e4b63"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "approval_requests",
        sa.Column("approval_id", sa.String(length=255), nullable=False),
        sa.Column(
            "run_id",
            sa.String(length=36),
            nullable=False,
        ),
        sa.Column("step_id", sa.String(length=255), nullable=False),
        sa.Column("call_id", sa.String(length=255), nullable=False),
        sa.Column("tool_name", sa.String(length=255), nullable=False),
        sa.Column(
            "idempotency_key",
            sa.String(length=255),
            nullable=False,
        ),
        sa.Column("status", sa.String(length=50), nullable=False),
        sa.Column("policy_name", sa.String(length=255), nullable=False),
        sa.Column(
            "policy_version",
            sa.String(length=100),
            nullable=False,
        ),
        sa.Column("risk_tier", sa.String(length=50), nullable=False),
        sa.Column("requested_action", sa.String(), nullable=False),
        sa.Column(
            "policy_metadata",
            sa.JSON().with_variant(postgresql.JSONB(), "postgresql"),
            nullable=False,
        ),
        sa.Column("resolved_by", sa.String(length=255), nullable=True),
        sa.Column("resolution_reason", sa.String(), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.func.now(),
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.func.now(),
        ),
        sa.Column(
            "resolved_at",
            sa.DateTime(timezone=True),
            nullable=True,
        ),
        sa.ForeignKeyConstraint(
            ["run_id"],
            ["agent_runs.run_id"],
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("approval_id"),
        sa.UniqueConstraint(
            "run_id",
            "step_id",
            "call_id",
            name="uq_approval_requests_run_step_call",
        ),
    )

    op.create_index(
        "ix_approval_requests_run_id",
        "approval_requests",
        ["run_id"],
    )
    op.create_index(
        "ix_approval_requests_status",
        "approval_requests",
        ["status"],
    )
    op.create_index(
        "ix_approval_requests_run_step",
        "approval_requests",
        ["run_id", "step_id"],
    )


def downgrade() -> None:
    op.drop_index(
        "ix_approval_requests_run_step",
        table_name="approval_requests",
    )
    op.drop_index(
        "ix_approval_requests_status",
        table_name="approval_requests",
    )
    op.drop_index(
        "ix_approval_requests_run_id",
        table_name="approval_requests",
    )
    op.drop_table("approval_requests")
