"""create approval overrides

Revision ID: 20260927approval_override
Revises: 20260926approval
Create Date: 2026-09-27 01:00:00
"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op


revision: str = "20260927approval_override"
down_revision: Union[str, Sequence[str], None] = "20260926approval"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "approval_overrides",
        sa.Column(
            "override_id",
            sa.String(length=255),
            nullable=False,
        ),
        sa.Column(
            "approval_id",
            sa.String(length=255),
            nullable=False,
        ),
        sa.Column(
            "run_id",
            sa.String(length=36),
            nullable=False,
        ),
        sa.Column(
            "actor",
            sa.String(length=255),
            nullable=False,
        ),
        sa.Column(
            "reason",
            sa.String(),
            nullable=False,
        ),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.func.now(),
        ),
        sa.PrimaryKeyConstraint("override_id"),
        sa.ForeignKeyConstraint(
            ["approval_id"],
            ["approval_requests.approval_id"],
            ondelete="CASCADE",
        ),
        sa.ForeignKeyConstraint(
            ["run_id"],
            ["agent_runs.run_id"],
            ondelete="CASCADE",
        ),
        sa.UniqueConstraint(
            "approval_id",
            name="uq_approval_overrides_approval_id",
        ),
    )

    op.create_index(
        "ix_approval_overrides_approval_id",
        "approval_overrides",
        ["approval_id"],
        unique=False,
    )
    op.create_index(
        "ix_approval_overrides_run_id",
        "approval_overrides",
        ["run_id"],
        unique=False,
    )


def downgrade() -> None:
    op.drop_index(
        "ix_approval_overrides_run_id",
        table_name="approval_overrides",
    )
    op.drop_index(
        "ix_approval_overrides_approval_id",
        table_name="approval_overrides",
    )
    op.drop_table("approval_overrides")
