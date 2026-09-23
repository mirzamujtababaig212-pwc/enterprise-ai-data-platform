"""add tenant identity to agent runs

Revision ID: 7c4e91b2d6a8
Revises: 3aa74a32a967
"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa

revision: str = "7c4e91b2d6a8"
down_revision: Union[str, Sequence[str], None] = "3aa74a32a967"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column(
        "agent_runs",
        sa.Column(
            "tenant_id",
            sa.String(length=255),
            nullable=True,
        ),
    )
    op.create_index(
        "ix_agent_runs_tenant_id",
        "agent_runs",
        ["tenant_id"],
        unique=False,
    )


def downgrade() -> None:
    op.drop_index(
        "ix_agent_runs_tenant_id",
        table_name="agent_runs",
    )
    op.drop_column(
        "agent_runs",
        "tenant_id",
    )
