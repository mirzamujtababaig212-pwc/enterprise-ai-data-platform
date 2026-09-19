"""add agent run leases

Revision ID: db2ca1d37aec
Revises: 4c20be996647
Create Date: 2026-09-19 15:57:50.718553

"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa

# revision identifiers, used by Alembic.
revision: str = "db2ca1d37aec"
down_revision: Union[str, Sequence[str], None] = "4c20be996647"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    op.add_column(
        "agent_runs",
        sa.Column(
            "lease_id",
            sa.String(length=36),
            nullable=True,
        ),
    )

    op.add_column(
        "agent_runs",
        sa.Column(
            "lease_expires_at",
            sa.DateTime(timezone=True),
            nullable=True,
        ),
    )

    op.create_index(
        "ix_agent_runs_lease_id",
        "agent_runs",
        ["lease_id"],
        unique=False,
    )

    op.create_index(
        "ix_agent_runs_lease_expires_at",
        "agent_runs",
        ["lease_expires_at"],
        unique=False,
    )


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_index(
        "ix_agent_runs_lease_expires_at",
        table_name="agent_runs",
    )

    op.drop_index(
        "ix_agent_runs_lease_id",
        table_name="agent_runs",
    )

    op.drop_column(
        "agent_runs",
        "lease_expires_at",
    )

    op.drop_column(
        "agent_runs",
        "lease_id",
    )
