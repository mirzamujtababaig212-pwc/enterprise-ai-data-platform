"""add_principal_to_checkpoints_and_events

Revision ID: 4030ddbb4224
Revises: f3a91c7e2d44
Create Date: 2026-09-22 11:29:24.845181

"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = "4030ddbb4224"
down_revision: Union[str, Sequence[str], None] = "f3a91c7e2d44"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column(
        "agent_run_checkpoints",
        sa.Column("principal", sa.String(length=255), nullable=True),
    )
    op.create_index(
        "ix_agent_run_checkpoints_principal",
        "agent_run_checkpoints",
        ["principal"],
        unique=False,
    )

    op.add_column(
        "agent_run_events",
        sa.Column("principal", sa.String(length=255), nullable=True),
    )
    op.create_index(
        "ix_agent_run_events_principal",
        "agent_run_events",
        ["principal"],
        unique=False,
    )


def downgrade() -> None:
    op.drop_index(
        "ix_agent_run_events_principal",
        table_name="agent_run_events",
    )
    op.drop_column("agent_run_events", "principal")

    op.drop_index(
        "ix_agent_run_checkpoints_principal",
        table_name="agent_run_checkpoints",
    )
    op.drop_column("agent_run_checkpoints", "principal")
