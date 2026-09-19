"""add user id to agent run events

Revision ID: 4c20be996647
Revises: c4e7a91b2f63
Create Date: 2026-09-19 14:19:41.779515

"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = "4c20be996647"
down_revision: Union[str, Sequence[str], None] = "c4e7a91b2f63"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    op.add_column(
        "agent_run_events",
        sa.Column(
            "user_id",
            sa.String(length=255),
            nullable=True,
        ),
    )

    op.create_index(
        "ix_agent_run_events_user_id",
        "agent_run_events",
        ["user_id"],
        unique=False,
    )


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_index(
        "ix_agent_run_events_user_id",
        table_name="agent_run_events",
    )

    op.drop_column(
        "agent_run_events",
        "user_id",
    )
