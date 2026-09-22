"""add agent run idempotency key

Revision ID: e5b7c2d91a46
Revises: d39f49619987
Create Date: 2026-09-22
"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa

revision: str = "e5b7c2d91a46"
down_revision: Union[str, Sequence[str], None] = "d39f49619987"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column(
        "agent_runs",
        sa.Column(
            "idempotency_key",
            sa.String(length=255),
            nullable=True,
        ),
    )

    op.create_index(
        "ix_agent_runs_idempotency_key",
        "agent_runs",
        ["idempotency_key"],
    )

    op.create_unique_constraint(
        "uq_agent_runs_user_id_idempotency_key",
        "agent_runs",
        ["user_id", "idempotency_key"],
    )


def downgrade() -> None:
    op.drop_constraint(
        "uq_agent_runs_user_id_idempotency_key",
        "agent_runs",
        type_="unique",
    )

    op.drop_index(
        "ix_agent_runs_idempotency_key",
        table_name="agent_runs",
    )

    op.drop_column(
        "agent_runs",
        "idempotency_key",
    )
