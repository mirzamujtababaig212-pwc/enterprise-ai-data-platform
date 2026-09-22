"""add authenticated principal to agent runs

Revision ID: f3a91c7e2d44
Revises: e5b7c2d91a46
"""

from alembic import op
import sqlalchemy as sa


revision = "f3a91c7e2d44"
down_revision = "e5b7c2d91a46"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "agent_runs",
        sa.Column("principal", sa.String(length=255), nullable=True),
    )
    op.create_index(
        "ix_agent_runs_principal",
        "agent_runs",
        ["principal"],
    )


def downgrade() -> None:
    op.drop_index("ix_agent_runs_principal", table_name="agent_runs")
    op.drop_column("agent_runs", "principal")
