"""add durable agent run hierarchy

Revision ID: 20260930010000
Revises: 0fdf7764a738
Create Date: 2026-09-30
"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = "20260930010000"
down_revision: Union[str, Sequence[str], None] = "0fdf7764a738"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column(
        "agent_runs",
        sa.Column(
            "root_run_id",
            sa.String(length=36),
            nullable=True,
        ),
    )
    op.add_column(
        "agent_runs",
        sa.Column(
            "parent_run_id",
            sa.String(length=36),
            nullable=True,
        ),
    )
    op.add_column(
        "agent_runs",
        sa.Column(
            "parent_step_id",
            sa.String(length=255),
            nullable=True,
        ),
    )
    op.add_column(
        "agent_runs",
        sa.Column(
            "causation_id",
            sa.String(length=255),
            nullable=True,
        ),
    )

    # Existing runs are roots.
    op.execute(
        sa.text(
            """
            UPDATE agent_runs
            SET root_run_id = run_id
            WHERE root_run_id IS NULL
            """
        )
    )

    # Every persisted run must have a root.
    op.alter_column(
        "agent_runs",
        "root_run_id",
        existing_type=sa.String(length=36),
        nullable=False,
    )

    op.create_index(
        "ix_agent_runs_root_run_id",
        "agent_runs",
        ["root_run_id"],
    )
    op.create_index(
        "ix_agent_runs_parent_run_id",
        "agent_runs",
        ["parent_run_id"],
    )
    op.create_index(
        "ix_agent_runs_causation_id",
        "agent_runs",
        ["causation_id"],
    )

    # A root run points to itself. A child run points to another run.
    op.create_foreign_key(
        "fk_agent_runs_root_run_id",
        "agent_runs",
        "agent_runs",
        ["root_run_id"],
        ["run_id"],
        ondelete="RESTRICT",
    )

    op.create_foreign_key(
        "fk_agent_runs_parent_run_id",
        "agent_runs",
        "agent_runs",
        ["parent_run_id"],
        ["run_id"],
        ondelete="RESTRICT",
    )

    # agent_run_steps has a composite primary key:
    # (run_id, step_id). Preserve that structure rather than pretending
    # parent_step_id is globally unique.
    op.create_foreign_key(
        "fk_agent_runs_parent_step",
        "agent_runs",
        "agent_run_steps",
        ["parent_run_id", "parent_step_id"],
        ["run_id", "step_id"],
        ondelete="RESTRICT",
    )


def downgrade() -> None:
    op.drop_constraint(
        "fk_agent_runs_parent_step",
        "agent_runs",
        type_="foreignkey",
    )
    op.drop_constraint(
        "fk_agent_runs_parent_run_id",
        "agent_runs",
        type_="foreignkey",
    )
    op.drop_constraint(
        "fk_agent_runs_root_run_id",
        "agent_runs",
        type_="foreignkey",
    )

    op.drop_index(
        "ix_agent_runs_causation_id",
        table_name="agent_runs",
    )
    op.drop_index(
        "ix_agent_runs_parent_run_id",
        table_name="agent_runs",
    )
    op.drop_index(
        "ix_agent_runs_root_run_id",
        table_name="agent_runs",
    )

    op.drop_column("agent_runs", "causation_id")
    op.drop_column("agent_runs", "parent_step_id")
    op.drop_column("agent_runs", "parent_run_id")
    op.drop_column("agent_runs", "root_run_id")
