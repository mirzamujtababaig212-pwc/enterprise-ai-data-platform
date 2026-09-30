"""enforce agent run hierarchy invariants

Revision ID: 20260930020000
Revises: 20260930010000
Create Date: 2026-09-30
"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = "20260930020000"
down_revision: Union[str, Sequence[str], None] = "20260930010000"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_check_constraint(
        "ck_agent_runs_root_or_child",
        "agent_runs",
        sa.text(
            """
            (
                parent_run_id IS NULL
                AND parent_step_id IS NULL
                AND root_run_id = run_id
            )
            OR
            (
                parent_run_id IS NOT NULL
                AND parent_step_id IS NOT NULL
                AND root_run_id <> run_id
            )
            """
        ),
    )


def downgrade() -> None:
    op.drop_constraint(
        "ck_agent_runs_root_or_child",
        "agent_runs",
        type_="check",
    )
