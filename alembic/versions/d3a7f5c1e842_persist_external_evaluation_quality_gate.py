"""persist external evaluation quality gate

Revision ID: d3a7f5c1e842
Revises: cb092000833d
Create Date: 2026-09-12 00:00:00.000000
"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

revision: str = "d3a7f5c1e842"
down_revision: Union[str, Sequence[str], None] = "cb092000833d"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    op.add_column(
        "retrieval_evaluation_runs",
        sa.Column(
            "external_quality_gate",
            sa.JSON().with_variant(
                postgresql.JSONB(astext_type=sa.Text()),
                "postgresql",
            ),
            nullable=True,
        ),
    )


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_column(
        "retrieval_evaluation_runs",
        "external_quality_gate",
    )
