"""Add opaque claim ownership token to voice turns."""

from collections.abc import Sequence

import sqlalchemy as sa

from alembic import op

revision: str = "20261020010000"
down_revision: str | Sequence[str] | None = "20261018030000"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.add_column(
        "voice_call_turns",
        sa.Column("claim_token", sa.String(length=36), nullable=True),
    )


def downgrade() -> None:
    op.drop_column("voice_call_turns", "claim_token")
