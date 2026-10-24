"""Add opaque claim ownership token to voice turns."""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


revision: str = "20261020010000"
down_revision: Union[str, Sequence[str], None] = "20261018030000"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column(
        "voice_call_turns",
        sa.Column("claim_token", sa.String(length=36), nullable=True),
    )


def downgrade() -> None:
    op.drop_column("voice_call_turns", "claim_token")
