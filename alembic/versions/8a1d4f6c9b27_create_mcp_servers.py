"""create tenant-owned MCP servers

Revision ID: 8a1d4f6c9b27
Revises: 7c4e91b2d6a8
Create Date: 2026-09-24
"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

revision: str = "8a1d4f6c9b27"
down_revision: Union[str, Sequence[str], None] = "7c4e91b2d6a8"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "mcp_servers",
        sa.Column(
            "server_id",
            sa.String(length=255),
            nullable=False,
        ),
        sa.Column(
            "tenant_id",
            sa.String(length=255),
            nullable=False,
        ),
        sa.Column(
            "name",
            sa.String(length=255),
            nullable=False,
        ),
        sa.Column(
            "transport",
            sa.String(length=50),
            nullable=False,
        ),
        sa.Column(
            "desired_state",
            sa.String(length=50),
            nullable=False,
        ),
        sa.Column(
            "configuration",
            sa.JSON().with_variant(postgresql.JSONB(), "postgresql"),
            nullable=False,
        ),
        sa.Column(
            "secret_references",
            sa.JSON().with_variant(postgresql.JSONB(), "postgresql"),
            nullable=False,
        ),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
        sa.PrimaryKeyConstraint("server_id"),
        sa.UniqueConstraint(
            "name",
            name="uq_mcp_servers_name",
        ),
    )

    op.create_index(
        "ix_mcp_servers_tenant_id",
        "mcp_servers",
        ["tenant_id"],
    )

    op.create_index(
        "ix_mcp_servers_desired_state",
        "mcp_servers",
        ["desired_state"],
    )


def downgrade() -> None:
    op.drop_index(
        "ix_mcp_servers_desired_state",
        table_name="mcp_servers",
    )
    op.drop_index(
        "ix_mcp_servers_tenant_id",
        table_name="mcp_servers",
    )
    op.drop_table("mcp_servers")
