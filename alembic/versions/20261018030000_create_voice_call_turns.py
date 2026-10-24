"""Create durable voice call turns.

Revision ID: 20261018030000
Revises: 20261018020000
Create Date: 2026-10-19
"""

from collections.abc import Sequence

import sqlalchemy as sa

from alembic import op

revision: str = "20261018030000"
down_revision: str | Sequence[str] | None = "20261018020000"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "voice_call_turns",
        sa.Column("turn_id", sa.String(length=36), nullable=False),
        sa.Column("tenant_id", sa.String(length=255), nullable=False),
        sa.Column("call_id", sa.String(length=36), nullable=False),
        sa.Column("turn_number", sa.Integer(), nullable=False),
        sa.Column("callback_token", sa.String(length=128), nullable=False),
        sa.Column(
            "status",
            sa.String(length=24),
            server_default="awaiting_input",
            nullable=False,
        ),
        sa.Column("request_fingerprint", sa.String(length=64), nullable=True),
        sa.Column("agent_run_id", sa.String(length=36), nullable=True),
        sa.Column("response_text", sa.Text(), nullable=True),
        sa.Column("failure_code", sa.String(length=100), nullable=True),
        sa.Column("started_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("completed_at", sa.DateTime(timezone=True), nullable=True),
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
        sa.CheckConstraint(
            "turn_number >= 1",
            name="ck_voice_call_turns_positive_turn_number",
        ),
        sa.CheckConstraint(
            "status IN ('awaiting_input', 'processing', 'completed', " "'failed', 'no_input')",
            name="ck_voice_call_turns_status",
        ),
        sa.CheckConstraint(
            "request_fingerprint IS NULL OR length(request_fingerprint) = 64",
            name="ck_voice_call_turns_fingerprint_length",
        ),
        sa.CheckConstraint(
            "response_text IS NULL OR length(response_text) <= 8000",
            name="ck_voice_call_turns_response_length",
        ),
        sa.ForeignKeyConstraint(
            ["call_id"],
            ["voice_calls.call_id"],
            ondelete="RESTRICT",
        ),
        sa.PrimaryKeyConstraint("turn_id"),
        sa.UniqueConstraint(
            "tenant_id",
            "call_id",
            "turn_number",
            name="uq_voice_call_turns_tenant_call_turn",
        ),
        sa.UniqueConstraint(
            "callback_token",
            name="uq_voice_call_turns_callback_token",
        ),
    )

    op.create_index(
        "ix_voice_call_turns_tenant_status",
        "voice_call_turns",
        ["tenant_id", "status"],
        unique=False,
    )
    op.create_index(
        "ix_voice_call_turns_call_turn_number",
        "voice_call_turns",
        ["call_id", "turn_number"],
        unique=False,
    )


def downgrade() -> None:
    op.drop_index(
        "ix_voice_call_turns_call_turn_number",
        table_name="voice_call_turns",
    )
    op.drop_index(
        "ix_voice_call_turns_tenant_status",
        table_name="voice_call_turns",
    )
    op.drop_table("voice_call_turns")
