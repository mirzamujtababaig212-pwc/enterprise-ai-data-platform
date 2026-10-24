"""Create tenant-scoped voice line configurations.

Revision ID: 20261018010000
Revises: 20261002010000
Create Date: 2026-10-18
"""

from collections.abc import Sequence

import sqlalchemy as sa

from alembic import op

revision: str = "20261018010000"
down_revision: str | Sequence[str] | None = "20261002010000"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "voice_line_configs",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("tenant_id", sa.String(length=255), nullable=False),
        sa.Column("configured_phone_number", sa.String(length=32), nullable=False),
        sa.Column("carrier_trunk_line", sa.String(length=255), nullable=False),
        sa.Column("call_forwarding_code", sa.String(length=128), nullable=True),
        sa.Column(
            "line_status",
            sa.String(length=16),
            server_default="paused",
            nullable=False,
        ),
        sa.Column("inbound_opening_greeting", sa.Text(), nullable=False),
        sa.Column("inbound_system_prompt_override", sa.Text(), nullable=True),
        sa.Column("outbound_opening_message", sa.Text(), nullable=False),
        sa.Column("outbound_system_prompt_override", sa.Text(), nullable=True),
        sa.Column("provider_key", sa.String(length=100), nullable=True),
        sa.Column("agent_uuid", sa.String(length=255), nullable=True),
        sa.Column(
            "provider_sync_status",
            sa.String(length=24),
            server_default="not_configured",
            nullable=False,
        ),
        sa.Column("last_provider_sync_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("last_provider_sync_error_code", sa.String(length=100), nullable=True),
        sa.Column(
            "voice_synthesizer",
            sa.String(length=255),
            server_default="polly-kajal",
            nullable=False,
        ),
        sa.Column(
            "spoken_language",
            sa.String(length=20),
            server_default="hi-IN",
            nullable=False,
        ),
        sa.Column(
            "conversational_pause_pacing",
            sa.Float(),
            server_default="1.0",
            nullable=False,
        ),
        sa.Column(
            "barge_in_sensitivity",
            sa.Integer(),
            server_default="2",
            nullable=False,
        ),
        sa.Column(
            "llm_reasoning_model",
            sa.String(length=255),
            server_default="openai/gpt-4o-mini",
            nullable=False,
        ),
        sa.Column(
            "max_call_duration_minutes",
            sa.Integer(),
            server_default="10",
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
        sa.CheckConstraint(
            "line_status IN ('active', 'paused', 'offline')",
            name="ck_voice_line_configs_line_status",
        ),
        sa.CheckConstraint(
            "provider_sync_status IN ('not_configured', 'pending', 'synced', 'error')",
            name="ck_voice_line_configs_provider_sync_status",
        ),
        sa.CheckConstraint(
            "conversational_pause_pacing >= 0 AND conversational_pause_pacing <= 10",
            name="ck_voice_line_configs_pause_pacing",
        ),
        sa.CheckConstraint(
            "barge_in_sensitivity BETWEEN 1 AND 10",
            name="ck_voice_line_configs_barge_in_sensitivity",
        ),
        sa.CheckConstraint(
            "max_call_duration_minutes BETWEEN 1 AND 240",
            name="ck_voice_line_configs_max_call_duration",
        ),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint(
            "configured_phone_number",
            name="uq_voice_line_configs_configured_phone_number",
        ),
    )
    op.create_index(
        "ix_voice_line_configs_tenant_updated_at",
        "voice_line_configs",
        ["tenant_id", "updated_at"],
        unique=False,
    )


def downgrade() -> None:
    op.drop_index(
        "ix_voice_line_configs_tenant_updated_at",
        table_name="voice_line_configs",
    )
    op.drop_table("voice_line_configs")
