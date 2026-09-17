from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

revision = "9b97f6f62cf7"
down_revision = "cff669198c0d"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "memory_items",
        sa.Column(
            "content_tsv",
            postgresql.TSVECTOR(),
            sa.Computed(
                "to_tsvector('simple'::regconfig, content)",
                persisted=True,
            ),
            nullable=True,
        ),
    )

    op.create_index(
        "ix_memory_items_content_tsv_gin",
        "memory_items",
        ["content_tsv"],
        unique=False,
        postgresql_using="gin",
    )


def downgrade() -> None:
    op.drop_index(
        "ix_memory_items_content_tsv_gin",
        table_name="memory_items",
    )
    op.drop_column("memory_items", "content_tsv")
