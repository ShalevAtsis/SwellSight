"""Initial platform schema

Revision ID: 001
Revises:
Create Date: 2026-05-21
"""

from alembic import op
import sqlalchemy as sa

revision = "001"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "users",
        sa.Column("id", sa.String(36), primary_key=True),
        sa.Column("email", sa.String(255), nullable=False),
        sa.Column("password_hash", sa.String(255), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
    )
    op.create_index("ix_users_email", "users", ["email"], unique=True)

    op.create_table(
        "analyses",
        sa.Column("id", sa.String(36), primary_key=True),
        sa.Column("user_id", sa.String(36), sa.ForeignKey("users.id"), nullable=False),
        sa.Column("status", sa.String(32), nullable=False),
        sa.Column("storage_key", sa.String(512), nullable=True),
        sa.Column("result_json", sa.JSON(), nullable=True),
        sa.Column("surf_score", sa.Float(), nullable=True),
        sa.Column("score_breakdown", sa.JSON(), nullable=True),
        sa.Column("model_version", sa.String(64), nullable=True),
        sa.Column("error_message", sa.Text(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.Column("completed_at", sa.DateTime(timezone=True), nullable=True),
    )
    op.create_index("ix_analyses_user_id", "analyses", ["user_id"])
    op.create_index("ix_analyses_status", "analyses", ["status"])

    op.create_table(
        "model_versions",
        sa.Column("id", sa.String(64), primary_key=True),
        sa.Column("checkpoint_uri", sa.String(512), nullable=False),
        sa.Column("metrics_json", sa.JSON(), nullable=True),
        sa.Column("promoted_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.Column("is_active", sa.Boolean(), default=False),
    )


def downgrade() -> None:
    op.drop_table("model_versions")
    op.drop_table("analyses")
    op.drop_table("users")
