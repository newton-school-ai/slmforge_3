"""add build checkpoint columns.

Revision ID: a1b2c3d4e5f6
Revises: 95ec140c902c
Create Date: 2026-08-05 12:00:00.000000

"""

from collections.abc import Sequence

import sqlalchemy as sa

from alembic import op

revision: str = "a1b2c3d4e5f6"
down_revision: str | Sequence[str] | None = "95ec140c902c"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Add checkpoint tracking columns to builds."""
    with op.batch_alter_table("builds") as batch_op:
        batch_op.add_column(sa.Column("build_id", sa.String(), nullable=True))
        batch_op.add_column(sa.Column("checkpoint_path", sa.String(), nullable=True))
        batch_op.add_column(sa.Column("seed", sa.Integer(), nullable=True))
        batch_op.add_column(sa.Column("epochs_completed", sa.Integer(), nullable=True))
        batch_op.add_column(sa.Column("total_epochs", sa.Integer(), nullable=True))
        batch_op.add_column(sa.Column("status", sa.String(), nullable=True))
        batch_op.create_index("ix_builds_build_id", ["build_id"], unique=True)


def downgrade() -> None:
    """Remove checkpoint tracking columns from builds."""
    with op.batch_alter_table("builds") as batch_op:
        batch_op.drop_index("ix_builds_build_id")
        batch_op.drop_column("status")
        batch_op.drop_column("total_epochs")
        batch_op.drop_column("epochs_completed")
        batch_op.drop_column("seed")
        batch_op.drop_column("checkpoint_path")
        batch_op.drop_column("build_id")
