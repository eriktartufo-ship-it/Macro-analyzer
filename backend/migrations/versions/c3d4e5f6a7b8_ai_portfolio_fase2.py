"""AI Portfolio Fase 2: real PnL columns + learnings + reasoning tables.

Revision ID: c3d4e5f6a7b8
Revises: b2c3d4e5f6a7
Create Date: 2026-06-11
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


revision: str = "c3d4e5f6a7b8"
down_revision: Union[str, None] = "b2c3d4e5f6a7"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # Add price tracking columns
    op.add_column(
        "ai_portfolio_positions",
        sa.Column("avg_entry_price", sa.Float, nullable=True),
    )
    op.add_column(
        "ai_portfolio_positions",
        sa.Column("current_price", sa.Float, nullable=True),
    )

    # ai_portfolio_learnings
    op.create_table(
        "ai_portfolio_learnings",
        sa.Column("id", sa.Integer, primary_key=True),
        sa.Column("pattern_key", sa.String(128), nullable=False, unique=True),
        sa.Column("n_wins", sa.Integer, nullable=False, server_default="0"),
        sa.Column("n_losses", sa.Integer, nullable=False, server_default="0"),
        sa.Column("total_pnl_pct", sa.Float, nullable=False, server_default="0"),
        sa.Column("avg_pnl_pct", sa.Float, nullable=False, server_default="0"),
        sa.Column("win_rate", sa.Float, nullable=False, server_default="0.5"),
        sa.Column("entry_threshold_shift", sa.Float, nullable=False, server_default="0"),
        sa.Column("last_pattern_seen_at", sa.Date, nullable=False),
        sa.Column("insight_text", sa.Text, nullable=True),
        sa.Column("created_at", sa.DateTime, nullable=False, server_default=sa.func.now()),
        sa.Column("updated_at", sa.DateTime, nullable=False, server_default=sa.func.now()),
    )
    op.create_index("ix_ai_learnings_pattern", "ai_portfolio_learnings", ["pattern_key"])

    # ai_portfolio_reasoning
    op.create_table(
        "ai_portfolio_reasoning",
        sa.Column("id", sa.Integer, primary_key=True),
        sa.Column("date", sa.Date, nullable=False, unique=True),
        sa.Column("data_hash", sa.String(64), nullable=False),
        sa.Column("daily_summary", sa.Text, nullable=False),
        sa.Column("decisions_reasoning_json", sa.Text, nullable=False),
        sa.Column("regime_classifier_says", sa.String(20), nullable=False),
        sa.Column("ai_agrees", sa.String(16), nullable=False, server_default="high"),
        sa.Column("ai_alternative_regime", sa.String(20), nullable=True),
        sa.Column("regime_challenge_reasoning", sa.Text, nullable=True),
        sa.Column("provider", sa.String(48), nullable=False),
        sa.Column("created_at", sa.DateTime, nullable=False, server_default=sa.func.now()),
    )
    op.create_index("ix_ai_reasoning_date", "ai_portfolio_reasoning", ["date"])


def downgrade() -> None:
    op.drop_index("ix_ai_reasoning_date", table_name="ai_portfolio_reasoning")
    op.drop_table("ai_portfolio_reasoning")
    op.drop_index("ix_ai_learnings_pattern", table_name="ai_portfolio_learnings")
    op.drop_table("ai_portfolio_learnings")
    op.drop_column("ai_portfolio_positions", "current_price")
    op.drop_column("ai_portfolio_positions", "avg_entry_price")
