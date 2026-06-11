"""AI Portfolio: positions + decisions + performance tables.

User request 2026-06-10: paper trading autonomo daily con tranche DCA.

Revision ID: b2c3d4e5f6a7
Revises: a1b2c3d4e5f6
Create Date: 2026-06-10
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


revision: str = "b2c3d4e5f6a7"
down_revision: Union[str, None] = "a1b2c3d4e5f6"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # ai_portfolio_positions
    op.create_table(
        "ai_portfolio_positions",
        sa.Column("id", sa.Integer, primary_key=True),
        sa.Column("asset_class", sa.String(64), nullable=False, unique=True),
        sa.Column("target_weight_pct", sa.Float, nullable=False, server_default="0"),
        sa.Column("current_weight_pct", sa.Float, nullable=False, server_default="0"),
        sa.Column("tranches_filled", sa.Integer, nullable=False, server_default="0"),
        sa.Column("tranches_total", sa.Integer, nullable=False, server_default="3"),
        sa.Column("opened_at", sa.Date, nullable=False),
        sa.Column("entry_regime", sa.String(20), nullable=False),
        sa.Column("avg_entry_score", sa.Float, nullable=False, server_default="0"),
        sa.Column("estimated_pnl_pct", sa.Float, nullable=False, server_default="0"),
        sa.Column("took_partial_tp", sa.Boolean, nullable=False, server_default=sa.text("false")),
        sa.Column("last_action", sa.String(24), nullable=False, server_default="OPEN_TRANCHE"),
        sa.Column("last_action_date", sa.Date, nullable=False),
        sa.Column("created_at", sa.DateTime, nullable=False, server_default=sa.func.now()),
        sa.Column("updated_at", sa.DateTime, nullable=False, server_default=sa.func.now()),
    )
    op.create_index("ix_ai_positions_asset", "ai_portfolio_positions", ["asset_class"])

    # ai_portfolio_decisions
    op.create_table(
        "ai_portfolio_decisions",
        sa.Column("id", sa.Integer, primary_key=True),
        sa.Column("date", sa.Date, nullable=False),
        sa.Column("asset_class", sa.String(64), nullable=False),
        sa.Column("action", sa.String(24), nullable=False),
        sa.Column("size_pct", sa.Float, nullable=False, server_default="0"),
        sa.Column("tranche_index", sa.Integer, nullable=False, server_default="0"),
        sa.Column("tranches_total", sa.Integer, nullable=False, server_default="0"),
        sa.Column("score", sa.Float, nullable=False, server_default="0"),
        sa.Column("confidence", sa.Float, nullable=False, server_default="0"),
        sa.Column("regime", sa.String(20), nullable=False),
        sa.Column("momentum_signal", sa.String(16), nullable=False, server_default="NEUTRAL"),
        sa.Column("vol_60d", sa.Float, nullable=True),
        sa.Column("estimated_pnl_pct", sa.Float, nullable=True),
        sa.Column("reason_short", sa.Text, nullable=False),
        sa.Column("created_at", sa.DateTime, nullable=False, server_default=sa.func.now()),
    )
    op.create_index("ix_ai_decisions_date", "ai_portfolio_decisions", ["date"])
    op.create_index("ix_ai_decisions_asset", "ai_portfolio_decisions", ["asset_class"])
    op.create_index("ix_ai_decisions_date_asset", "ai_portfolio_decisions", ["date", "asset_class"])

    # ai_portfolio_performance
    op.create_table(
        "ai_portfolio_performance",
        sa.Column("id", sa.Integer, primary_key=True),
        sa.Column("date", sa.Date, nullable=False, unique=True),
        sa.Column("nav", sa.Float, nullable=False, server_default="100000"),
        sa.Column("cumulative_return_pct", sa.Float, nullable=False, server_default="0"),
        sa.Column("drawdown_pct", sa.Float, nullable=False, server_default="0"),
        sa.Column("peak_nav", sa.Float, nullable=False, server_default="100000"),
        sa.Column("n_positions", sa.Integer, nullable=False, server_default="0"),
        sa.Column("deployed_pct", sa.Float, nullable=False, server_default="0"),
        sa.Column("top_position_asset", sa.String(64), nullable=True),
        sa.Column("top_position_weight_pct", sa.Float, nullable=True),
        sa.Column("portfolio_vol_30d", sa.Float, nullable=True),
        sa.Column("benchmark_60_40_return_pct", sa.Float, nullable=True),
        sa.Column("benchmark_sp500_return_pct", sa.Float, nullable=True),
        sa.Column("created_at", sa.DateTime, nullable=False, server_default=sa.func.now()),
    )
    op.create_index("ix_ai_perf_date", "ai_portfolio_performance", ["date"])


def downgrade() -> None:
    op.drop_index("ix_ai_perf_date", table_name="ai_portfolio_performance")
    op.drop_table("ai_portfolio_performance")
    op.drop_index("ix_ai_decisions_date_asset", table_name="ai_portfolio_decisions")
    op.drop_index("ix_ai_decisions_asset", table_name="ai_portfolio_decisions")
    op.drop_index("ix_ai_decisions_date", table_name="ai_portfolio_decisions")
    op.drop_table("ai_portfolio_decisions")
    op.drop_index("ix_ai_positions_asset", table_name="ai_portfolio_positions")
    op.drop_table("ai_portfolio_positions")
