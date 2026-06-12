"""AI Portfolio horserace: add strategy_type to all tables.

User 2026-06-12: 2 strategie competono — model_driven (classifier) vs
data_driven (raw indicators rule-based). Capitale separato $100k cadauno.

Revision ID: d4e5f6a7b8c9
Revises: c3d4e5f6a7b8
Create Date: 2026-06-12
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


revision: str = "d4e5f6a7b8c9"
down_revision: Union[str, None] = "c3d4e5f6a7b8"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def _add_strategy_col(table_name: str) -> None:
    op.add_column(
        table_name,
        sa.Column(
            "strategy_type", sa.String(16),
            nullable=False, server_default="model_driven",
        ),
    )
    op.create_index(
        f"ix_{table_name.replace('ai_portfolio_', 'ai_')}_strategy",
        table_name, ["strategy_type"],
    )


def upgrade() -> None:
    # 1. ai_portfolio_positions: rimuovi unique su asset_class, aggiungi unique composito
    op.drop_index("ix_ai_positions_asset", table_name="ai_portfolio_positions")
    op.execute("ALTER TABLE ai_portfolio_positions DROP CONSTRAINT IF EXISTS ai_portfolio_positions_asset_class_key")
    _add_strategy_col("ai_portfolio_positions")
    op.create_index("ix_ai_positions_asset_strategy", "ai_portfolio_positions", ["asset_class", "strategy_type"])
    op.create_unique_constraint(
        "uq_position_asset_strategy", "ai_portfolio_positions",
        ["asset_class", "strategy_type"],
    )

    # 2. ai_portfolio_decisions: aggiungi colonna
    _add_strategy_col("ai_portfolio_decisions")

    # 3. ai_portfolio_performance: rimuovi unique su date, aggiungi composito
    op.drop_index("ix_ai_perf_date", table_name="ai_portfolio_performance")
    op.execute("ALTER TABLE ai_portfolio_performance DROP CONSTRAINT IF EXISTS ai_portfolio_performance_date_key")
    _add_strategy_col("ai_portfolio_performance")
    op.create_index("ix_ai_perf_date_strategy", "ai_portfolio_performance", ["date", "strategy_type"])
    op.create_unique_constraint(
        "uq_perf_date_strategy", "ai_portfolio_performance",
        ["date", "strategy_type"],
    )

    # 4. ai_portfolio_learnings: rimuovi unique pattern, aggiungi composito
    op.drop_index("ix_ai_learnings_pattern", table_name="ai_portfolio_learnings")
    op.execute("ALTER TABLE ai_portfolio_learnings DROP CONSTRAINT IF EXISTS ai_portfolio_learnings_pattern_key_key")
    _add_strategy_col("ai_portfolio_learnings")
    op.create_index("ix_ai_learnings_pattern_strategy", "ai_portfolio_learnings", ["pattern_key", "strategy_type"])
    op.create_unique_constraint(
        "uq_learning_pattern_strategy", "ai_portfolio_learnings",
        ["pattern_key", "strategy_type"],
    )

    # 5. ai_portfolio_reasoning: rimuovi unique date, aggiungi composito
    op.drop_index("ix_ai_reasoning_date", table_name="ai_portfolio_reasoning")
    op.execute("ALTER TABLE ai_portfolio_reasoning DROP CONSTRAINT IF EXISTS ai_portfolio_reasoning_date_key")
    _add_strategy_col("ai_portfolio_reasoning")
    op.create_index("ix_ai_reasoning_date_strategy", "ai_portfolio_reasoning", ["date", "strategy_type"])
    op.create_unique_constraint(
        "uq_reasoning_date_strategy", "ai_portfolio_reasoning",
        ["date", "strategy_type"],
    )


def downgrade() -> None:
    for table in [
        "ai_portfolio_reasoning",
        "ai_portfolio_learnings",
        "ai_portfolio_performance",
        "ai_portfolio_decisions",
        "ai_portfolio_positions",
    ]:
        op.execute(f"ALTER TABLE {table} DROP CONSTRAINT IF EXISTS uq_position_asset_strategy")
        op.execute(f"ALTER TABLE {table} DROP CONSTRAINT IF EXISTS uq_perf_date_strategy")
        op.execute(f"ALTER TABLE {table} DROP CONSTRAINT IF EXISTS uq_learning_pattern_strategy")
        op.execute(f"ALTER TABLE {table} DROP CONSTRAINT IF EXISTS uq_reasoning_date_strategy")
        op.drop_column(table, "strategy_type")
