"""Reset AI portfolio con scope per strategia (2026-07-15).

Il wipe mirato serve a rimuovere i dati orfani di un bot dismesso (data_driven)
SENZA toccare il track record delle strategie vive. Un bug qui distrugge dati
reali → test esplicito che l'isolamento regga.
"""
from __future__ import annotations

from datetime import date

import pytest

from app.models.ai_portfolio import AiPortfolioPosition


def _pos(db, asset: str, strategy: str):
    p = AiPortfolioPosition(
        asset_class=asset,
        strategy_type=strategy,
        target_weight_pct=0.10,
        current_weight_pct=0.05,
        tranches_filled=1,
        tranches_total=2,
        opened_at=date(2026, 7, 1),
        entry_regime="stagflation",
        avg_entry_score=60.0,
        estimated_pnl_pct=0.0,
        took_partial_tp=False,
        last_action="OPEN_TRANCHE",
        last_action_date=date(2026, 7, 1),
    )
    db.add(p)
    return p


@pytest.fixture
def seeded():
    from sqlalchemy import create_engine
    from sqlalchemy.orm import sessionmaker
    from sqlalchemy.pool import StaticPool

    from app.database import Base
    from app.models.ai_portfolio import (
        AiPortfolioDecision,
        AiPortfolioLearning,
        AiPortfolioPerformance,
        AiPortfolioReasoning,
    )

    engine = create_engine(
        "sqlite://", connect_args={"check_same_thread": False}, poolclass=StaticPool,
    )
    Base.metadata.create_all(bind=engine, tables=[
        AiPortfolioPosition.__table__, AiPortfolioDecision.__table__,
        AiPortfolioPerformance.__table__, AiPortfolioReasoning.__table__,
        AiPortfolioLearning.__table__,
    ])
    db = sessionmaker(bind=engine)()

    _pos(db, "gold", "data_driven")
    _pos(db, "energy", "data_driven")
    _pos(db, "gold", "model_driven")
    _pos(db, "silver", "llm_driven")
    db.commit()
    yield db
    db.close()


def _reset(db, strategy=None):
    from app.api.routes import reset_ai_portfolio
    return reset_ai_portfolio(confirm="YES_WIPE_ALL", strategy=strategy, db=db)


def test_scoped_reset_removes_only_target_strategy(seeded):
    db = seeded
    out = _reset(db, strategy="data_driven")
    assert out["scope"] == "data_driven"
    assert out["deleted"]["ai_portfolio_positions"] == 2

    remaining = db.query(AiPortfolioPosition).all()
    kinds = sorted(p.strategy_type for p in remaining)
    assert kinds == ["llm_driven", "model_driven"]  # le vive sopravvivono


def test_scoped_reset_is_noop_for_unknown_strategy(seeded):
    db = seeded
    out = _reset(db, strategy="does_not_exist")
    assert out["deleted"]["ai_portfolio_positions"] == 0
    assert db.query(AiPortfolioPosition).count() == 4


def test_unscoped_reset_still_wipes_everything(seeded):
    db = seeded
    out = _reset(db, strategy=None)
    assert out["scope"] == "ALL"
    assert out["deleted"]["ai_portfolio_positions"] == 4
    assert db.query(AiPortfolioPosition).count() == 0


def test_reset_requires_confirm_token(seeded):
    from fastapi import HTTPException
    from app.api.routes import reset_ai_portfolio
    with pytest.raises(HTTPException):
        reset_ai_portfolio(confirm="", strategy="data_driven", db=seeded)
    assert seeded.query(AiPortfolioPosition).count() == 4  # nulla cancellato
