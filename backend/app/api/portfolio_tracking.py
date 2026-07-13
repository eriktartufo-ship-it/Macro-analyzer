"""API Portfolio Tracking — /api/v1/pt/* (dietro ENABLE_PORTFOLIO_TRACKING).

Router separato dal monolite routes.py (modulo additivo). Tutte le rotte dati
richiedono il Bearer token utente; strategy-push e import bulk richiedono
PT_ADMIN_TOKEN (canale Claude Code).
"""
from __future__ import annotations

import json
from datetime import date as date_type

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from app.database import get_db
from app.models import PtStrategy, PtTransaction
from app.services.portfolio_tracking import advisor, prices
from app.services.portfolio_tracking.auth import (
    PtUser,
    check_login,
    mint_token,
    require_pt_admin,
    require_pt_user,
)
from app.services.portfolio_tracking.positions import (
    METAL_PURITIES,
    PRESET_ASSETS,
    build_portfolio,
)

router = APIRouter(prefix="/api/v1/pt", tags=["portfolio-tracking"])


# ---------- Schemi ----------

class LoginRequest(BaseModel):
    username: str
    password: str


class TransactionIn(BaseModel):
    asset_key: str = Field(..., max_length=32)
    side: str = Field("buy", pattern="^(buy|sell)$")
    quantity: float = Field(..., gt=0)
    unit: str = Field(..., max_length=8)
    unit_price_eur: float = Field(..., ge=0)
    fee_eur: float = Field(0.0, ge=0)
    trade_date: date_type
    note: str | None = Field(None, max_length=500)
    # solo per asset_key custom (ticker libero)
    label: str | None = Field(None, max_length=64)
    # purezza metallo (frazione 0<p<=1); ignorata per non-metalli
    purity: float = Field(1.0, gt=0, le=1)


class BulkTransactionIn(TransactionIn):
    username: str


class StrategyIn(BaseModel):
    content: dict


def _resolve_asset(tx: TransactionIn) -> dict:
    """Preset (gold/silver/bitcoin) o ticker Yahoo libero (asset_class=ticker)."""
    key = tx.asset_key.strip()
    preset = PRESET_ASSETS.get(key.lower())
    if preset:
        if tx.unit not in preset["units"]:
            raise HTTPException(
                status_code=422,
                detail=f"Unità '{tx.unit}' non valida per {preset['label']} "
                       f"(ammesse: {', '.join(preset['units'])})",
            )
        return {
            "asset_key": key.lower(),
            "asset_class": preset["asset_class"],
            "label": preset["label"],
            "ticker": preset["ticker"],
        }
    ticker = key.upper()
    return {
        "asset_key": ticker,
        "asset_class": "ticker",
        "label": (tx.label or ticker).strip(),
        "ticker": ticker,
    }


def _purity_for(asset: dict, requested: float) -> float:
    """La purezza vale solo per i metalli; per il resto è sempre 1.0."""
    return requested if asset["asset_class"] == "metal" else 1.0


def _tx_dict(t: PtTransaction) -> dict:
    return {
        "id": t.id,
        "asset_key": t.asset_key,
        "asset_class": t.asset_class,
        "label": t.label,
        "ticker": t.ticker,
        "side": t.side,
        "quantity": t.quantity,
        "unit": t.unit,
        "unit_price_eur": t.unit_price_eur,
        "fee_eur": t.fee_eur,
        "purity": t.purity,
        "trade_date": t.trade_date.isoformat(),
        "note": t.note,
    }


# ---------- Auth ----------

@router.post("/login")
def login(body: LoginRequest):
    from app.config import settings

    if not settings.pt_secret:
        raise HTTPException(status_code=503, detail="PT_SECRET non configurato")
    user = check_login(body.username, body.password)
    if user is None:
        raise HTTPException(status_code=401, detail="Credenziali non valide")
    return {
        "token": mint_token(user.username),
        "username": user.username,
        "display_name": user.display_name,
    }


@router.get("/me")
def me(user: PtUser = Depends(require_pt_user)):
    return {"username": user.username, "display_name": user.display_name}


@router.post("/refresh-token")
def refresh_token(user: PtUser = Depends(require_pt_user)):
    """Sessione rolling: il frontend lo chiama all'avvio, il token si rinnova."""
    return {"token": mint_token(user.username)}


# ---------- Portafoglio ----------

@router.get("/assets")
def list_assets():
    return {
        "presets": [
            {"asset_key": k, **{f: v[f] for f in ("label", "asset_class", "units", "canonical_unit")}}
            for k, v in PRESET_ASSETS.items()
        ],
        "purities": METAL_PURITIES,
    }


@router.get("/portfolio")
def get_portfolio(user: PtUser = Depends(require_pt_user), db: Session = Depends(get_db)):
    txs = (
        db.query(PtTransaction)
        .filter(PtTransaction.username == user.username)
        .all()
    )
    tx_dicts = [
        {**_tx_dict(t), "trade_date": t.trade_date}  # date object per il sort
        for t in txs
    ]
    key_ticker = {t.asset_key: t.ticker for t in txs}
    spots = prices.spot_eur_for_assets(db, key_ticker)
    result = build_portfolio(tx_dicts, spots)
    result["username"] = user.username
    result["display_name"] = user.display_name
    return result


@router.get("/transactions")
def list_transactions(
    limit: int = 100,
    user: PtUser = Depends(require_pt_user),
    db: Session = Depends(get_db),
):
    txs = (
        db.query(PtTransaction)
        .filter(PtTransaction.username == user.username)
        .order_by(PtTransaction.trade_date.desc(), PtTransaction.id.desc())
        .limit(min(limit, 500))
        .all()
    )
    return {"transactions": [_tx_dict(t) for t in txs]}


@router.post("/transactions")
def add_transaction(
    body: TransactionIn,
    user: PtUser = Depends(require_pt_user),
    db: Session = Depends(get_db),
):
    asset = _resolve_asset(body)
    row = PtTransaction(
        username=user.username,
        side=body.side,
        quantity=body.quantity,
        unit=body.unit,
        unit_price_eur=body.unit_price_eur,
        fee_eur=body.fee_eur,
        purity=_purity_for(asset, body.purity),
        trade_date=body.trade_date,
        note=body.note,
        **asset,
    )
    db.add(row)
    db.commit()
    db.refresh(row)
    return _tx_dict(row)


@router.put("/transactions/{tx_id}")
def update_transaction(
    tx_id: int,
    body: TransactionIn,
    user: PtUser = Depends(require_pt_user),
    db: Session = Depends(get_db),
):
    row = (
        db.query(PtTransaction)
        .filter(PtTransaction.id == tx_id, PtTransaction.username == user.username)
        .first()
    )
    if row is None:
        raise HTTPException(status_code=404, detail="Movimento non trovato")
    asset = _resolve_asset(body)
    row.side = body.side
    row.quantity = body.quantity
    row.unit = body.unit
    row.unit_price_eur = body.unit_price_eur
    row.fee_eur = body.fee_eur
    row.purity = _purity_for(asset, body.purity)
    row.trade_date = body.trade_date
    row.note = body.note
    row.asset_key = asset["asset_key"]
    row.asset_class = asset["asset_class"]
    row.label = asset["label"]
    row.ticker = asset["ticker"]
    db.commit()
    db.refresh(row)
    return _tx_dict(row)


@router.delete("/transactions/{tx_id}")
def delete_transaction(
    tx_id: int,
    user: PtUser = Depends(require_pt_user),
    db: Session = Depends(get_db),
):
    row = (
        db.query(PtTransaction)
        .filter(PtTransaction.id == tx_id, PtTransaction.username == user.username)
        .first()
    )
    if row is None:
        raise HTTPException(status_code=404, detail="Movimento non trovato")
    db.delete(row)
    db.commit()
    return {"deleted": tx_id}


# ---------- Consigli macro ----------

@router.get("/advice/latest")
def get_latest_advice(user: PtUser = Depends(require_pt_user), db: Session = Depends(get_db)):
    return {"advice": advisor.latest_advice(db)}


@router.post("/advice/refresh")
def refresh_advice(user: PtUser = Depends(require_pt_user), db: Session = Depends(get_db)):
    if not advisor.can_manual_refresh(db):
        raise HTTPException(
            status_code=429,
            detail="Analisi già aggiornata di recente, riprova tra un'ora",
        )
    row = advisor.generate_advice(db, trigger="manual")
    if row is None:
        raise HTTPException(
            status_code=503,
            detail="Analisi non disponibile ora (LLM o dati macro non raggiungibili)",
        )
    return {"advice": advisor.latest_advice(db)}


# ---------- Strategia (vetrina) ----------

@router.get("/strategy")
def get_strategy(user: PtUser = Depends(require_pt_user), db: Session = Depends(get_db)):
    row = db.query(PtStrategy).filter(PtStrategy.username == user.username).first()
    if row is None:
        return {"strategy": None}
    return {
        "strategy": json.loads(row.content_json),
        "updated_at": row.updated_at.isoformat(),
    }


# ---------- Canale admin (Claude Code) ----------

@router.post("/strategy/{username}", dependencies=[Depends(require_pt_admin)])
def put_strategy(username: str, body: StrategyIn, db: Session = Depends(get_db)):
    username = username.strip().lower()
    row = db.query(PtStrategy).filter(PtStrategy.username == username).first()
    payload = json.dumps(body.content, ensure_ascii=False)
    if row is None:
        row = PtStrategy(username=username, content_json=payload)
        db.add(row)
    else:
        row.content_json = payload
    db.commit()
    return {"username": username, "ok": True}


@router.post("/transactions/bulk", dependencies=[Depends(require_pt_admin)])
def bulk_import(body: list[BulkTransactionIn], db: Session = Depends(get_db)):
    created = 0
    for item in body:
        asset = _resolve_asset(item)
        db.add(PtTransaction(
            username=item.username.strip().lower(),
            side=item.side,
            quantity=item.quantity,
            unit=item.unit,
            unit_price_eur=item.unit_price_eur,
            fee_eur=item.fee_eur,
            purity=_purity_for(asset, item.purity),
            trade_date=item.trade_date,
            note=item.note,
            **asset,
        ))
        created += 1
    db.commit()
    return {"created": created}
