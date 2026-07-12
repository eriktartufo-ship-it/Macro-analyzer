"""Test Portfolio Tracking (pt_*) — logica pura, auth, API, isolamento flag.

Nessuna rete: prezzi mockati, LLM mai chiamato.
"""
from __future__ import annotations

from datetime import date

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.services.portfolio_tracking import auth as pt_auth
from app.services.portfolio_tracking.advisor import validate_advice
from app.services.portfolio_tracking.positions import (
    GRAMS_PER_TROY_OZ,
    build_portfolio,
    compute_position,
    to_canonical,
)


# ---------- Unità e costo medio (logica pura) ----------

def test_to_canonical_grams_to_oz_preserves_value():
    qty_oz, price_oz = to_canonical(31.1034768, 100.0, "g")  # 31.1g a 100 €/g
    assert qty_oz == pytest.approx(1.0)
    assert price_oz == pytest.approx(3110.34768)
    # controvalore invariato
    assert qty_oz * price_oz == pytest.approx(31.1034768 * 100.0)


def test_to_canonical_other_units_passthrough():
    assert to_canonical(2.0, 50.0, "oz") == (2.0, 50.0)
    assert to_canonical(0.5, 90000.0, "btc") == (0.5, 90000.0)


def _tx(side, qty, price, unit="oz", fee=0.0, day=1, **extra):
    return {
        "side": side, "quantity": qty, "unit_price_eur": price, "unit": unit,
        "fee_eur": fee, "trade_date": date(2026, 1, day), "id": day,
        "asset_key": "gold", "asset_class": "metal", "label": "Oro",
        "ticker": "GC=F", **extra,
    }


def test_average_cost_two_buys():
    pos = compute_position([_tx("buy", 1, 2000, day=1), _tx("buy", 1, 2400, day=2)])
    assert pos["quantity"] == pytest.approx(2.0)
    assert pos["avg_cost_eur"] == pytest.approx(2200.0)
    assert pos["cost_basis_eur"] == pytest.approx(4400.0)
    assert pos["realized_pnl_eur"] == pytest.approx(0.0)


def test_average_cost_sell_realizes_pnl_and_keeps_avg():
    pos = compute_position([
        _tx("buy", 2, 2000, day=1),
        _tx("sell", 1, 2500, day=2),
    ])
    assert pos["quantity"] == pytest.approx(1.0)
    assert pos["avg_cost_eur"] == pytest.approx(2000.0)  # il PMC non cambia con le sell
    assert pos["realized_pnl_eur"] == pytest.approx(500.0)


def test_fees_increase_cost_basis_and_reduce_realized():
    pos = compute_position([
        _tx("buy", 1, 2000, fee=10, day=1),
        _tx("sell", 1, 2500, fee=10, day=2),
    ])
    assert pos["quantity"] == pytest.approx(0.0)
    assert pos["realized_pnl_eur"] == pytest.approx(500 - 10 - 10)


def test_sell_clamped_to_owned_quantity():
    pos = compute_position([_tx("buy", 1, 2000, day=1), _tx("sell", 5, 2500, day=2)])
    assert pos["quantity"] == pytest.approx(0.0)
    assert pos["realized_pnl_eur"] == pytest.approx(500.0)  # solo 1 oz venduta


def test_mixed_units_grams_and_oz():
    # 31.1034768 g a 65 €/g + 1 oz a 2100 €/oz → 2 oz totali
    pos = compute_position([
        _tx("buy", GRAMS_PER_TROY_OZ, 65.0, unit="g", day=1),
        _tx("buy", 1, 2100, unit="oz", day=2),
    ])
    assert pos["quantity"] == pytest.approx(2.0)
    expected_cost = GRAMS_PER_TROY_OZ * 65.0 + 2100
    assert pos["cost_basis_eur"] == pytest.approx(expected_cost)


def test_build_portfolio_totals_and_allocation():
    txs = [
        _tx("buy", 1, 2000, day=1),
        _tx("buy", 0.05, 80000, unit="btc", day=2,
            asset_key="bitcoin", asset_class="crypto", label="Bitcoin", ticker="BTC-EUR"),
    ]
    result = build_portfolio(txs, {"gold": 2500.0, "bitcoin": 90000.0})
    totals = result["totals"]
    assert totals["invested_eur"] == pytest.approx(2000 + 4000)
    assert totals["market_value_eur"] == pytest.approx(2500 + 4500)
    assert totals["pnl_eur"] == pytest.approx(1000.0)
    allocs = {p["asset_key"]: p["allocation_pct"] for p in result["positions"]}
    assert allocs["bitcoin"] == pytest.approx(4500 / 7000 * 100, abs=0.01)
    assert sum(a for a in allocs.values()) == pytest.approx(100.0, abs=0.05)


def test_build_portfolio_missing_spot_degrades_gracefully():
    result = build_portfolio([_tx("buy", 1, 2000, day=1)], {"gold": None})
    p = result["positions"][0]
    assert p["market_value_eur"] is None
    assert p["pnl_pct"] is None
    assert result["totals"]["market_value_eur"] is None  # nessun valore inventato


# ---------- Auth ----------

def test_parse_users_formats():
    users = pt_auth.parse_users("erik:Erik:pw1, marco:pw2,malformed")
    assert set(users) == {"erik", "marco"}
    assert users["erik"].display_name == "Erik"
    assert users["marco"].display_name == "Marco"


def test_token_roundtrip_and_tamper():
    token = pt_auth.mint_token("erik", secret="s3cret")
    assert pt_auth.verify_token(token, secret="s3cret") == "erik"
    assert pt_auth.verify_token(token + "x", secret="s3cret") is None
    assert pt_auth.verify_token(token, secret="other") is None


def test_token_expiry():
    import time

    old = time.time() - pt_auth.TOKEN_TTL_SECONDS - 10
    token = pt_auth.mint_token("erik", secret="s", now=old)
    assert pt_auth.verify_token(token, secret="s") is None


def test_check_login(monkeypatch):
    from app.config import settings

    monkeypatch.setattr(settings, "pt_users", "erik:Erik:topsecret")
    assert pt_auth.check_login("Erik", "topsecret").username == "erik"
    assert pt_auth.check_login("erik", "wrong") is None
    assert pt_auth.check_login("ghost", "topsecret") is None


# ---------- Advisor: validazione output LLM ----------

def test_validate_advice_clamps_and_defaults():
    out = validate_advice({
        "macro_summary": "x" * 900,
        "assets": {
            "gold": {"light": "GREEN", "headline": "ok", "reasons": ["a"] * 10},
            "silver": {"light": "viola", "headline": "h"},
            "junk": {"light": "green"},
        },
    })
    assert out["assets"]["gold"]["light"] == "green"
    assert len(out["assets"]["gold"]["reasons"]) == 4
    assert out["assets"]["silver"]["light"] == "yellow"  # colore ignoto → neutro
    assert "junk" not in out["assets"]
    assert len(out["macro_summary"]) == 500


def test_validate_advice_rejects_empty():
    assert validate_advice({}) is None
    assert validate_advice({"assets": {}}) is None
    assert validate_advice("not a dict") is None


# ---------- API (sqlite in-memory, prezzi mockati) ----------

@pytest.fixture
def pt_client(monkeypatch):
    from sqlalchemy import create_engine
    from sqlalchemy.orm import sessionmaker
    from sqlalchemy.pool import StaticPool

    from app.api.portfolio_tracking import router as pt_router
    from app.config import settings
    from app.database import Base, get_db
    from app.models import PtAdvice, PtPriceCache, PtStrategy, PtTransaction

    engine = create_engine(
        "sqlite://", connect_args={"check_same_thread": False}, poolclass=StaticPool,
    )
    Base.metadata.create_all(bind=engine, tables=[
        PtTransaction.__table__, PtPriceCache.__table__,
        PtAdvice.__table__, PtStrategy.__table__,
    ])
    TestSession = sessionmaker(bind=engine)

    monkeypatch.setattr(settings, "pt_users", "erik:Erik:pw-erik,marco:Marco:pw-marco")
    monkeypatch.setattr(settings, "pt_secret", "test-secret")
    monkeypatch.setattr(settings, "pt_admin_token", "admin-token")

    # niente rete: spot fissi
    from app.services.portfolio_tracking import prices

    monkeypatch.setattr(
        prices, "spot_eur_for_assets",
        lambda db, kt: {k: {"gold": 2500.0, "bitcoin": 90000.0}.get(k) for k in kt},
    )

    app = FastAPI()
    app.include_router(pt_router)

    def override_get_db():
        db = TestSession()
        try:
            yield db
        finally:
            db.close()

    app.dependency_overrides[get_db] = override_get_db
    return TestClient(app)


def _login(client, username="erik", password="pw-erik"):
    resp = client.post("/api/v1/pt/login", json={"username": username, "password": password})
    assert resp.status_code == 200
    return {"Authorization": f"Bearer {resp.json()['token']}"}


def test_login_and_me(pt_client):
    headers = _login(pt_client)
    me = pt_client.get("/api/v1/pt/me", headers=headers)
    assert me.status_code == 200
    assert me.json() == {"username": "erik", "display_name": "Erik"}
    assert pt_client.post(
        "/api/v1/pt/login", json={"username": "erik", "password": "no"},
    ).status_code == 401
    assert pt_client.get("/api/v1/pt/me").status_code == 401


def test_transactions_crud_and_portfolio(pt_client):
    headers = _login(pt_client)
    resp = pt_client.post("/api/v1/pt/transactions", headers=headers, json={
        "asset_key": "gold", "side": "buy", "quantity": 1, "unit": "oz",
        "unit_price_eur": 2000, "trade_date": "2026-01-10",
    })
    assert resp.status_code == 200
    tx_id = resp.json()["id"]
    assert resp.json()["ticker"] == "GC=F"

    pt_client.post("/api/v1/pt/transactions", headers=headers, json={
        "asset_key": "bitcoin", "side": "buy", "quantity": 0.05, "unit": "btc",
        "unit_price_eur": 80000, "trade_date": "2026-02-01",
    })

    pf = pt_client.get("/api/v1/pt/portfolio", headers=headers).json()
    assert pf["totals"]["market_value_eur"] == pytest.approx(2500 + 4500)
    assert pf["totals"]["pnl_eur"] == pytest.approx(1000.0)

    # unit sbagliata per un metallo → 422
    bad = pt_client.post("/api/v1/pt/transactions", headers=headers, json={
        "asset_key": "gold", "side": "buy", "quantity": 1, "unit": "btc",
        "unit_price_eur": 2000, "trade_date": "2026-01-10",
    })
    assert bad.status_code == 422

    assert pt_client.delete(
        f"/api/v1/pt/transactions/{tx_id}", headers=headers,
    ).status_code == 200
    txs = pt_client.get("/api/v1/pt/transactions", headers=headers).json()["transactions"]
    assert len(txs) == 1


def test_users_are_isolated(pt_client):
    erik = _login(pt_client)
    marco = _login(pt_client, "marco", "pw-marco")
    resp = pt_client.post("/api/v1/pt/transactions", headers=erik, json={
        "asset_key": "gold", "side": "buy", "quantity": 1, "unit": "oz",
        "unit_price_eur": 2000, "trade_date": "2026-01-10",
    })
    tx_id = resp.json()["id"]
    # marco non vede né cancella i movimenti di erik
    assert pt_client.get(
        "/api/v1/pt/transactions", headers=marco,
    ).json()["transactions"] == []
    assert pt_client.delete(
        f"/api/v1/pt/transactions/{tx_id}", headers=marco,
    ).status_code == 404


def test_strategy_admin_push_and_read(pt_client):
    admin = {"Authorization": "Bearer admin-token"}
    resp = pt_client.post(
        "/api/v1/pt/strategy/erik", headers=admin,
        json={"content": {"targets": {"VUAG": 30}, "note": "test"}},
    )
    assert resp.status_code == 200
    headers = _login(pt_client)
    strategy = pt_client.get("/api/v1/pt/strategy", headers=headers).json()
    assert strategy["strategy"]["targets"]["VUAG"] == 30
    # senza admin token → 403
    assert pt_client.post(
        "/api/v1/pt/strategy/erik",
        headers={"Authorization": "Bearer wrong"},
        json={"content": {}},
    ).status_code == 403


def test_bulk_import_admin(pt_client):
    admin = {"Authorization": "Bearer admin-token"}
    resp = pt_client.post("/api/v1/pt/transactions/bulk", headers=admin, json=[
        {"username": "erik", "asset_key": "VUAG.L", "label": "S&P 500 (VUAG)",
         "side": "buy", "quantity": 70, "unit": "share",
         "unit_price_eur": 100, "trade_date": "2026-07-05"},
    ])
    assert resp.status_code == 200
    assert resp.json()["created"] == 1
    headers = _login(pt_client)
    txs = pt_client.get("/api/v1/pt/transactions", headers=headers).json()["transactions"]
    assert txs[0]["asset_key"] == "VUAG.L"
    assert txs[0]["asset_class"] == "ticker"


# ---------- Isolamento flag ----------

def test_pt_routes_only_when_flag_enabled():
    """Con ENABLE_PORTFOLIO_TRACKING=false (default) l'app NON espone /pt."""
    from app.config import settings
    from app.main import app

    has_pt = any(getattr(r, "path", "").startswith("/api/v1/pt") for r in app.routes)
    assert has_pt == settings.enable_portfolio_tracking
