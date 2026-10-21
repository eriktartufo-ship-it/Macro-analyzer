"""Client Groq unico (2026-10-21).

Il difetto: `llama-3.3-70b-versatile` scritto a mano in tre file; Groq l'ha ritirato (404
`model_not_found`) e per oltre un mese il sito non ha valutato una notizia, senza un rosso.
"""
from __future__ import annotations

import pathlib
import re

import pytest

from app.services.llm import groq_client

APP = pathlib.Path(__file__).resolve().parents[1] / "app"


def test_censimento_url_e_modello_groq_vivono_solo_nel_client():
    sparsi = []
    for f in APP.rglob("*.py"):
        if f.name == "groq_client.py":
            continue
        t = f.read_text(encoding="utf-8")
        if "api.groq.com" in t or re.search(r"[\"']llama-\d", t) or re.search(r"[\"']openai/gpt-oss", t):
            sparsi.append(str(f.relative_to(APP)))
    assert sparsi == []


def test_il_censimento_vede_davvero_i_file():
    # controllo positivo: senza file da scorrere il censimento passerebbe a vuoto
    assert len(list(APP.rglob("*.py"))) > 50
    assert "api.groq.com" in (APP / "services" / "llm" / "groq_client.py").read_text(encoding="utf-8")


def test_modello_di_default_e_override(monkeypatch):
    monkeypatch.setattr(groq_client.settings, "groq_model", "")
    assert groq_client.get_model() == "openai/gpt-oss-120b"
    monkeypatch.setattr(groq_client.settings, "groq_model", "qwen/qwen3.8-27b")
    assert groq_client.get_model() == "qwen/qwen3.8-27b"


def test_i_modelli_che_ragionano_hanno_lo_sforzo_basso():
    b = groq_client.request_body("s", "u", 100, model="openai/gpt-oss-120b")
    assert b["reasoning_effort"] == "low" and b["response_format"] == {"type": "json_object"}
    assert "reasoning_effort" not in groq_client.request_body("s", "u", 100, model="qwen/qwen3.8-27b")


class _Resp:
    def __init__(self, status, text, payload=None):
        self.status_code, self.text, self._p = status, text, payload

    def json(self):
        return self._p

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")


def test_modello_ritirato_dice_cosa_fare_e_non_crasha(monkeypatch):
    msgs = []
    monkeypatch.setattr(groq_client.settings, "groq_api_key", "x")
    monkeypatch.setattr(groq_client.logger, "error", lambda m: msgs.append(m))
    monkeypatch.setattr(groq_client.requests, "post", lambda *a, **k: _Resp(
        404, '{"error":{"code":"model_not_found"}}'))
    assert groq_client.chat_json("s", "u", 10) is None
    assert msgs and "ritirato" in msgs[0] and "GROQ_MODEL" in msgs[0]


def test_risposta_buona_ritorna_il_contenuto(monkeypatch):
    monkeypatch.setattr(groq_client.settings, "groq_api_key", "x")
    monkeypatch.setattr(groq_client.requests, "post", lambda *a, **k: _Resp(
        200, "", {"choices": [{"message": {"content": '{"ok": 1}'}}]}))
    assert groq_client.chat_json("s", "u", 10) == '{"ok": 1}'


def test_lo_scorer_delle_notizie_passa_dal_client(monkeypatch):
    from app.services.news import scorer

    monkeypatch.setattr(scorer.settings, "groq_api_key", "x")
    monkeypatch.setattr(groq_client, "chat_json", lambda *a, **k: (
        '{"results": [{"headline": "Fed holds", "sentiment": -0.2, "relevance": 0.8, "summary": "s"}]}'))
    out = scorer.score_news_batch([{"title": "Fed holds", "source": "x", "url": "u", "date": "d"}])
    assert len(out) == 1 and out[0]["relevance"] == pytest.approx(0.8)


def test_stato_delle_notizie_dice_da_quando_sono_ferme(monkeypatch):
    from datetime import date, timedelta

    from sqlalchemy import create_engine
    from sqlalchemy.orm import sessionmaker
    from sqlalchemy.pool import StaticPool

    from app.api.routes import get_news_status
    from app.database import Base
    from app.models.news_signals import NewsSignal

    engine = create_engine("sqlite://", connect_args={"check_same_thread": False}, poolclass=StaticPool)
    Base.metadata.create_all(bind=engine, tables=[NewsSignal.__table__])
    db = sessionmaker(bind=engine)()
    monkeypatch.setattr(groq_client.settings, "groq_api_key", "x")
    monkeypatch.setattr(groq_client.settings, "groq_model", "")
    vuoto = get_news_status(db=db)
    assert vuoto["latest_date"] is None and vuoto["model"] == "openai/gpt-oss-120b"
    db.add(NewsSignal(date=date.today() - timedelta(days=34), source="x", title="t", content_summary="s",
                      sentiment_score=0.1, relevance_score=0.5, affected_assets="{}"))
    db.commit()
    st = get_news_status(db=db)
    assert st["days_since"] == 34 and st["groq_configured"] is True
