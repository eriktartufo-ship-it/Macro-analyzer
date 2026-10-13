"""Accesso admin (S51, 2026-10-13): ogni endpoint che CAMBIA qualcosa vuole il token.

Fino al 12/10 chiunque poteva, da internet: cambiare i flag del modello, cambiare chiave e modello
Gemini (/llm-settings), azzerare il portafoglio AI, lanciare refresh e backfill pesanti, rigenerare
analisi LLM a pagamento. Le letture (GET) restano pubbliche: il sito si guarda senza login.

Il censimento non elenca gli endpoint a mano: li scorre tutti. Un endpoint nuovo che scrive e non
e' protetto fa fallire il test, senza che qualcuno debba ricordarsene.
"""
import pytest
from fastapi.routing import APIRoute
from fastapi.testclient import TestClient

from app.api import routes
from app.api import portfolio_tracking
from app.config import settings
from app.main import app

SCRIVONO = {"POST", "PUT", "DELETE", "PATCH"}
GUARDIE = {"require_admin", "require_pt_user", "require_pt_admin"}
# Pubblici per progetto, ognuno con la sua ragione.
PUBBLICI = {
    "/api/v1/pt/login": "ingresso del portfolio tracking: verifica la password",
    "/api/v1/pt/refresh-token": "rinnova una sessione PT: verifica il token di sessione",
}
TOKEN = "token-di-prova-s51"


def _guardie(dependant) -> set[str]:
    nomi = set()
    for d in dependant.dependencies:
        nomi.add(getattr(d.call, "__name__", ""))
        nomi |= _guardie(d)
    return nomi


def test_censimento_nessun_endpoint_che_scrive_e_scoperto():
    scoperti = []
    for router in (routes.router, portfolio_tracking.router):
        for r in router.routes:
            if not isinstance(r, APIRoute) or not (r.methods & SCRIVONO):
                continue
            if r.path in PUBBLICI:
                continue
            if not (_guardie(r.dependant) & GUARDIE):
                scoperti.append(f"{'/'.join(sorted(r.methods & SCRIVONO))} {r.path}")
    assert scoperti == [], f"endpoint che scrivono senza protezione: {scoperti}"


def test_il_censimento_vede_davvero_gli_endpoint():
    # controllo positivo: se il censimento non trovasse niente da controllare, passerebbe a vuoto
    scrivono = [r for r in routes.router.routes if isinstance(r, APIRoute) and r.methods & SCRIVONO]
    assert len(scrivono) >= 17


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setattr(settings, "macro_admin_token", TOKEN)
    import app.scheduler.jobs as jobs
    monkeypatch.setattr(jobs, "daily_refresh", lambda: None)
    return TestClient(app)


def _h(tok):
    return {"Authorization": f"Bearer {tok}"}


def test_senza_token_401(client):
    assert client.post("/api/v1/refresh").status_code == 401


def test_token_sbagliato_403(client):
    assert client.post("/api/v1/refresh", headers=_h("sbagliato")).status_code == 403


def test_token_giusto_passa(client):
    r = client.post("/api/v1/refresh", headers=_h(TOKEN))
    assert r.status_code == 200, r.text


def test_server_senza_token_configurato_rifiuta_tutto(monkeypatch):
    # fail-closed: un server senza token non diventa aperto
    monkeypatch.setattr(settings, "macro_admin_token", "")
    c = TestClient(app)
    assert c.post("/api/v1/refresh", headers=_h("qualunque")).status_code == 503


def test_i_flag_non_si_cambiano_senza_token(client):
    from app.services.config_flags import get_all_flags_state

    prima = get_all_flags_state()["USE_EU_STRESS_TRIGGER"]
    r = client.post("/api/v1/config/flags", json={"name": "USE_EU_STRESS_TRIGGER", "value": True})
    assert r.status_code == 401
    assert get_all_flags_state()["USE_EU_STRESS_TRIGGER"] == prima


def test_le_letture_restano_pubbliche(client):
    assert client.get("/api/v1/health").status_code == 200


@pytest.mark.parametrize("path", [
    "/api/v1/macro-llm-analysis?force_refresh=true",
    "/api/v1/ai-portfolio/reasoning?force_refresh=true",
    "/api/v1/fomc/report?force_refresh=true",
])
def test_rigenerare_un_analisi_llm_richiede_il_token(client, path):
    assert client.get(path).status_code == 403


def test_verifica_del_token_dal_sito(client):
    assert client.get("/api/v1/admin/check").status_code == 401
    assert client.get("/api/v1/admin/check", headers=_h("sbagliato")).status_code == 403
    assert client.get("/api/v1/admin/check", headers=_h(TOKEN)).json() == {"ok": True}
