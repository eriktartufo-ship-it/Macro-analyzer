"""Controllo incrociato AUTOMATICO (S51, 2026-10-13): dopo ogni refresh i numeri del sito si
confrontano con le fonti primarie, e l'esito si vede sul sito.

Il bug di S49 (CPI a 13 mesi, bilanci Fed/BCE su 5 anni) e' rimasto in produzione ~11 mesi perche'
il confronto esisteva solo come script da lanciare a mano. Una guardia che qualcuno deve ricordarsi
di lanciare, prima o poi non viene lanciata.
"""
import inspect

import pytest

from app.services.validation import source_crosscheck as sc


def _checks(valore_vero, errore=None):
    def fonte():
        if errore:
            raise errore
        return valore_vero
    return [("CPI a 12 mesi (SA)", "cpi_yoy", fonte, 0.05, "BLS")]


def test_dentro_tolleranza_ok():
    r = sc.run_crosscheck({"cpi_yoy": 3.35}, checks=_checks(3.36))
    assert r["n_ok"] == 1 and r["n_fail"] == 0
    assert r["results"][0]["status"] == "ok"


def test_fuori_tolleranza_fail_e_si_vede():
    # il caso vero di S49: il sito diceva 3,71, il BLS 3,35
    r = sc.run_crosscheck({"cpi_yoy": 3.71}, checks=_checks(3.35))
    assert r["n_fail"] == 1
    assert r["results"][0]["status"] == "fail"
    assert r["results"][0]["site"] == pytest.approx(3.71)
    assert r["results"][0]["source_value"] == pytest.approx(3.35)


def test_fonte_giu_e_non_giudicabile_non_un_errore():
    r = sc.run_crosscheck({"cpi_yoy": 3.35}, checks=_checks(None, RuntimeError("BLS in timeout")))
    assert r["n_na"] == 1 and r["n_fail"] == 0
    assert "BLS in timeout" in r["results"][0]["note"]


def test_valore_assente_sul_sito_non_giudicabile():
    r = sc.run_crosscheck({}, checks=_checks(3.35))
    assert r["results"][0]["status"] == "na"


def test_i_14_controlli_di_default_coprono_i_numeri_corretti_in_s49_e_s50():
    chiavi = {c[1] for c in sc.default_checks()}
    for k in ("cpi_yoy", "fed_balance_roc_12m", "ecb_balance_roc_12m", "foreign_treasury_roc_12m",
              "ea_hicp_yoy", "ecb_deposit_rate", "brent_spot", "ttf_price"):
        assert k in chiavi
    assert "ea_gas_storage_pct" in chiavi  # S52: stoccaggi AGSI
    assert len(chiavi) == 14


def test_salva_e_rilegge_con_storia(tmp_path, monkeypatch):
    monkeypatch.setattr(sc, "STORE", tmp_path / "x.json")
    for v in (3.71, 3.35):
        sc.save_result(sc.run_crosscheck({"cpi_yoy": v}, checks=_checks(3.35)))
    ultimo = sc.load_result()
    assert ultimo["n_ok"] == 1
    assert [h["n_fail"] for h in ultimo["history"]] == [1, 0]


def test_nessun_risultato_salvato():
    import pathlib
    assert sc.load_result(pathlib.Path("non/esiste.json")) is None


def test_il_refresh_giornaliero_lancia_il_controllo():
    # censimento del cablaggio: il refresh deve chiamare il controllo dopo aver salvato i risultati
    from app.scheduler import jobs

    src = inspect.getsource(jobs.daily_refresh)
    assert "_crosscheck_after_refresh(" in src
    assert src.index("_save_results(") < src.index("_crosscheck_after_refresh(")


def test_endpoint_qualita_dati(monkeypatch):
    from fastapi.testclient import TestClient
    from app.main import app

    monkeypatch.setattr(sc, "load_result", lambda path=None: None)
    assert TestClient(app).get("/api/v1/data-quality").json() == {"available": False}
    finto = {"checked_at": "2026-10-13T06:00:00", "n_ok": 13, "n_fail": 0, "n_na": 0, "results": [], "history": []}
    monkeypatch.setattr(sc, "load_result", lambda path=None: finto)
    d = TestClient(app).get("/api/v1/data-quality").json()
    assert d["available"] is True and d["n_ok"] == 13


def test_agsi_sceglie_il_giorno_piu_recente_per_data_non_per_posizione(monkeypatch):
    # righe in ordine CRESCENTE: un parser che prende la prima sbaglierebbe giorno
    risposta = {"data": [{"gasDayStart": "2026-10-16", "full": "74.10"},
                         {"gasDayStart": "2026-10-18", "full": "74.58"},
                         {"gasDayStart": "2026-10-17", "full": "74.35"}]}
    monkeypatch.setattr(sc, "_agsi_json", lambda key: risposta)
    assert sc.agsi_ultimo_riempimento("chiave-finta") == pytest.approx(74.58)


def test_agsi_senza_chiave_non_giudicabile(monkeypatch):
    from app.config import settings

    monkeypatch.setattr(settings, "agsi_api_key", "")
    check = [c for c in sc.default_checks() if c[1] == "ea_gas_storage_pct"]
    r = sc.run_crosscheck({"ea_gas_storage_pct": 74.58}, checks=check)
    assert r["results"][0]["status"] == "na"
    assert "chiave AGSI" in r["results"][0]["note"]


def test_agsi_chiave_rifiutata_non_giudicabile(monkeypatch):
    monkeypatch.setattr(sc, "_agsi_json", lambda key: {"error": "access denied", "message": "Invalid or missing API key"})
    with pytest.raises(RuntimeError):
        sc.agsi_ultimo_riempimento("sbagliata")
