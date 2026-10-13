"""Controllo incrociato: i numeri del sito contro le fonti primarie (S50 script, S51 automatico).

Per ogni indicatore: il valore che il sito mostra contro quello ricalcolato qui dalla fonte, con un
calcolo INDIPENDENTE dall'app (date esplicite, niente transforms.py).
    ok   entro la tolleranza   ·   fail   fuori   ·   na   non giudicabile (fonte giu' o dato assente)

Nato dal bug di S49: per ~11 mesi il CPI del sito e' stato una variazione a 13 mesi (3,71 contro 3,35
del BLS) e il bilancio Fed "-22% in 12 mesi, QT in corso" era la variazione da dicembre 2021, e nessuno
se n'e' accorto. Dal 13/10 gira da solo dopo ogni refresh giornaliero (jobs.daily_refresh) e l'esito
si legge in GET /data-quality e sul sito. Lo script backend/scripts/controllo_incrociato.py usa questo
stesso modulo contro il sito pubblicato.
Fonti senza chiave (BLS v2, Treasury.gov, BCE, Yahoo) + FRED con la chiave del progetto.
"""

from __future__ import annotations

import io
import json
import urllib.request
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Callable, Optional

import pandas as pd

# Stesso volume persistente della cache FRED (/app/.cache/fred in produzione): sopravvive ai deploy.
STORE = Path(__file__).resolve().parents[3] / ".cache" / "fred" / "_crosscheck_latest.json"
HISTORY_LEN = 30

Check = tuple[str, str, Callable[[], Optional[float]], float, str]


def _get(url: str, data: bytes | None = None, timeout: int = 60) -> bytes:
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json",
                                                          "User-Agent": "macro-controllo-incrociato"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read()


def yoy_per_data(s: pd.Series, months: int = 12) -> Optional[float]:
    """Ultimo valore contro quello ESATTAMENTE `months` mesi prima (stessa data), altrimenti None."""
    s = s.dropna().sort_index()
    base = s.get(s.index[-1] - pd.DateOffset(months=months))
    return None if base is None or pd.isna(base) else (s.iloc[-1] / base - 1) * 100


def ultimo_entro(s: pd.Series, months: int, giorni: int = 7) -> Optional[float]:
    """Variazione % dell'ultimo valore contro l'ultimo osservato entro `giorni` prima di `months` mesi fa."""
    s = s.dropna().sort_index()
    t = s.index[-1] - pd.DateOffset(months=months)
    prima = s[s.index <= t]
    if prima.empty or (t - prima.index[-1]).days > giorni:
        return None
    return (s.iloc[-1] / prima.iloc[-1] - 1) * 100


def bls(serie: list[str]) -> dict[str, pd.Series]:
    y = date.today().year
    body = json.dumps({"seriesid": serie, "startyear": str(y - 2), "endyear": str(y)}).encode()
    d = json.loads(_get("https://api.bls.gov/publicAPI/v2/timeseries/data/", body))
    out = {}
    for s in d["Results"]["series"]:
        v = {pd.Timestamp(int(x["year"]), int(x["period"][1:]), 1): float(x["value"])
             for x in s["data"] if x["period"].startswith("M") and x["period"] != "M13" and x["value"] not in ("-", "")}
        out[s["seriesID"]] = pd.Series(v).sort_index()
    return out


def treasury(tipo: str) -> pd.DataFrame:
    y = date.today().year
    url = ("https://home.treasury.gov/resource-center/data-chart-center/interest-rates/daily-treasury-rates.csv/"
           f"{y}/all?type={tipo}&field_tdr_date_value={y}&page&_format=csv")
    df = pd.read_csv(io.StringIO(_get(url).decode("utf-8")))
    df["Date"] = pd.to_datetime(df["Date"])
    return df.sort_values("Date").set_index("Date")


def bce(chiave: str) -> pd.Series:
    """Serie del Data Portal BCE letta qui, senza passare dal modulo europe_energy dell'app."""
    df = pd.read_csv(io.StringIO(_get(f"https://data-api.ecb.europa.eu/service/data/{chiave}"
                                      "?format=csvdata&detail=dataonly&lastNObservations=5").decode("utf-8")))
    df = df.dropna(subset=["OBS_VALUE"])
    return pd.Series(df["OBS_VALUE"].astype(float).to_numpy(), index=pd.to_datetime(df["TIME_PERIOD"].astype(str)))


def yahoo_ultimo(ticker: str) -> float:
    import yfinance as yf
    return float(yf.Ticker(ticker).history(period="10d")["Close"].dropna().iloc[-1])


def fred(sid: str) -> pd.Series:
    from app.config import settings
    from fredapi import Fred
    return Fred(api_key=settings.fred_api_key).get_series(sid, observation_start="2019-01-01").dropna()


def default_checks() -> list[Check]:
    """I 13 controlli: (nome, chiave sul sito, valore vero dalla fonte, tolleranza, fonte)."""
    cache: dict = {}

    def b():
        if "bls" not in cache:
            cache["bls"] = bls(["CUSR0000SA0", "LNS14000000"])
        return cache["bls"]

    return [
        ("CPI a 12 mesi (SA)", "cpi_yoy", lambda: yoy_per_data(b()["CUSR0000SA0"]), 0.05, "BLS CUSR0000SA0"),
        ("Disoccupazione", "unrate", lambda: float(b()["LNS14000000"].iloc[-1]), 0.05, "BLS LNS14000000"),
        ("Curva 10a-2a", "yield_curve_10y2y",
         lambda: float((lambda t: t["10 Yr"].iloc[-1] - t["2 Yr"].iloc[-1])(treasury("daily_treasury_yield_curve"))),
         0.12, "Treasury.gov"),
        ("Reale 10a", "real_yield_10y",
         lambda: float(treasury("daily_treasury_real_yield_curve")["10 YR"].iloc[-1]), 0.12, "Treasury.gov TIPS"),
        ("Bilancio Fed 12m", "fed_balance_roc_12m", lambda: ultimo_entro(fred("WALCL"), 12), 0.5, "FRED WALCL"),
        ("Bilancio BCE 12m", "ecb_balance_roc_12m", lambda: ultimo_entro(fred("ECBASSETSW"), 12), 0.5, "FRED ECBASSETSW"),
        ("Detentori esteri Treasury 12m", "foreign_treasury_roc_12m", lambda: yoy_per_data(fred("FDHBFIN")), 0.5,
         "FRED FDHBFIN"),
        ("High yield OAS", "hy_oas", lambda: float(fred("BAMLH0A0HYM2").iloc[-1]), 0.15, "FRED BAMLH0A0HYM2"),
        ("Breakeven 10a", "breakeven_10y", lambda: float(fred("T10YIE").iloc[-1]), 0.08, "FRED T10YIE"),
        ("Inflazione area euro", "ea_hicp_yoy", lambda: float(bce("HICP/M.U2.N.000000.4D0.ANR").iloc[-1]), 0.05,
         "BCE HICP"),
        ("Tasso BCE sui depositi", "ecb_deposit_rate", lambda: float(bce("FM/D.U2.EUR.4F.KR.DFR.LEV").iloc[-1]), 0.01,
         "BCE FM DFR"),
        ("Brent fisico", "brent_spot", lambda: float(fred("DCOILBRENTEU").iloc[-1]), 1.0, "FRED DCOILBRENTEU"),
        ("Gas TTF", "ttf_price", lambda: yahoo_ultimo("TTF=F"), 5.0, "Yahoo TTF=F (intraday: tolleranza larga)"),
    ]


def run_crosscheck(site: dict, checks: Optional[list[Check]] = None) -> dict:
    """Confronta i valori del sito con le fonti. Una fonte giu' e' `na`, mai `fail`."""
    results = []
    for nome, chiave, vero, tol, fonte in (checks if checks is not None else default_checks()):
        v_sito = site.get(chiave)
        nota = ""
        try:
            v_vero = vero()
        except Exception as e:
            v_vero, nota = None, f"fonte non disponibile: {str(e)[:80]}"
        if v_sito is None or v_vero is None:
            stato = "na"
            if v_sito is None and not nota:
                nota = "valore assente sul sito"
        else:
            stato = "ok" if abs(float(v_sito) - float(v_vero)) <= tol else "fail"
        results.append({"name": nome, "key": chiave, "site": None if v_sito is None else float(v_sito),
                        "source_value": None if v_vero is None else float(v_vero), "tolerance": tol,
                        "source": fonte, "status": stato, "note": nota})
    return {
        "checked_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "n_ok": sum(r["status"] == "ok" for r in results),
        "n_fail": sum(r["status"] == "fail" for r in results),
        "n_na": sum(r["status"] == "na" for r in results),
        "results": results,
    }


def save_result(result: dict, path: Optional[Path] = None) -> None:
    path = path or STORE
    precedente = load_result(path) or {}
    storia = (precedente.get("history") or []) + [
        {k: result[k] for k in ("checked_at", "n_ok", "n_fail", "n_na")}]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({**result, "history": storia[-HISTORY_LEN:]}, ensure_ascii=False), encoding="utf-8")


def load_result(path: Optional[Path] = None) -> Optional[dict]:
    path = path or STORE
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
