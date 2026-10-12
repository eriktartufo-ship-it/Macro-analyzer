"""Il sito dice la verita'? Confronta i numeri live di Macro con le fonti primarie.

Nato il 2026-10-12: per ~11 mesi il CPI del sito e' stato una variazione a 13 mesi (3,71 contro 3,35
del BLS), il bilancio Fed "-22% in 12 mesi, QT in corso" era la variazione da dicembre 2021, e
nessuno se n'e' accorto perche' nessuno confrontava il sito con la fonte. Questo script lo fa.

Per ogni indicatore: valore dal sito (`/data-snapshot`) contro valore ricalcolato qui dalla fonte,
con un calcolo indipendente da quello dell'app (date esplicite, niente transforms.py).
    ✅ entro la tolleranza   ❌ fuori   ⚪ non giudicabile (fonte giu' o dato assente)
Esce con 1 se c'e' almeno un ❌: si puo' mettere in uno scheduler o in una Action.

Uso:  cd backend && PYTHONPATH=. ../.venv/Scripts/python.exe scripts/controllo_incrociato.py [--url https://macro.pieranlab.cloud]
Fonti: BLS API v2 (senza chiave), Treasury.gov (CSV), FRED API (chiave da .env).
⚠️ Le giornaliere possono differire di un giorno di mercato fra sito e fonte: tolleranze piu' larghe.
"""

from __future__ import annotations

import io
import json
import sys
import urllib.request
from datetime import date

import pandas as pd

sys.stdout.reconfigure(encoding="utf-8")
URL = sys.argv[sys.argv.index("--url") + 1] if "--url" in sys.argv else "https://macro.pieranlab.cloud"


def get(url: str, data: bytes | None = None, timeout: int = 60) -> bytes:
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json", "User-Agent": "controllo-incrociato"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read()


def yoy_per_data(s: pd.Series, months: int = 12) -> float | None:
    """Ultimo valore contro quello ESATTAMENTE `months` mesi prima (stessa data), altrimenti None."""
    s = s.dropna().sort_index()
    base = s.get(s.index[-1] - pd.DateOffset(months=months))
    return None if base is None or pd.isna(base) else (s.iloc[-1] / base - 1) * 100


def ultimo_entro(s: pd.Series, months: int, giorni: int = 7) -> float | None:
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
    d = json.loads(get("https://api.bls.gov/publicAPI/v2/timeseries/data/", body))
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
    df = pd.read_csv(io.StringIO(get(url).decode("utf-8")))
    df["Date"] = pd.to_datetime(df["Date"])
    return df.sort_values("Date").set_index("Date")


def bce(chiave: str) -> pd.Series:
    """Serie del Data Portal BCE letta qui, senza passare dal modulo dell'app."""
    df = pd.read_csv(io.StringIO(get(f"https://data-api.ecb.europa.eu/service/data/{chiave}"
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


def main() -> int:
    snap = json.loads(get(f"{URL}/api/v1/data-snapshot", timeout=120))
    sito = {**snap.get("current_dedollar_indicators", {}), **snap.get("current_indicators", {})}

    controlli = []  # (nome, chiave sul sito, funzione che calcola il valore vero, tolleranza, fonte)
    fonti: dict = {}

    def b():
        if "bls" not in fonti:
            fonti["bls"] = bls(["CUSR0000SA0", "LNS14000000"])
        return fonti["bls"]

    controlli += [
        ("CPI a 12 mesi (SA)", "cpi_yoy", lambda: yoy_per_data(b()["CUSR0000SA0"]), 0.05, "BLS CUSR0000SA0"),
        ("Disoccupazione", "unrate", lambda: float(b()["LNS14000000"].iloc[-1]), 0.05, "BLS LNS14000000"),
        ("Curva 10a-2a", "yield_curve_10y2y",
         lambda: float((lambda t: t["10 Yr"].iloc[-1] - t["2 Yr"].iloc[-1])(treasury("daily_treasury_yield_curve"))), 0.12, "Treasury.gov"),
        ("Reale 10a", "real_yield_10y",
         lambda: float(treasury("daily_treasury_real_yield_curve")["10 YR"].iloc[-1]), 0.12, "Treasury.gov TIPS"),
        ("Bilancio Fed 12m", "fed_balance_roc_12m", lambda: ultimo_entro(fred("WALCL"), 12), 0.5, "FRED WALCL"),
        ("Bilancio BCE 12m", "ecb_balance_roc_12m", lambda: ultimo_entro(fred("ECBASSETSW"), 12), 0.5, "FRED ECBASSETSW"),
        ("Detentori esteri Treasury 12m", "foreign_treasury_roc_12m", lambda: yoy_per_data(fred("FDHBFIN")), 0.5, "FRED FDHBFIN"),
        ("High yield OAS", "hy_oas", lambda: float(fred("BAMLH0A0HYM2").iloc[-1]), 0.15, "FRED BAMLH0A0HYM2"),
        ("Breakeven 10a", "breakeven_10y", lambda: float(fred("T10YIE").iloc[-1]), 0.08, "FRED T10YIE"),
        # S50: area euro ed energia
        ("Inflazione area euro", "ea_hicp_yoy", lambda: float(bce("HICP/M.U2.N.000000.4D0.ANR").iloc[-1]), 0.05, "BCE HICP"),
        ("Tasso BCE sui depositi", "ecb_deposit_rate", lambda: float(bce("FM/D.U2.EUR.4F.KR.DFR.LEV").iloc[-1]), 0.01, "BCE FM DFR"),
        ("Brent fisico", "brent_spot", lambda: float(fred("DCOILBRENTEU").iloc[-1]), 1.0, "FRED DCOILBRENTEU"),
        ("Gas TTF", "ttf_price", lambda: yahoo_ultimo("TTF=F"), 5.0, "Yahoo TTF=F (intraday: toll. larga)"),
    ]

    rossi = 0
    print(f"Controllo incrociato {URL} — snapshot del {snap.get('date')}\n")
    print(f"   {'indicatore':30} {'sito':>9} {'fonte':>9}  fonte")
    for nome, chiave, vero, tol, fonte in controlli:
        v_sito = sito.get(chiave)
        try:
            v_vero = vero()
        except Exception as e:  # una fonte giu' non e' un ❌: e' un "non so"
            v_vero, fonte = None, f"{fonte} — errore: {str(e)[:60]}"
        if v_sito is None or v_vero is None:
            segno = "⚪"
        elif abs(float(v_sito) - v_vero) <= tol:
            segno = "✅"
        else:
            segno, rossi = "❌", rossi + 1
        fs = "—" if v_sito is None else f"{float(v_sito):9.2f}"
        fv = "—" if v_vero is None else f"{v_vero:9.2f}"
        print(f"{segno} {nome:30} {fs:>9} {fv:>9}  {fonte} (toll. {tol})")
    print(f"\n{rossi} fuori tolleranza" if rossi else "\nTutto entro tolleranza")
    return 1 if rossi else 0


if __name__ == "__main__":
    sys.exit(main())
