"""Area euro ed energia: i dati che i video citano e Macro non aveva (S50, 2026-10-12).

Macro era costruito sugli USA. L'inventario del 12/10 (logbook S49) ha messo in fila cio' che
misuravamo a mano negli ingest e che il sito non vedeva: inflazione dell'area euro, tasso BCE sui
depositi, curva e spread sovrano euro, disoccupazione dell'area, gas TTF e Brent (anche il premio del
fisico sulla carta).

Fonti, tutte senza chiave:
- BCE Data Portal. ⚠️ L'inflazione 2026 sta nel dataflow `HICP` (struttura ECB_ICP3, suffisso 4D0):
  il vecchio dataflow `ICP` si ferma a dicembre 2025 (riclassificazione COICOP 2026), e chi lo usa
  vede un dato vecchio di 9 mesi senza errore;
- Yahoo: TTF=F (dal 2017), BZ=F (future Brent, dal 2007);
- FRED DCOILBRENTEU: Brent spot (dal 1987), arriva con qualche giorno di ritardo.

Gli indicatori sono INFORMATIVI: entrano nel dizionario del refresh (snapshot, /regime/explain, LLM)
ma il classificatore li usa solo dietro un flag, e solo se passano il cancello A/B (criteri S50).
Ogni fonte e' protetta: se cade, spariscono solo le sue chiavi.
"""

from __future__ import annotations

import io
import pickle
import time
from pathlib import Path
from typing import Callable, Optional

import pandas as pd
import requests
from loguru import logger

from app.services.indicators.transforms import roc_between, value_at_or_before

ECB_API = "https://data-api.ecb.europa.eu/service/data/"

# nome interno -> (chiave BCE, frequenza)
ECB_SERIES: dict[str, tuple[str, str]] = {
    "ea_hicp": ("HICP/M.U2.N.000000.4D0.ANR", "monthly"),          # gia' tasso annuo %
    "ea_core_hicp": ("HICP/M.U2.N.XEF000.4D0.ANR", "monthly"),     # senza energia e alimentari
    "ea_energy_hicp": ("HICP/M.U2.N.NRGY00.4D0.ANR", "monthly"),
    "ea_services_hicp": ("HICP/M.U2.N.SERV00.4D0.ANR", "monthly"),
    "ecb_deposit_rate": ("FM/D.U2.EUR.4F.KR.DFR.LEV", "daily"),
    "ea_aaa_10y": ("YC/B.U2.EUR.4F.G_N_A.SV_C_YM.SR_10Y", "daily"),
    "ea_aaa_2y": ("YC/B.U2.EUR.4F.G_N_A.SV_C_YM.SR_2Y", "daily"),
    "ea_all_10y": ("YC/B.U2.EUR.4F.G_N_C.SV_C_YM.SR_10Y", "daily"),  # curva di tutti i rating
    "ea_unemployment": ("LFSI/M.I9.S.UNEHRT.TOTAL0.15_74.T", "monthly"),
}
YAHOO_SERIES = {"ttf": "TTF=F", "brent_future": "BZ=F"}

_CACHE_DIR = Path(__file__).resolve().parents[3] / ".cache" / "ecb"
_TTL = {"daily": 6 * 3600, "monthly": 24 * 3600}


def parse_ecb_csv(text: str) -> pd.Series:
    """CSV del Data Portal BCE -> Series datata (mensili al primo del mese). Valori vuoti saltati."""
    df = pd.read_csv(io.StringIO(text), usecols=["TIME_PERIOD", "OBS_VALUE"])
    df = df.dropna(subset=["OBS_VALUE"])
    idx = pd.to_datetime(df["TIME_PERIOD"].astype(str))
    return pd.Series(df["OBS_VALUE"].astype(float).to_numpy(), index=idx).sort_index()


def fetch_ecb(name: str, timeout: int = 60) -> pd.Series:
    """Scarica una serie BCE con cache su disco (TTL per frequenza)."""
    key, freq = ECB_SERIES[name]
    path = _CACHE_DIR / f"{name}.pkl"
    if path.exists() and time.time() - path.stat().st_mtime < _TTL[freq]:
        with path.open("rb") as f:
            return pickle.load(f)
    r = requests.get(f"{ECB_API}{key}", params={"format": "csvdata", "detail": "dataonly"}, timeout=timeout)
    r.raise_for_status()
    s = parse_ecb_csv(r.text)
    try:
        _CACHE_DIR.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as f:
            pickle.dump(s, f)
    except Exception as e:  # la cache e' un'ottimizzazione, non un requisito
        logger.warning(f"cache BCE {name}: {e}")
    return s


def percentile_of_last(s: pd.Series) -> Optional[float]:
    """Quota della storia (in %) che sta sotto o al pari dell'ultimo valore."""
    s = s.dropna()
    if s.empty:
        return None
    return float((s <= s.iloc[-1]).mean() * 100)


def _last(s: Optional[pd.Series]) -> Optional[tuple[pd.Timestamp, float]]:
    if s is None:
        return None
    s = s.dropna()
    if s.empty:
        return None
    return s.index[-1], float(s.iloc[-1])


def europe_energy_indicators(series: dict[str, pd.Series], us: Optional[dict] = None) -> dict[str, float]:
    """Indicatori derivati. Funzione pura: ogni blocco salta da solo se manca la sua serie."""
    out: dict[str, float] = {}

    def blocco(fn: Callable[[], None], nome: str) -> None:
        try:
            fn()
        except Exception as e:
            logger.warning(f"europe_energy {nome}: {e}")

    def inflazione():
        for src, key in (("ea_hicp", "ea_hicp_yoy"), ("ea_core_hicp", "ea_core_hicp_yoy"),
                         ("ea_energy_hicp", "ea_energy_hicp_yoy"), ("ea_services_hicp", "ea_services_hicp_yoy")):
            v = _last(series.get(src))
            if v:
                out[key] = v[1]
        h = series.get("ea_hicp")
        v = _last(h)
        if v:
            prima = value_at_or_before(h, v[0] - pd.DateOffset(months=6))
            if prima is not None:
                out["ea_hicp_change_6m"] = v[1] - prima

    def tassi():
        d = _last(series.get("ecb_deposit_rate"))
        if d:
            out["ecb_deposit_rate"] = d[1]
            if "ea_hicp_yoy" in out:
                out["ea_real_rate"] = d[1] - out["ea_hicp_yoy"]

    def lavoro():
        u = series.get("ea_unemployment")
        v = _last(u)
        if v:
            out["ea_unemployment"] = v[1]
            prima = value_at_or_before(u, v[0] - pd.DateOffset(months=12))
            if prima is not None:
                out["ea_unemployment_change_12m"] = v[1] - prima

    def curva():
        a10, a2, tutti = series.get("ea_aaa_10y"), series.get("ea_aaa_2y"), series.get("ea_all_10y")
        v10 = _last(a10)
        if v10:
            out["ea_aaa_10y"] = v10[1]
            v2 = _last(a2)
            if v2 and v2[0] == v10[0]:
                out["ea_aaa_2y"] = v2[1]
                out["ea_aaa_slope_2_10"] = v10[1] - v2[1]
        if a10 is not None and tutti is not None and not a10.dropna().empty and not tutti.dropna().empty:
            # spread sovrano = curva di tutti i rating meno AAA, sulle date comuni
            spread = (tutti - a10).dropna()
            if not spread.empty:
                out["ea_sovereign_spread_10y"] = float(spread.iloc[-1])
                p = percentile_of_last(spread)
                if p is not None:
                    out["ea_sovereign_spread_pctile"] = p

    def gas():
        t = series.get("ttf")
        v = _last(t)
        if v:
            out["ttf_price"] = v[1]
            y = roc_between(t, v[0], 12)
            if y is not None:
                out["ttf_yoy"] = y

    def petrolio():
        spot, fut = series.get("brent_spot"), series.get("brent_future")
        vf = _last(fut)
        if vf:
            out["brent_future"] = vf[1]
        vs = _last(spot)
        if vs:
            out["brent_spot"] = vs[1]
            y = roc_between(spot, vs[0], 12)
            if y is not None:
                out["brent_yoy"] = y
                out["oil_yoy"] = y  # ingresso del pilastro shock energetico (flag, S50)
            if fut is not None:
                # lo spot arriva in ritardo: confronto col future ALLA DATA dello spot
                f_stessa_data = value_at_or_before(fut, vs[0])
                if f_stessa_data is not None:
                    out["brent_physical_premium"] = vs[1] - f_stessa_data

    def differenziale():
        if us and "ea_real_rate" in out:
            ff, cpi = us.get("fed_funds_rate"), us.get("cpi_yoy")
            if ff is not None and cpi is not None:
                out["us_ea_real_rate_diff"] = (ff - cpi) - out["ea_real_rate"]

    for fn in (inflazione, tassi, lavoro, curva, gas, petrolio, differenziale):
        blocco(fn, fn.__name__)
    return out


def fetch_europe_energy_series(yahoo=None, fred=None) -> dict[str, pd.Series]:
    """Scarica tutte le serie; una fonte che cade lascia la sua chiave fuori (e lo logga)."""
    series: dict[str, pd.Series] = {}
    for name in ECB_SERIES:
        try:
            series[name] = fetch_ecb(name)
        except Exception as e:
            logger.warning(f"BCE {name} non disponibile: {e}")
    if yahoo is not None:
        for name, ticker in YAHOO_SERIES.items():
            try:
                series[name] = yahoo.fetch(ticker)
            except Exception as e:
                logger.warning(f"Yahoo {ticker} non disponibile: {e}")
    if fred is not None:
        try:
            series["brent_spot"] = fred.fetch_series("brent_spot")
        except Exception as e:
            logger.warning(f"FRED brent_spot non disponibile: {e}")
    return series


AGSI_URL = "https://agsi.gie.eu/api"


def _agsi(api_key: str, **params) -> Optional[dict]:
    """Prima riga di AGSI+ (aggregato UE). Una chiave rifiutata torna un JSON con `error`, con HTTP 200."""
    r = requests.get(AGSI_URL, params={"type": "eu", "size": 1, **params}, headers={"x-key": api_key}, timeout=30)
    r.raise_for_status()
    j = r.json()
    if j.get("error"):
        raise RuntimeError(f"AGSI: {j.get('message') or j['error']}")
    rows = j.get("data") or []
    return rows[0] if rows else None


def gas_storage_indicators(api_key: str) -> dict[str, float]:
    """S51: stoccaggi di gas UE (AGSI+, GIE). Senza chiave nessuna chiamata, nessuna chiave.

    ea_gas_storage_pct = riempimento %, ea_gas_storage_trend = variazione del giorno (pp),
    ea_gas_storage_vs_1y = punti in meno/in piu' rispetto allo STESSO giorno gas di un anno prima.
    ⚠️ Struttura della risposta dalla documentazione AGSI: da provare dal vivo con la chiave vera.
    """
    if not api_key:
        return {}
    out: dict[str, float] = {}
    try:
        oggi = _agsi(api_key)
        if not oggi or oggi.get("full") in (None, "", "-"):
            return {}
        out["ea_gas_storage_pct"] = float(oggi["full"])
        if oggi.get("trend") not in (None, "", "-"):
            out["ea_gas_storage_trend"] = float(oggi["trend"])
        giorno = pd.Timestamp(oggi["gasDayStart"])
        prima = _agsi(api_key, date=(giorno - pd.DateOffset(years=1)).date().isoformat())
        if prima and prima.get("full") not in (None, "", "-"):
            out["ea_gas_storage_vs_1y"] = out["ea_gas_storage_pct"] - float(prima["full"])
    except Exception as e:
        logger.warning(f"AGSI stoccaggi non disponibili: {e}")
        return {}
    return out


def live_europe_energy_indicators(us: Optional[dict] = None) -> dict[str, float]:
    """Punto d'ingresso del refresh giornaliero."""
    from app.config import settings
    from app.services.indicators.fetcher import FredFetcher
    from app.services.prices.yahoo_fetcher import YahooFetcher

    out = europe_energy_indicators(fetch_europe_energy_series(YahooFetcher(), FredFetcher()), us=us)
    out.update(gas_storage_indicators(settings.agsi_api_key))
    return out


# Il pannello del sito: gruppi, etichette e unita'. Una voce senza valore resta (None = "—"),
# cosi' una fonte giu' si vede invece di sparire.
PANEL_GROUPS: list[tuple[str, str, list[tuple[str, str, str]]]] = [
    ("inflazione", "Inflazione area euro", [
        ("ea_hicp_yoy", "Totale", "%"),
        ("ea_core_hicp_yoy", "Core (senza energia e alimentari)", "%"),
        ("ea_energy_hicp_yoy", "Energia", "%"),
        ("ea_services_hicp_yoy", "Servizi", "%"),
        ("ea_hicp_change_6m", "Variazione in 6 mesi", "pp"),
    ]),
    ("tassi", "BCE e tassi", [
        ("ecb_deposit_rate", "Tasso BCE sui depositi", "%"),
        ("ea_real_rate", "Tasso reale (depositi − inflazione)", "%"),
        ("us_ea_real_rate_diff", "Differenziale reale USA − area euro", "pp"),
        ("ea_aaa_2y", "Titoli AAA 2 anni", "%"),
        ("ea_aaa_10y", "Titoli AAA 10 anni", "%"),
        ("ea_aaa_slope_2_10", "Pendenza 2-10 anni", "pp"),
        ("ea_sovereign_spread_10y", "Spread sovrano 10 anni", "pp"),
    ]),
    ("energia", "Energia", [
        ("brent_spot", "Brent fisico", "$"),
        ("brent_future", "Brent future", "$"),
        ("brent_physical_premium", "Premio del fisico sulla carta", "$"),
        ("brent_yoy", "Brent in 12 mesi", "%"),
        ("ttf_price", "Gas TTF", "€/MWh"),
        ("ttf_yoy", "Gas TTF in 12 mesi", "%"),
        ("ea_gas_storage_pct", "Stoccaggi gas UE", "%"),
        ("ea_gas_storage_vs_1y", "Stoccaggi rispetto a un anno fa", "pp"),
    ]),
    ("lavoro", "Lavoro", [
        ("ea_unemployment", "Disoccupazione area euro", "%"),
        ("ea_unemployment_change_12m", "Variazione in 12 mesi", "pp"),
    ]),
]


def europe_energy_payload(indicators: dict, as_of) -> dict:
    """Risposta di GET /europe-energy, costruita dagli indicatori dell'ultimo refresh."""
    groups = []
    found = 0
    for gid, title, items in PANEL_GROUPS:
        rows = []
        for key, label, unit in items:
            v = indicators.get(key)
            v = float(v) if isinstance(v, (int, float)) else None
            found += v is not None
            rows.append({"key": key, "label": label, "unit": unit, "value": v})
        groups.append({"id": gid, "title": title, "items": rows})
    pct = indicators.get("ea_sovereign_spread_pctile")
    return {
        "as_of": str(as_of) if as_of is not None else None,
        "available": found > 0,
        "groups": groups,
        "sovereign_spread_pctile": float(pct) if isinstance(pct, (int, float)) else None,
    }
