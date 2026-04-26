"""Fama-French 3-factor + momentum: fetch da Kenneth French data library.

Factor monthly returns dal 1926 (Mkt-RF, SMB, HML) + Mom dal 1926 separato.
Cache disco parquet con TTL lungo (factor data e' append-only mensile).

Glossario:
  - Mkt-RF = market excess return (S&P 500 - Tbill)
  - SMB    = Small Minus Big (size factor: small caps - large caps)
  - HML    = High Minus Low book/market (value factor: value - growth)
  - Mom    = Momentum (cross-section: vincitori - perdenti 12-1 mesi)
  - RF     = risk-free rate (T-bill 1m)

Tutti i valori sono in **percento** (es. 2.89 = +2.89%).
"""

from __future__ import annotations

import io
import time
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import pandas as pd
import requests
from loguru import logger


_CACHE_ROOT = Path(__file__).resolve().parents[3] / ".cache" / "fama_french"
_CACHE_TTL_SECONDS = 24 * 60 * 60  # 24h: i factor data sono mensili, no need a refresh frequenti

_FF_3F_URL = "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/F-F_Research_Data_Factors_CSV.zip"
_FF_MOM_URL = "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/F-F_Momentum_Factor_CSV.zip"


def _cache_path(name: str) -> Path:
    return _CACHE_ROOT / f"{name}.parquet"


def _is_fresh(p: Path) -> bool:
    if not p.exists():
        return False
    return (time.time() - p.stat().st_mtime) < _CACHE_TTL_SECONDS


def _parse_ff_csv(text: str) -> pd.DataFrame:
    """Parsa il formato Kenneth French.

    Il file CSV contiene multiple sezioni: monthly returns, annual returns, etc.
    Prendiamo SOLO la prima sezione (monthly), che e' tipo:
      `,Mkt-RF,SMB,HML,RF\\n192607, 2.89, ...\\n` etc.
    Le sezioni sono separate da righe vuote o testo.
    """
    lines = text.split("\n")
    # Trova l'header (prima riga che inizia con ',' seguita da almeno un nome di colonna)
    header_idx = None
    for i, L in enumerate(lines):
        stripped = L.strip()
        if stripped.startswith(",") and len(stripped) > 1:
            # Almeno un carattere alfabetico dopo la virgola = header colonne
            after = stripped[1:].strip()
            if after and after[0].isalpha():
                header_idx = i
                break
    if header_idx is None:
        raise ValueError("Header CSV non trovato")

    rows = []
    for L in lines[header_idx + 1:]:
        L = L.strip()
        if not L:
            break
        parts = L.split(",")
        first = parts[0].strip()
        # Se non e' una data YYYYMM (6 digits) interrompi (siamo arrivati alla annual section)
        if not (first.isdigit() and len(first) == 6):
            break
        rows.append(parts)

    if not rows:
        raise ValueError("Nessuna riga monthly parsata")

    cols = lines[header_idx].split(",")[1:]  # skip leading empty
    cols = [c.strip() for c in cols]

    df_data = {c: [] for c in cols}
    dates = []
    for parts in rows:
        ym = parts[0].strip()
        year = int(ym[:4])
        month = int(ym[4:6])
        # Fine mese
        dates.append(pd.Timestamp(year=year, month=month, day=1) + pd.offsets.MonthEnd(0))
        for c, v in zip(cols, parts[1:]):
            df_data[c].append(float(v.strip()))

    df = pd.DataFrame(df_data, index=pd.DatetimeIndex(dates, name="date"))
    return df


def _fetch_zip_csv(url: str) -> pd.DataFrame:
    r = requests.get(url, timeout=30)
    r.raise_for_status()
    z = zipfile.ZipFile(io.BytesIO(r.content))
    name = z.namelist()[0]
    with z.open(name) as f:
        text = f.read().decode("utf-8", errors="ignore")
    return _parse_ff_csv(text)


@dataclass
class FactorReturns:
    monthly: pd.DataFrame   # cols: Mkt-RF, SMB, HML, RF, Mom
    source: str


def fetch_fama_french(force_refresh: bool = False) -> FactorReturns:
    """Restituisce DataFrame mensile dei 4 fattori (Mkt-RF, SMB, HML, Mom) + RF."""
    _CACHE_ROOT.mkdir(parents=True, exist_ok=True)
    cache3f = _cache_path("ff_3factor")
    cachemom = _cache_path("ff_momentum")

    if not force_refresh and _is_fresh(cache3f) and _is_fresh(cachemom):
        df3 = pd.read_parquet(cache3f)
        dfm = pd.read_parquet(cachemom)
        df3.index = pd.to_datetime(df3.index)
        dfm.index = pd.to_datetime(dfm.index)
    else:
        logger.info("Fama-French: fetching 3-factor + momentum from Dartmouth")
        df3 = _fetch_zip_csv(_FF_3F_URL)
        try:
            dfm = _fetch_zip_csv(_FF_MOM_URL)
        except Exception as e:
            logger.warning(f"Momentum factor fetch failed: {e}; continuing without Mom")
            dfm = pd.DataFrame(index=df3.index)
        df3.to_parquet(cache3f)
        dfm.to_parquet(cachemom)

    # Combina: il momentum file ha colonna "Mom" (a volte "MOM")
    if not dfm.empty:
        mom_col = next((c for c in dfm.columns if "mom" in c.lower()), None)
        if mom_col:
            df3 = df3.join(dfm[[mom_col]].rename(columns={mom_col: "Mom"}), how="left")

    return FactorReturns(monthly=df3.dropna(how="all"), source="Kenneth French Data Library")


def factor_keys() -> list[str]:
    """Lista delle colonne factor (esclude RF). Mkt-RF in formato sicuro per JSON."""
    return ["Mkt-RF", "SMB", "HML", "Mom"]
