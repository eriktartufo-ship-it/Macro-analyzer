"""Dati di mercato per il libro reale: barre giornaliere GREZZE (apertura/chiusura) e cedole.

Perché non `prices/yahoo_fetcher.py`: quello dà solo la chiusura, e RETTIFICATA per i
dividendi (giusto per il modello, sbagliato qui). Un prezzo di esecuzione rettificato
cambierebbe a ogni stacco cedola successivo: l'eseguito del 21/10 non sarebbe più quello
del 21/10. Qui `auto_adjust=False` ⇒ prezzi veri di borsa, e le cedole si accreditano a
parte in cassa (`dividends`).

Orario d'apertura: Xetra e Borsa Italiana aprono alle 09:00 ora di Francoforte/Milano.
L'ora legale UE si calcola a mano (ultima domenica di marzo → ultima di ottobre, alle
01:00 UTC) per non dipendere da `tzdata`, che nell'immagine slim può mancare.
"""
from __future__ import annotations

import logging
from datetime import date, datetime, timedelta
from typing import Protocol

import pandas as pd

logger = logging.getLogger(__name__)

OPEN_LOCAL_HOUR = 9


class Market(Protocol):
    def bars(self, symbol: str, start: date, end: date) -> pd.DataFrame:
        """Indice = date (datetime.date), colonne Open/Close in valuta di quotazione."""
        ...

    def dividends(self, symbol: str, start: date, end: date) -> pd.Series:
        """Indice = data di stacco (datetime.date), valore = importo per quota."""
        ...


def _last_sunday(year: int, month: int) -> date:
    d = date(year, month + 1, 1) - timedelta(days=1) if month < 12 else date(year, 12, 31)
    return d - timedelta(days=(d.weekday() + 1) % 7)


def eu_summer_time(d: date) -> bool:
    """Ora legale UE in vigore alle 09:00 locali del giorno `d`."""
    return _last_sunday(d.year, 3) <= d < _last_sunday(d.year, 10)


def open_time_utc(d: date) -> datetime:
    """Istante UTC (naive) dell'apertura delle 09:00 a Francoforte/Milano."""
    offset = 2 if eu_summer_time(d) else 1
    return datetime(d.year, d.month, d.day, OPEN_LOCAL_HOUR - offset, 0)


def _to_date_index(idx) -> list[date]:
    out = []
    for ts in idx:
        ts = pd.Timestamp(ts)
        out.append(ts.date())
    return out


class YahooMarket:
    """Barre grezze da yfinance con cache in memoria per la durata di un giro."""

    def __init__(self) -> None:
        self._bars: dict[str, pd.DataFrame] = {}
        self._divs: dict[str, pd.Series] = {}

    def _history(self, symbol: str) -> pd.DataFrame:
        if symbol not in self._bars:
            df = pd.DataFrame(columns=["Open", "Close"])
            try:
                import yfinance as yf

                h = yf.Ticker(symbol).history(period="2y", auto_adjust=False)
                if h is not None and not h.empty:
                    df = h[["Open", "Close"]].copy()
                    df.index = _to_date_index(df.index)
                    df = df[(df["Open"] > 0) & (df["Close"] > 0)]
            except Exception as e:
                logger.warning(f"real_book: barre {symbol} non disponibili: {e}")
            self._bars[symbol] = df
        return self._bars[symbol]

    def bars(self, symbol: str, start: date, end: date) -> pd.DataFrame:
        df = self._history(symbol)
        if df.empty:
            return df
        mask = [(start <= d <= end) for d in df.index]
        return df[mask]

    def dividends(self, symbol: str, start: date, end: date) -> pd.Series:
        if symbol not in self._divs:
            s = pd.Series(dtype=float)
            try:
                import yfinance as yf

                d = yf.Ticker(symbol).dividends
                if d is not None and not d.empty:
                    s = pd.Series(d.values, index=_to_date_index(d.index), dtype=float)
            except Exception as e:
                logger.warning(f"real_book: cedole {symbol} non disponibili: {e}")
            self._divs[symbol] = s
        s = self._divs[symbol]
        if s.empty:
            return s
        return s[[(start <= d <= end) for d in s.index]]


def first_bar_on_or_after(market: Market, symbol: str, d: date, until: date) -> tuple[date, float, float] | None:
    """Prima seduta con data >= d: (data, apertura, chiusura). None se non c'è ancora."""
    if until < d:
        return None
    df = market.bars(symbol, d, until)
    if df is None or df.empty:
        return None
    first = min(df.index)
    row = df.loc[first]
    return first, float(row["Open"]), float(row["Close"])


def last_close_on_or_before(market: Market, symbol: str, d: date, lookback_days: int = 14) -> float | None:
    df = market.bars(symbol, d - timedelta(days=lookback_days), d)
    if df is None or df.empty:
        return None
    return float(df.loc[max(df.index)]["Close"])


def close_on(market: Market, symbol: str, d: date) -> float | None:
    df = market.bars(symbol, d, d)
    if df is None or df.empty:
        return None
    return float(df.iloc[-1]["Close"])
