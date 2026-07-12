"""Spot prices in EUR con cache DB (TTL breve) e stale-if-error.

Diverso da services/prices/yahoo_fetcher (storico daily per il modello): qui
serve l'ULTIMO prezzo per valorizzare portafogli, con conversione in EUR.

Strategia per ticker:
1. cache DB fresca (< TTL) → usala
2. fetch yfinance (fast_info, fallback history 5d) + conversione FX → aggiorna cache
3. fetch fallito → serve la cache stale (qualunque età) marcata is_stale
"""
from __future__ import annotations

from datetime import datetime, timedelta

from loguru import logger
from sqlalchemy.orm import Session

from app.models import PtPriceCache

CACHE_TTL = timedelta(minutes=15)

# GBp (pence) → GBP prima della conversione EUR
_PENCE_CURRENCIES = {"GBp", "GBX"}


def _fetch_spot_raw(ticker: str) -> tuple[float, str] | None:
    """(prezzo, valuta) via yfinance, o None. Import lazy come yahoo_fetcher."""
    try:
        import yfinance as yf

        t = yf.Ticker(ticker)
        price = None
        currency = None
        try:
            fi = t.fast_info
            price = fi.get("lastPrice") if hasattr(fi, "get") else getattr(fi, "last_price", None)
            currency = fi.get("currency") if hasattr(fi, "get") else getattr(fi, "currency", None)
        except Exception:
            pass
        if not price:
            hist = t.history(period="5d")
            if hist is not None and not hist.empty:
                price = float(hist["Close"].iloc[-1])
        if not price:
            return None
        return float(price), str(currency or "USD")
    except Exception as e:
        logger.warning(f"pt spot fetch failed {ticker}: {e}")
        return None


def _eur_rate(db: Session, currency: str) -> float | None:
    """Unità di `currency` per 1 EUR (es. EURUSD=X ~ 1.08). EUR → 1."""
    if currency == "EUR":
        return 1.0
    pair = f"EUR{currency}=X"
    cached = get_spot(db, pair, _convert=False)
    return cached["price"] if cached else None


def to_eur(db: Session, price: float, currency: str) -> float | None:
    if currency in _PENCE_CURRENCIES:
        price = price / 100.0
        currency = "GBP"
    rate = _eur_rate(db, currency)
    if rate is None or rate <= 0:
        return None
    return price / rate


def get_spot(db: Session, ticker: str, _convert: bool = True) -> dict | None:
    """Spot per ticker: {price, currency, price_eur, fetched_at, is_stale} o None."""
    row = db.query(PtPriceCache).filter(PtPriceCache.ticker == ticker).first()
    now = datetime.utcnow()
    if row and now - row.fetched_at < CACHE_TTL:
        return _row_dict(row, is_stale=False)

    raw = _fetch_spot_raw(ticker)
    if raw is None:
        if row:  # stale-if-error: meglio un prezzo vecchio marcato che niente
            return _row_dict(row, is_stale=True)
        return None

    price, currency = raw
    price_eur = to_eur(db, price, currency) if _convert else None
    if price_eur is None:
        # FX non disponibile: per pair FX (=X) il price "è" il dato; per gli
        # altri degradiamo a EUR sconosciuto ma salviamo comunque il nativo.
        price_eur = price if currency == "EUR" or ticker.endswith("=X") else 0.0

    if row is None:
        row = PtPriceCache(ticker=ticker, price=price, currency=currency, price_eur=price_eur)
        db.add(row)
    else:
        row.price = price
        row.currency = currency
        row.price_eur = price_eur
        row.fetched_at = now
    db.commit()
    return _row_dict(row, is_stale=False)


def _row_dict(row: PtPriceCache, is_stale: bool) -> dict:
    return {
        "ticker": row.ticker,
        "price": float(row.price),
        "currency": row.currency,
        "price_eur": float(row.price_eur) if row.price_eur else None,
        "fetched_at": row.fetched_at.isoformat(),
        "is_stale": is_stale,
    }


def spot_eur_for_assets(db: Session, key_ticker: dict[str, str]) -> dict[str, float | None]:
    """{asset_key: prezzo EUR per unità canonica} per la valorizzazione portafoglio."""
    out: dict[str, float | None] = {}
    for key, ticker in key_ticker.items():
        spot = get_spot(db, ticker)
        out[key] = spot["price_eur"] if spot and spot.get("price_eur") else None
    return out
