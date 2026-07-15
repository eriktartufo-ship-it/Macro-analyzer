"""Yahoo Finance price fetcher con cache su disco (parquet).

Pattern simmetrico al FredFetcher:
  - cache di sessione in memoria
  - cache disco parquet in `.cache/yahoo/<ticker>.parquet`
  - retry con backoff per errori transitori

Fornisce serie price daily (close adjusted) per il calcolo dei rendimenti reali.
"""

from __future__ import annotations

import time
from datetime import date, timedelta
from pathlib import Path
from typing import Optional

import pandas as pd
from loguru import logger

from app.services.prices.asset_universe import ASSET_TICKERS, BENCHMARK_TICKERS

# Cache disco: backend/.cache/yahoo/
_CACHE_ROOT = Path(__file__).resolve().parents[3] / ".cache" / "yahoo"
_CACHE_TTL_SECONDS = 6 * 60 * 60  # 6 ore: i prezzi cambiano daily, ma non bombardiamo Yahoo
# AUDIT 2026-06-19: mem-cache aveva NO TTL → restituiva la prima serie cache-ata
# per sempre finché il container non riavviava. SPY mostrava daily -0.5962% di 2gg fa.
# Aggiungo TTL in-memory di 1h (forza refresh dentro lo stesso giorno se cache vecchia).
_MEM_TTL_SECONDS = 60 * 60  # 1 ora


def _cache_path(ticker: str) -> Path:
    safe = ticker.replace("/", "_").replace("=", "_").replace("^", "_")
    return _CACHE_ROOT / f"{safe}.parquet"


def _is_fresh(path: Path) -> bool:
    if not path.exists():
        return False
    age = time.time() - path.stat().st_mtime
    return age < _CACHE_TTL_SECONDS


class YahooFetcher:
    """Fetcher con cache per serie storiche prezzi Yahoo Finance."""

    def __init__(self) -> None:
        # AUDIT 2026-06-19: ogni entry è (timestamp_load, series). TTL 1h.
        self._mem_cache: dict[str, tuple[float, pd.Series]] = {}
        _CACHE_ROOT.mkdir(parents=True, exist_ok=True)

    def _load_disk(
        self, ticker: str, want_end: Optional[date] = None,
    ) -> Optional[pd.Series]:
        """Carica serie da disco se fresca E con dati fino a want_end - 2 trading days.

        AUDIT 2026-06-19 bughunter #1 residuo: TTL file (6h) non garantiva
        freshness dei DATI dentro. Se SPY.parquet ha max_date Jun 16 e age 2h,
        viene servito comunque per richieste end=Jun 19 → daily_change stale.
        Fix: se max(disk.index) < (want_end - 2 days) → invalida e re-fetch.
        """
        path = _cache_path(ticker)
        if not _is_fresh(path):
            return None
        try:
            df = pd.read_parquet(path)
            s = df["close"]
            s.index = pd.to_datetime(s.index)
            if want_end is not None and not s.empty:
                # Tolleranza 3 gg per coprire weekend/holiday market USA
                cutoff = pd.Timestamp(want_end) - pd.Timedelta(days=3)
                if s.index.max() < cutoff:
                    logger.info(
                        f"Yahoo disk cache for {ticker} stale: "
                        f"max={s.index.max().date()} < cutoff={cutoff.date()}, "
                        f"forcing re-fetch"
                    )
                    return None
            return s
        except Exception as e:
            logger.warning(f"Yahoo disk cache read failed for {ticker}: {e}")
            return None

    def _save_disk(self, ticker: str, s: pd.Series) -> None:
        path = _cache_path(ticker)
        try:
            df = s.to_frame(name="close")
            df.to_parquet(path)
        except Exception as e:
            logger.warning(f"Yahoo disk cache write failed for {ticker}: {e}")

    def fetch(
        self,
        ticker: str,
        start_date: Optional[date] = None,
        end_date: Optional[date] = None,
        force_refresh: bool = False,
    ) -> pd.Series:
        """Scarica serie chiusura adjusted per `ticker`.

        Default range: max storia disponibile (start = 1970-01-01).
        Returns: Series con index DatetimeIndex e valori close adjusted.

        AUDIT 2026-06-19: cache_key include end_date (default oggi) per evitare
        di restituire serie stale tra giorni diversi. Mem-cache TTL 1h.

        AUDIT 2026-07-15 (BUG SILENZIOSO): la chiave aveva end_date ma NON
        start_date → chi chiedeva PIÙ storia di un chiamante precedente (stesso
        ticker, stessa TTL) riceveva la serie CORTA di quello, senza errore.
        Caso reale: `llm_strategist` fetcha SPY su 200 giorni di calendario (~135
        di borsa) e mette in cache; poi `risk_managed_sp` chiede 3 anni per la
        MA200 → cache HIT → 135 righe → MA200 impossibile → il bot saltava ogni
        giorno. La regola era già in memoria (`feedback_cache_key_include_daterange`:
        "key con DATE RANGE, mai solo per nome risorsa"): era applicata a metà.
        Ora la chiave include anche lo start. Servire un SOVRAinsieme è innocuo
        (il chiamante fa tail()); servire un SOTTOinsieme è il bug.
        """
        end_for_key = (end_date or date.today()).isoformat()
        start_for_key = (start_date or date(1970, 1, 1)).isoformat()
        cache_key = f"{ticker}|{start_for_key}|{end_for_key}"

        if not force_refresh and cache_key in self._mem_cache:
            ts, s = self._mem_cache[cache_key]
            if (time.time() - ts) < _MEM_TTL_SECONDS:
                return s
            # mem-cache scaduto, ricarica
            self._mem_cache.pop(cache_key, None)

        if not force_refresh:
            disk = self._load_disk(ticker, want_end=end_date or date.today())
            if disk is not None:
                # Filtra alla data richiesta per coerenza con il cache_key
                if end_date is not None:
                    disk = disk[disk.index <= pd.Timestamp(end_date)]
                self._mem_cache[cache_key] = (time.time(), disk)
                return disk

        # Import lazy: yfinance ha import slow
        import yfinance as yf

        start = start_date or date(1970, 1, 1)
        end = end_date or date.today()
        # AUDIT 2026-06-19 bughunter #4: yfinance interpreta end come exclusive.
        # Per ottenere il close del giorno corrente (post-market US 22:00 UTC)
        # serve end = today + 1. Pattern standard yfinance.
        end_for_yf = end + timedelta(days=1)

        logger.info(f"Yahoo: fetching {ticker} from {start} to {end} (yf end={end_for_yf})")

        last_err: Optional[Exception] = None
        for attempt in range(3):
            try:
                df = yf.download(
                    ticker,
                    start=start.isoformat(),
                    end=end_for_yf.isoformat(),
                    progress=False,
                    auto_adjust=True,
                    actions=False,
                    threads=False,
                )
                if df is None or df.empty:
                    raise ValueError(f"empty result for {ticker}")
                # yf.download puo' tornare MultiIndex columns: appiattisco
                if isinstance(df.columns, pd.MultiIndex):
                    df.columns = df.columns.get_level_values(0)
                close = df["Close"].dropna()
                close.name = "close"
                self._mem_cache[cache_key] = (time.time(), close)
                self._save_disk(ticker, close)
                return close
            except Exception as e:
                last_err = e
                logger.warning(f"Yahoo fetch {ticker} attempt {attempt + 1}/3 failed: {e}")
                time.sleep(2 ** attempt)

        raise RuntimeError(f"Yahoo fetch failed for {ticker}: {last_err}")

    def fetch_asset(self, asset_class: str) -> pd.Series:
        """Fetcha la serie principale per un asset class. Se primary ha storia
        corta concatena via:
          1. backfill_proxy (altro ticker Yahoo) se presente, oppure
          2. sintetizzatore TR da yield FRED (per bonds/cash) se asset supportato.
        """
        from app.services.prices.synthetic_bonds import _DURATION as _BOND_ASSETS

        if asset_class not in ASSET_TICKERS:
            raise ValueError(f"Unknown asset class: {asset_class}")
        cfg = ASSET_TICKERS[asset_class]
        primary = self.fetch(cfg["ticker"])

        bf = cfg.get("backfill_proxy")
        if bf:
            return self._concat_with_proxy(primary, bf, asset_class)

        # Nessun proxy Yahoo — proviamo sintetizzatore TR per bonds/cash
        if asset_class in _BOND_ASSETS and primary.index.min() > pd.Timestamp("1970-01-01"):
            try:
                from app.services.prices.synthetic_bonds import synthesize_bond_tr_index
                synth = synthesize_bond_tr_index(asset_class)
                return self._stitch(primary, synth)
            except Exception as e:
                logger.warning(f"Synthetic TR for {asset_class} failed: {e}")

        return primary

    def _concat_with_proxy(self, primary: pd.Series, proxy_ticker: str,
                            asset_class: str) -> pd.Series:
        try:
            proxy = self.fetch(proxy_ticker)
        except Exception as e:
            logger.warning(f"Backfill proxy {proxy_ticker} for {asset_class} failed: {e}")
            return primary
        if proxy.index.min() >= primary.index.min():
            return primary
        return self._stitch(primary, proxy)

    def _stitch(self, primary: pd.Series, extender: pd.Series) -> pd.Series:
        """Concatena due serie scalando l'extender al punto di overlap."""
        common = primary.index.intersection(extender.index)
        if len(common) == 0:
            return primary
        anchor = common.min()
        scale = float(primary.loc[anchor]) / float(extender.loc[anchor])
        extender_pre = extender.loc[: anchor - timedelta(days=1)] * scale
        combined = pd.concat([extender_pre, primary]).sort_index()
        return combined[~combined.index.duplicated(keep="last")]

    def fetch_benchmark(self, name: str) -> pd.Series:
        if name not in BENCHMARK_TICKERS:
            raise ValueError(f"Unknown benchmark: {name}")
        ticker = BENCHMARK_TICKERS[name]
        if ticker is None:
            raise ValueError(f"Benchmark {name} is computed, not fetchable directly")
        return self.fetch(ticker)
