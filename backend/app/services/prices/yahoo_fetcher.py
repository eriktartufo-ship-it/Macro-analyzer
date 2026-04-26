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
        self._mem_cache: dict[str, pd.Series] = {}
        _CACHE_ROOT.mkdir(parents=True, exist_ok=True)

    def _load_disk(self, ticker: str) -> Optional[pd.Series]:
        path = _cache_path(ticker)
        if not _is_fresh(path):
            return None
        try:
            df = pd.read_parquet(path)
            s = df["close"]
            s.index = pd.to_datetime(s.index)
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
        """
        cache_key = ticker
        if not force_refresh and cache_key in self._mem_cache:
            return self._mem_cache[cache_key]

        if not force_refresh:
            disk = self._load_disk(ticker)
            if disk is not None:
                self._mem_cache[cache_key] = disk
                return disk

        # Import lazy: yfinance ha import slow
        import yfinance as yf

        start = start_date or date(1970, 1, 1)
        end = end_date or date.today()

        logger.info(f"Yahoo: fetching {ticker} from {start} to {end}")

        last_err: Optional[Exception] = None
        for attempt in range(3):
            try:
                df = yf.download(
                    ticker,
                    start=start.isoformat(),
                    end=end.isoformat(),
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
                self._mem_cache[cache_key] = close
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
