"""Servizio per fetch dati da FRED API."""

import pickle
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date
from pathlib import Path
from typing import Optional

import pandas as pd
from fredapi import Fred
from loguru import logger

from app.config import settings
from app.services.indicators.fred_codes import FRED_SERIES
from app.services.indicators.transforms import (
    calculate_roc,
    calculate_zscore,
)

# --- Disk cache: TTL per frequenza della serie ---
# Evitiamo di ri-fetchare serie mensili/trimestrali se non sono passati giorni sufficienti
# perché il loro valore non cambia più spesso di così.
_CACHE_DIR = Path(__file__).resolve().parents[3] / ".cache" / "fred"
_TTL_SECONDS_BY_FREQ = {
    "daily": 6 * 3600,             # 6h: prezzi/yield intra-day tipicamente aggiornati una volta
    "weekly": 24 * 3600,            # 1 giorno
    "monthly": 3 * 24 * 3600,       # 3 giorni (release schedule BLS/Fed)
    "quarterly": 7 * 24 * 3600,     # 1 settimana
    "annual": 30 * 24 * 3600,       # 30 giorni
}


class FredFetcher:
    """Client per scaricare e trasformare dati FRED."""

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or settings.fred_api_key
        self.fred = Fred(api_key=self.api_key)
        self._cache: dict[str, pd.Series] = {}
        self._transform_cache: dict[str, dict[str, pd.Series]] = {}

    # ---------------- disk cache helpers ----------------
    @staticmethod
    def _cache_path(series_name: str) -> Path:
        return _CACHE_DIR / f"{series_name}.pkl"

    @classmethod
    def _load_disk_cache(cls, series_name: str, freq: str) -> Optional[pd.Series]:
        path = cls._cache_path(series_name)
        if not path.exists():
            return None
        ttl = _TTL_SECONDS_BY_FREQ.get(freq, 24 * 3600)
        age = time.time() - path.stat().st_mtime
        if age > ttl:
            return None
        try:
            with path.open("rb") as f:
                return pickle.load(f)
        except Exception as e:
            logger.warning(f"Disk cache read failed for {series_name}: {e}")
            return None

    @classmethod
    def _save_disk_cache(cls, series_name: str, series: pd.Series) -> None:
        try:
            _CACHE_DIR.mkdir(parents=True, exist_ok=True)
            with cls._cache_path(series_name).open("wb") as f:
                pickle.dump(series, f)
        except Exception as e:
            logger.warning(f"Disk cache write failed for {series_name}: {e}")

    def fetch_series(
        self,
        series_name: str,
        start_date: Optional[date] = None,
        end_date: Optional[date] = None,
    ) -> pd.Series:
        """Scarica una singola serie da FRED (cache session + disk).

        Args:
            series_name: Nome interno (chiave in FRED_SERIES)
            start_date: Data inizio (default: 1990-01-01)
            end_date: Data fine (default: oggi)

        Returns:
            Serie pandas con dati raw
        """
        if series_name not in FRED_SERIES:
            raise ValueError(f"Serie sconosciuta: {series_name}. Disponibili: {list(FRED_SERIES.keys())}")

        start = start_date or date(1990, 1, 1)
        end = end_date or date.today()
        is_default_range = start_date is None and end_date is None
        cache_key = f"{series_name}_{start}_{end}"

        if cache_key in self._cache:
            return self._cache[cache_key]

        config = FRED_SERIES[series_name]
        fred_id = config["fred_id"]
        freq = config.get("frequency", "monthly")

        # Disk cache (solo per range di default per non sporcarla con fetch custom)
        if is_default_range:
            disk_data = self._load_disk_cache(series_name, freq)
            if disk_data is not None:
                self._cache[cache_key] = disk_data
                return disk_data

        logger.info(f"Fetching {fred_id} ({config['description']}) from {start} to {end}")

        # Retry con backoff per errori 500 transitori FRED
        last_err = None
        for attempt in range(3):
            try:
                data = self.fred.get_series(fred_id, observation_start=start, observation_end=end)
                result = data.dropna()
                self._cache[cache_key] = result
                if is_default_range:
                    self._save_disk_cache(series_name, result)
                return result
            except Exception as e:
                last_err = e
                msg = str(e).lower()
                # Re-raise subito su errori permanenti (400: series non esiste)
                if "bad request" in msg or "does not exist" in msg:
                    raise
                if attempt < 2:
                    time.sleep(1.5 * (attempt + 1))
        raise last_err  # type: ignore[misc]

    def fetch_and_transform(
        self,
        series_name: str,
        start_date: Optional[date] = None,
        end_date: Optional[date] = None,
    ) -> dict[str, pd.Series]:
        """Scarica e applica trasformazioni (ROC, z-score, YoY).

        Returns:
            Dict con chiavi: 'raw', 'roc_3m', 'roc_6m', 'roc_12m',
            'zscore_12m', 'zscore_36m'
        """
        # Cache session: le stesse trasformazioni vengono invocate più volte dalle
        # routine di preparazione (indicatori + history) — memoizziamo il risultato.
        transform_key = f"{series_name}_{start_date}_{end_date}"
        cached = self._transform_cache.get(transform_key)
        if cached is not None:
            return cached

        raw = self.fetch_series(series_name, start_date, end_date)
        config = FRED_SERIES[series_name]

        result = {"raw": raw}

        # Calcola ROC a vari orizzonti
        if config["frequency"] == "monthly":
            result["roc_3m"] = calculate_roc(raw, periods=3)
            result["roc_6m"] = calculate_roc(raw, periods=6)
            result["roc_12m"] = calculate_roc(raw, periods=12)
            result["zscore_12m"] = calculate_zscore(raw, window=12)
            result["zscore_36m"] = calculate_zscore(raw, window=36)
        elif config["frequency"] == "quarterly":
            result["roc_3m"] = calculate_roc(raw, periods=1)  # 1 trimestre = 3 mesi
            result["roc_6m"] = calculate_roc(raw, periods=2)
            result["roc_12m"] = calculate_roc(raw, periods=4)
            result["zscore_12m"] = calculate_zscore(raw, window=4)
            result["zscore_36m"] = calculate_zscore(raw, window=12)
        elif config["frequency"] == "annual":
            result["roc_3m"] = calculate_roc(raw, periods=1)  # 1 anno
            result["roc_6m"] = calculate_roc(raw, periods=1)
            result["roc_12m"] = calculate_roc(raw, periods=1)
            result["zscore_12m"] = calculate_zscore(raw, window=3)
            result["zscore_36m"] = calculate_zscore(raw, window=5)
        elif config["frequency"] in ("daily", "weekly"):
            result["roc_3m"] = calculate_roc(raw, periods=63)   # ~3 mesi trading days
            result["roc_6m"] = calculate_roc(raw, periods=126)
            result["roc_12m"] = calculate_roc(raw, periods=252)
            result["zscore_12m"] = calculate_zscore(raw, window=252)
            result["zscore_36m"] = calculate_zscore(raw, window=756)

        self._transform_cache[transform_key] = result
        return result

    def fetch_all_latest(self, max_workers: int = 20) -> dict[str, float]:
        """Scarica l'ultimo valore disponibile per tutte le serie in parallelo.

        Usa ThreadPoolExecutor per fare chiamate FRED concorrenti (I/O bound).
        Il disk cache riduce ulteriormente il lavoro per serie già aggiornate.

        Returns:
            Dict {series_name: latest_value}
        """
        latest: dict[str, float] = {}

        def _worker(name: str) -> tuple[str, Optional[float]]:
            try:
                data = self.fetch_series(name)
                if data.empty:
                    return name, None
                return name, float(data.iloc[-1])
            except Exception as e:
                logger.warning(f"Errore fetch {name}: {e}")
                return name, None

        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            futures = {pool.submit(_worker, name): name for name in FRED_SERIES}
            for fut in as_completed(futures):
                name, value = fut.result()
                if value is not None:
                    latest[name] = value
        return latest
