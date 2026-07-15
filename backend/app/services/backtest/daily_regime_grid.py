"""B2 (S41) — Regime su griglia settimanale, cache su disco, lookup daily ffill.

Perché settimanale: `classify_at` costa ~290ms/chiamata → daily puro (~7.5k giorni)
= ~37 min a run. Ma il regime è quasi piecewise-costante fra giorni consecutivi
(guidato da FRED mensile/settimanale; VIX/yield spostano poco). Precompute sui
venerdì (~1.5k punti su 30 anni ≈ 7 min UNA VOLTA, poi cache su disco → istantaneo),
e lookup daily = ultima griglia con data ≤ giorno (ffill).

NO FUTURE LEAK: `get(day)` usa searchsorted side="right"-1 → prende solo la
classificazione del venerdì PRECEDENTE o uguale, mai una futura.
"""
from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import Callable, Optional

import pandas as pd
from loguru import logger

_REGIMES = ("reflation", "stagflation", "deflation", "goldilocks")
_DEFAULT_CACHE = Path(__file__).resolve().parents[3] / ".cache" / "backtest" / "weekly_regime_grid.parquet"


class WeeklyRegimeGrid:
    """Griglia settimanale di classificazioni con lookup daily ffill (no leak)."""

    def __init__(self, frame: pd.DataFrame):
        # frame: index = grid dates (Timestamp), cols = regime, confidence, p_<regime>
        self._frame = frame.sort_index()
        self._dates = self._frame.index

    def get(self, day) -> Optional[dict]:
        """Ritorna la classificazione dell'ultima griglia con data ≤ day (ffill).

        None se `day` precede la prima data di griglia. Nessun lookahead.
        """
        if len(self._dates) == 0:
            return None
        d = pd.Timestamp(day)
        pos = self._dates.searchsorted(d, side="right") - 1
        if pos < 0:
            return None
        row = self._frame.iloc[pos]
        return {
            "regime": row["regime"],
            "confidence": float(row["confidence"]),
            "probs": {r: float(row[f"p_{r}"]) for r in _REGIMES},
            "grid_date": self._dates[pos],
        }

    @property
    def frame(self) -> pd.DataFrame:
        return self._frame

    def __len__(self) -> int:
        return len(self._frame)


def _grid_fridays(start: date, end: date) -> pd.DatetimeIndex:
    """Venerdì (W-FRI) nella finestra [start, end]."""
    return pd.date_range(pd.Timestamp(start), pd.Timestamp(end), freq="W-FRI")


def build_weekly_grid_from_fn(
    dates: pd.DatetimeIndex,
    classify_fn: Callable[[date], Optional[dict]],
) -> pd.DataFrame:
    """Costruisce il frame griglia chiamando `classify_fn(as_of)` per ogni data.

    Pura/testabile (nessuna dipendenza FRED): `classify_fn` ritorna il cm dict
    (probs+confidence+regime) o None. Le date con None sono saltate.
    """
    rows = []
    for d in dates:
        cm = classify_fn(d.date() if hasattr(d, "date") else d)
        if cm is None:
            continue
        probs = cm.get("probs", {})
        rows.append({
            "date": pd.Timestamp(d),
            "regime": cm.get("regime"),
            "confidence": float(cm.get("confidence", 0.5) or 0.5),
            **{f"p_{r}": float(probs.get(r, 0.0)) for r in _REGIMES},
        })
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).set_index("date").sort_index()


def build_weekly_grid(
    series: dict,
    start: date,
    end: date,
    use_cache: bool = True,
    cache_path: Path = _DEFAULT_CACHE,
) -> WeeklyRegimeGrid:
    """Griglia settimanale su [start, end] usando classify_at(series, as_of).

    Cache incrementale su disco: carica il parquet esistente, calcola SOLO i
    venerdì mancanti nella finestra, unisce e risalva.
    """
    from app.services.backtest.walk_forward import classify_at

    want = _grid_fridays(start, end)
    cached = pd.DataFrame()
    if use_cache and cache_path.exists():
        try:
            cached = pd.read_parquet(cache_path)
            cached.index = pd.to_datetime(cached.index)
        except Exception as e:
            logger.warning(f"Weekly grid cache read failed: {e}")

    have = set(cached.index) if not cached.empty else set()
    missing = [d for d in want if d not in have]
    if missing:
        logger.info(f"Weekly grid: calcolo {len(missing)} venerdì mancanti "
                    f"(cache ha {len(have)})")
        fresh = build_weekly_grid_from_fn(
            pd.DatetimeIndex(missing),
            lambda d: classify_at(series, d),
        )
        combined = pd.concat([cached, fresh]).sort_index()
        combined = combined[~combined.index.duplicated(keep="last")]
        if use_cache:
            try:
                cache_path.parent.mkdir(parents=True, exist_ok=True)
                combined.to_parquet(cache_path)
                logger.info(f"Saved weekly regime grid: {combined.shape} -> {cache_path}")
            except Exception as e:
                logger.warning(f"Weekly grid cache write failed: {e}")
    else:
        combined = cached

    # Restituisci la griglia limitata alla finestra richiesta (con un margine a sx
    # per garantire ffill all'inizio finestra).
    in_win = combined[(combined.index >= pd.Timestamp(start) - pd.Timedelta(days=10))
                      & (combined.index <= pd.Timestamp(end))]
    return WeeklyRegimeGrid(in_win)
