"""Debasement divergence signal — il segnale-firma di DREAM (Vito Lops).

Oro e tassi reali sono inversamente correlati (reali su -> oro giu). Su una finestra
ROLLING di 60 mesi si stima la relazione ln(oro) ~ a + b*tasso_reale e si proietta il
fair-value implicito dell'oro dato il tasso reale corrente. Il "gap" e' di quanto
l'oro attuale sta SOPRA (o sotto) quel fair value: un gap alto e persistente segnala
che il mercato prezza il *ciclo monetario* (debasement) piu' del *ciclo economico*.

Fonti riusate (OS-first):
  - oro spot: `YahooFetcher.fetch_asset("gold")` (GLD 2004+ stitchato a GC=F 2000+)
  - tasso reale: `FredFetcher.fetch_series("real_yield_10y")` (DFII10, TIPS 2003+)

Riproduzione live (2026-07): oro ~$4013 vs DFII10 2.35% -> gap +39.9%, fair ~$2907,
coerente col +39.5% / ~$2900 dichiarato nel video DREAM.

NB fedelta' storica: DFII10 parte dal 2003 e l'oro spot dal 2000 -> il picco 1980-81
citato nel video NON e' riproducibile con questi feed (servirebbe un proxy di tasso
reale DGS10-CPI + oro stitchato pre-1984). Questo modulo copre l'era moderna (live);
l'ancora storica resta un'illustrazione DA VERIFICARE, non load-bearing.

Vista, non trading: nessuna raccomandazione, solo misura informativa (spec R1 DREAM).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

WINDOW_MONTHS = 60


def _to_monthly(s: pd.Series) -> pd.Series:
    """Ultimo valore di ogni mese, indice datetime, senza NaN."""
    s = s.copy()
    s.index = pd.to_datetime(s.index)
    return s.resample("ME").last().dropna()


def compute_divergence(
    gold: pd.Series,
    real_yield: pd.Series,
    window: int = WINDOW_MONTHS,
) -> dict | None:
    """Gap oro vs fair-value implicito dai tassi reali (rolling `window` mesi).

    Per ogni mese t (dal warmup in poi) la relazione ln(oro)~a+b*reale e' stimata sui
    `window` mesi PRECEDENTI (finestra trailing esclusiva -> nessun future-leak: il
    fair value di t non usa l'oro di t), poi si proietta il fair value su reale(t).
    gap(t) = oro(t)/fair(t) - 1.

    Returns dict con l'ultimo punto + storia, oppure None se i dati non bastano
    (graceful: mai crash su serie corte / disallineate).
    """
    if gold is None or real_yield is None or len(gold) == 0 or len(real_yield) == 0:
        return None

    g = _to_monthly(gold)
    r = _to_monthly(real_yield)
    df = pd.concat([np.log(g).rename("lg"), r.rename("rr")], axis=1, join="inner").dropna()
    if len(df) < window + 1:
        return None

    lg = df["lg"].to_numpy()
    rr = df["rr"].to_numpy()
    dates = df.index

    history: list[dict] = []
    for i in range(window, len(df)):
        w_rr = rr[i - window:i]
        w_lg = lg[i - window:i]
        if np.ptp(w_rr) == 0:  # tassi reali costanti nella finestra -> fit degenere
            continue
        b1, b0 = np.polyfit(w_rr, w_lg, 1)
        pred_lg = b0 + b1 * rr[i]
        gap = float(np.exp(lg[i] - pred_lg) - 1.0)
        history.append(
            {
                "date": dates[i].strftime("%Y-%m-%d"),
                "gap": gap,
                "gold": float(np.exp(lg[i])),
                "fair_value": float(np.exp(pred_lg)),
                "real_yield": float(rr[i]),
            }
        )

    if not history:
        return None

    last = history[-1]
    return {
        "asof": last["date"],
        "gap": last["gap"],
        "gap_pct": last["gap"] * 100.0,
        "gold": last["gold"],
        "fair_value": last["fair_value"],
        "real_yield": last["real_yield"],
        "window": window,
        "n": len(df),
        "history": [{"date": h["date"], "gap": h["gap"]} for h in history],
    }


def load_live_divergence(window: int = WINDOW_MONTHS) -> dict | None:
    """Carica oro spot (Yahoo) + tasso reale (FRED DFII10) e calcola il segnale live.

    Graceful: se un feed esterno fallisce ritorna None (mai 500), coerente con la
    policy di degradazione del progetto.
    """
    try:
        from app.services.indicators.fetcher import FredFetcher
        from app.services.prices.yahoo_fetcher import YahooFetcher

        gold = YahooFetcher().fetch_asset("gold")
        real_yield = FredFetcher().fetch_series("real_yield_10y")
    except Exception:
        return None
    return compute_divergence(gold, real_yield, window=window)
