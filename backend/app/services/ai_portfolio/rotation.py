"""Segnale di ROTAZIONE cross-asset (dual momentum) — funzioni PURE.

Nasce dalla richiesta Erik 2026-07-15: "trovare il metodo di tracciare i soldi che
si spostano... se vanno via dall'S&P500 magari si stanno spostando da qualche altra
parte e quindi bisogna catturare lo spostamento per ottenere un alpha sul S&P".

⚠️ ONESTÀ SULLA PREMESSA: i "soldi" NON si spostano come un fluido conservato. In
borsa ogni compratore ha un venditore: la capitalizzazione sale perché sale il
prezzo marginale, non perché un secchio si riempie. "Quanti soldi sono usciti
dall'S&P" NON è misurabile così, e i flussi VERI dei fondi (EPFR/ICI/Lipper) sono
dati a pagamento che non abbiamo.
Ciò che è misurabile — ed è il nucleo CORRETTO e tradeable dell'intuizione — è la
**rotazione relativa**: quando gli allocatori riducono l'azionario e aumentano
oro/bond, lo si legge nella PERFORMANCE RELATIVA. La forza relativa è l'ombra
osservabile dei flussi.

Formalizzazione = **dual momentum** (Antonacci), che mappa 1:1 la richiesta:
  - momentum RELATIVO  → "dove stanno andando i soldi": compra i più forti;
  - momentum ASSOLUTO  → "smorzare le discese": se nemmeno il più forte batte il
    cash, stai in cash (è il filtro che evita i bear).

NESSUN aggancio al live: modulo puro, usato dal backtest. Diventa un bot SOLO se
il Δ regge sui dati reali (regola: backtest prima di creare una strategia).

═══════════════════════════════════════════════════════════════════════════════
VERDETTO MISURATO (`scripts/rotation_edge.py`, 5 seed × 40 sim, dati reali,
costi 10bps, 2026-07-15) — LEGGERE PRIMA DI PROMUOVERLO A BOT:

              | α vs S&P | DD rotazione | DD S&P
  broad       |  −7.4pp  |   −11.6%     | −19.3%
  2008        | **+10.2pp** (rebal-5: +18.9) | −26.5% | −42.0%
  2020        | −13.4pp  |   −18.7%     | −19.1%
  2022        |  −7.4pp  |   −21.2%     | −22.0%

1. **NON batte l'S&P** (−7.4pp/anno broad): diluire su 24 asset rinuncia per
   costruzione al beta azionario → non può battere un benchmark pure-equity in un
   campione bull-heavy. Vince solo nel bear LUNGO (2008). Batte però il 60/40 in
   tutte e 4 le finestre → è un'alternativa al 60/40, non un ammazza-S&P.
2. **Dimezza il drawdown**: reale e robusto (broad −11.6 vs −19.3; 2008 −26.5 vs −42).
   È l'ennesima conferma della legge del progetto: protezione del DD pagata in alpha.
3. ⚠️ **`use_absolute_momentum` è un DANNO NETTO**: broad −7.4 (ON) vs −3.4 (OFF);
   e nel 2008 — la crisi per cui esiste — dà lo STESSO risultato (+10.2 vs +10.2).
   Non protegge mai davvero e costa ~4pp. Default lasciato True solo per fedeltà al
   dual-momentum canonico: se questo modulo verrà promosso, **partire da OFF**.
═══════════════════════════════════════════════════════════════════════════════
"""
from __future__ import annotations

from typing import Optional

import pandas as pd


def trailing_return_at(
    matrix: pd.DataFrame, asset: str, day, lookback: int,
) -> Optional[float]:
    """Rendimento cumulato di `asset` sugli ultimi `lookback` giorni PRIMA di `day`.

    NO FUTURE LEAK: usa solo righe con indice < day.
    """
    if asset not in matrix.columns:
        return None
    hist = matrix.loc[matrix.index < day, asset].dropna().tail(lookback)
    if len(hist) < max(5, lookback // 2):
        return None
    return float((1.0 + hist).prod() - 1.0)


def momentum_score(
    matrix: pd.DataFrame, asset: str, day, lookbacks: tuple[int, ...] = (63, 126),
) -> Optional[float]:
    """Score di momentum = media dei rendimenti trailing su più orizzonti.

    Il blend (3m+6m di default) è meno fragile del singolo lookback: un solo
    orizzonte è la classica leva da overfitting.
    """
    vals = [trailing_return_at(matrix, asset, day, lb) for lb in lookbacks]
    vals = [v for v in vals if v is not None]
    if not vals:
        return None
    return sum(vals) / len(vals)


def rank_assets(
    matrix: pd.DataFrame, day, assets: list[str],
    lookbacks: tuple[int, ...] = (63, 126),
) -> list[tuple[str, float]]:
    """Asset ordinati per momentum decrescente (i più forti primi).

    È il "dove stanno andando i soldi": chi sovraperforma sta attirando capitale.
    """
    scored = []
    for a in assets:
        s = momentum_score(matrix, a, day, lookbacks)
        if s is not None:
            scored.append((a, s))
    scored.sort(key=lambda x: -x[1])
    return scored


def select_rotation_portfolio(
    matrix: pd.DataFrame, day, assets: list[str], top_k: int = 3,
    lookbacks: tuple[int, ...] = (63, 126), cash_asset: str = "cash_money_market",
    use_absolute_momentum: bool = True,
) -> dict[str, float]:
    """Portafoglio di rotazione equipesato sui top-K, con filtro absolute-momentum.

    Args:
        top_k: quanti asset tenere (i più forti).
        use_absolute_momentum: se True, un asset entra SOLO se il suo momentum
            batte quello del cash; il peso scartato va in cash. È il meccanismo
            che smorza le discese: nei bear generalizzati nessuno batte il cash
            → il portafoglio si sposta in liquidità da solo.

    Returns:
        dict {asset: peso}, Σ = 1.0 (il residuo è sempre in cash).
    """
    ranked = [(a, s) for a, s in rank_assets(matrix, day, assets, lookbacks)
              if a != cash_asset]
    if not ranked:
        return {cash_asset: 1.0}

    cash_mom = momentum_score(matrix, cash_asset, day, lookbacks)
    if cash_mom is None:
        cash_mom = 0.0

    picks = ranked[:top_k]
    if use_absolute_momentum:
        picks = [(a, s) for a, s in picks if s > cash_mom]

    if not picks:
        return {cash_asset: 1.0}  # nessuno batte il cash → tutto liquido

    w = 1.0 / top_k  # peso fisso: se meno di K passano il filtro, il resto va in cash
    out = {a: w for a, _ in picks}
    residual = 1.0 - w * len(picks)
    if residual > 1e-9:
        out[cash_asset] = out.get(cash_asset, 0.0) + residual
    return out
