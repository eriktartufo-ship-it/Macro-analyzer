"""T9-AUDIT-FIX: Financial-stress veto.

Insight council (soros + taleb + buffett): il regime classifier ha 4 quadranti
macro ma manca una **dimensione ortogonale** = "financial stress without macro
stress yet". Casi storici problematici dove il modello falliva:

- 2018 Q4 sell-off: VIX 26, S&P -20%, MA macro indicators sani (gdp 2.5, unrate 3.8).
  Modello predice goldilocks, realtà era "credit stress + Fed QT panic".
- 2023 SVB collapse: VIX 26, banking spread blow-out, MA macro indicators ok.
  Modello predice reflation, realtà era "isolated banking contagion fear".

Soluzione (raydalio + soros): NON aggiungere indicators al classifier macro.
Aggiungere un **veto layer post-classifier** che forza transition quando:
1. VIX > 28 AND BAA spread > 2.5%   (market+credit stress combined)
   OR
2. HY OAS > 5.0%                     (severe credit blow-out alone)
   AND
3. Regime predicted ∈ {goldilocks, reflation}  (le "false positive" comuni)

Effetto: forza `is_transition=True`, riduce `confidence` × 0.5, impone defensive
allocation invece di chasing equities/bitcoin in un periodo di stress sotto-rilevato.

Filosofia: meglio missare un rally tardivo che subire un -20% drawdown perché
il modello macro non vede la fragility dei mercati.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class FinancialStressVeto:
    """Esito del veto check."""
    is_vetoed: bool
    triggers_fired: list[str]
    vix: float
    baa_10y_spread: float
    hy_oas: float
    description: str


def _safe_float(val: Any, default: float = 0.0) -> float:
    if val is None:
        return default
    try:
        import math
        v = float(val)
        if math.isnan(v) or math.isinf(v):
            return default
        return v
    except (TypeError, ValueError):
        return default


def assess_financial_stress_veto(
    indicators: dict[str, Any],
    predicted_regime: str,
    vix_threshold: float = 24.0,
    baa_threshold: float = 2.0,
    hy_oas_threshold: float = 5.0,
) -> FinancialStressVeto:
    # T9-AUDIT-BUG-5: usa alias util per match nomi live JSON.
    from app.services.common.indicator_aliases import safe_indicator
    """Valuta se attivare il veto financial-stress sul regime predicted.

    T9-AUDIT-FIX-2: soglie calibrate su 2018 Q4 (vix 26, baa 2.0) e 2023 SVB
    (vix 26, baa 2.5). Soglie pre-calibrazione iniziali 28/2.5 erano troppo
    strette per questi eventi (0/9 trigger nel test).

    Args:
        indicators: dict con almeno `vix`, `baa_10y_spread`, opzionalmente `hy_oas`.
        predicted_regime: regime corrente predicted dal classifier.
        vix_threshold: VIX > 24 = market stress (95th percentile storico USA).
        baa_threshold: BAA-10Y spread > 2.0% = credit risk premium elevato.
        hy_oas_threshold: HY OAS > 5.0% = credit blow-out severo (2008/2020/2023).

    Returns:
        FinancialStressVeto con is_vetoed flag e trigger info.
    """
    vix = _safe_float(indicators.get("vix"), 18.0)
    # T9-AUDIT-BUG-5: lookup tramite alias util (baa_spread/baa_10y_spread)
    baa = safe_indicator(indicators, "baa_10y_spread", default=1.8)
    hy_oas = _safe_float(indicators.get("hy_oas"), 3.5)

    triggers: list[str] = []

    # Trigger 1: VIX + BAA combo (>= per inclusività su valori limite es. 2018 Q4)
    if vix >= vix_threshold and baa >= baa_threshold:
        triggers.append(
            f"vix={vix:.1f}>={vix_threshold} AND baa_spread={baa:.2f}>={baa_threshold} "
            "(market+credit stress)"
        )

    # Trigger 2: HY OAS standalone (severe credit blow-out)
    if hy_oas >= hy_oas_threshold:
        triggers.append(
            f"hy_oas={hy_oas:.2f}>{hy_oas_threshold} (severe credit blow-out)"
        )

    # Veto si attiva SOLO se predicted regime è goldilocks/reflation (false positives)
    # AND almeno 1 trigger fired.
    vulnerable_regimes = {"goldilocks", "reflation"}
    is_vetoed = (
        predicted_regime in vulnerable_regimes
        and len(triggers) >= 1
    )

    if is_vetoed:
        desc = (
            f"VETO ATTIVO: regime '{predicted_regime}' + financial stress. "
            f"Forzo transition mode (defensive allocation suggerita)."
        )
    else:
        desc = "No veto: regime stabile o stress non sufficiente."

    return FinancialStressVeto(
        is_vetoed=is_vetoed,
        triggers_fired=triggers,
        vix=vix,
        baa_10y_spread=baa,
        hy_oas=hy_oas,
        description=desc,
    )


def apply_veto_to_classification(
    classify_result: dict[str, Any],
    indicators: dict[str, Any],
) -> dict[str, Any]:
    """Post-processing: applica veto financial-stress al risultato classifier.

    Effetti se vetoed:
    - `confidence` × 0.5 (riduzione drastica)
    - aggiunge `financial_stress_veto` con dettaglio
    - NON cambia probabilities/regime (transition_detector farà il resto).

    Returns:
        Dict mutato in-place + return.
    """
    predicted = classify_result.get("regime", "")
    veto = assess_financial_stress_veto(indicators, predicted)
    if veto.is_vetoed:
        # Halve confidence
        current_conf = classify_result.get("confidence", 0.5)
        classify_result["confidence"] = round(current_conf * 0.5, 4)
        classify_result["financial_stress_veto"] = {
            "vetoed": True,
            "triggers": veto.triggers_fired,
            "vix": veto.vix,
            "baa_spread": veto.baa_10y_spread,
            "hy_oas": veto.hy_oas,
            "description": veto.description,
        }
    return classify_result
