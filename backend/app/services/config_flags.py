"""Feature flags via env var, centralizzati per consistenza.

Pattern: tutti gli scoring/regime layer "biased" sono OPT-IN. Default = data-driven puro.
Cosi' i numeri base sono trasparenti e l'utente puo' attivare layer aggiuntivi
(calibration shrinkage, dedollar bias) sapendo che modificano il risultato.
"""

from __future__ import annotations

import os

_TRUTHY = ("1", "true", "yes", "on")

# Runtime override (process-local). Settato via POST /config/flags endpoint
# per A/B testing live senza restart backend. Lookup order: runtime > env > default.
_RUNTIME_FLAGS: dict[str, bool] = {}


def _read_flag(env_name: str, default: bool = False) -> bool:
    """Lookup: runtime > env var > default."""
    if env_name in _RUNTIME_FLAGS:
        return _RUNTIME_FLAGS[env_name]
    raw = os.getenv(env_name)
    if raw is None:
        return default
    return raw.lower() in _TRUTHY


def set_runtime_flag(env_name: str, value: bool | None, persist: bool = True) -> None:
    """Set/clear runtime override. None = clear (back to env var).

    Thread-safe enough per single-process FastAPI (GIL).

    Args:
        env_name: nome del flag.
        value: True/False per attivare/disattivare, None per clear.
        persist: se True (default), salva su DB `runtime_flag_overrides` per
            sopravvivere a docker compose restart. Pass False per cambi
            temporanei (es. test, batch).
    """
    if value is None:
        _RUNTIME_FLAGS.pop(env_name, None)
    else:
        _RUNTIME_FLAGS[env_name] = bool(value)

    if persist:
        try:
            _persist_runtime_flag_to_db(env_name, value)
        except Exception:
            # Best-effort, non bloccare se DB non disponibile
            pass


def _persist_runtime_flag_to_db(env_name: str, value: bool | None) -> None:
    """T11+: salva override in DB per sopravvivenza restart."""
    from app.database import SessionLocal
    from app.models import RuntimeFlagOverride

    with SessionLocal() as session:
        existing = session.query(RuntimeFlagOverride).filter_by(flag_name=env_name).first()
        if value is None:
            if existing is not None:
                session.delete(existing)
        else:
            if existing is not None:
                existing.value = bool(value)
            else:
                session.add(RuntimeFlagOverride(flag_name=env_name, value=bool(value)))
        session.commit()


def load_runtime_flags_from_db() -> int:
    """T11+: ricarica runtime overrides dal DB allo startup app.

    Chiamato da `app.main.lifespan`. Ritorna numero flag caricati.
    Best-effort (se DB non pronto / table non esiste, ritorna 0 silenziosamente).
    """
    try:
        from app.database import SessionLocal
        from app.models import RuntimeFlagOverride

        with SessionLocal() as session:
            rows = session.query(RuntimeFlagOverride).all()
            for row in rows:
                _RUNTIME_FLAGS[row.flag_name] = row.value
            return len(rows)
    except Exception:
        return 0


def get_all_flags_state() -> dict[str, dict]:
    """Snapshot stato di TUTTI i flag noti per /config/flags GET.

    Returns: {flag_name: {value: bool, source: "runtime"|"env"|"default"}}

    T10b 2026-05-17: USE_CALIBRATED_SCORING, USE_GDP_COLLAPSE_OVERRIDE,
    USE_ML_REGIME_BLEND promossi default-True. Source "default" ora riflette
    il default *reale* del flag, non hardcoded False.
    """
    # Defaults reali per ogni flag (matchano i default delle use_*() funzioni)
    defaults: dict[str, bool] = {
        "USE_CALIBRATED_SCORING": True,         # T10b promoted
        "USE_GDP_COLLAPSE_OVERRIDE": True,      # T10b promoted
        "USE_ML_REGIME_BLEND": True,            # T10b promoted
        "USE_UNCERTAINTY_GATE": True,           # Paper trading promoted 2026-05-27
        "USE_MOMENTUM_PILLARS": True,           # Paper trading 4-test promoted 2026-05-27
        "USE_LIQUIDITY_SURGE_OVERRIDE": True,   # Council post-TEST A 2020 disaster 2026-05-27
        "USE_FORWARD_INFLATION_PILLAR": True,   # T13 Council 2026-05-31: MICH/sticky/T5YIFR
        "USE_LABOR_CREDIT_PILLAR": True,        # T17 Council 2026-05-31 sessione 23: wage/quits/hours/HY spread
        "USE_CYCLICAL_PILLAR": True,            # T18 Council 2026-05-31 sessione 25: heavy truck/permits/credit/durable
        "USE_NOWCAST_PILLAR": False,            # T19 council expected +1pp but validation -0.22pp drag, demoted opt-in
        "USE_AI_PORTFOLIO_CASH_OVERNIGHT": True, # 2026-07-13: cash idle horserace rende overnight (BIL/€STR) non 0%
        "USE_AI_PORTFOLIO_DEPLOY_BOOST": True,  # 2026-07-13: combo deadband+fastramp+floor PROMOSSO (walk-forward 2008/2020/all_crisis: α +1.2/+7.9/+2.2pp, severe 2020 30→11)
    }
    # Add new flags to known list below
    pass  # marker
    known = [
        "USE_CALIBRATED_SCORING",
        "USE_DEDOLLAR_BONUS",
        "USE_DEDOLLAR_PILLAR",
        "USE_CRISIS_MODULATION",
        "USE_NEWS_PILLAR",
        "USE_DFM_ASSET_BONUS",
        # "USE_ASSET_FEEDBACK" rimosso 2026-05-28 (dead code)
        "USE_ADAPTIVE_THRESHOLDS",
        # "USE_CORRELATION_REGIME" rimosso 2026-05-28
        "USE_CROSS_ASSET_PILLARS",
        "USE_LIVE_CALIBRATION",
        "USE_POSITION_SIZING_LAYER",
        "USE_VOL_TARGETING",
        "USE_DISCRETIONARY_OVERRIDES",
        "USE_FRESHNESS_WEIGHTING",
        # "USE_SUB_REGIMES" rimosso 2026-05-28
        # "USE_REGIME_2D" rimosso 2026-05-28 (dead)
        "USE_GDP_COLLAPSE_OVERRIDE",
        # "USE_RANK_PERCENTILE_SCORING" rimosso 2026-05-28 (council KILL)
        # "USE_DOWNSIDE_PROTECTION_BONUS" rimosso 2026-05-28
        "USE_ML_REGIME_BLEND",
        "USE_DEFENSIVE_TRANSITION_MODE",
        "USE_FINANCIAL_STRESS_VETO",
        # "USE_CONDITIONAL_ASSET_SCORING" rimosso 2026-05-28
        "USE_MOMENTUM_PILLARS",
        "USE_UNCERTAINTY_GATE",
        "USE_LIQUIDITY_SURGE_OVERRIDE",
        "USE_FORWARD_INFLATION_PILLAR",
        "USE_LABOR_CREDIT_PILLAR",
        "USE_CYCLICAL_PILLAR",
        "USE_NOWCAST_PILLAR",
        "USE_AI_PORTFOLIO_CASH_OVERNIGHT",
        "USE_AI_PORTFOLIO_DEPLOY_BOOST",
    ]
    out = {}
    for name in known:
        if name in _RUNTIME_FLAGS:
            out[name] = {"value": _RUNTIME_FLAGS[name], "source": "runtime"}
        elif os.getenv(name) is not None:
            out[name] = {"value": _read_flag(name), "source": "env"}
        else:
            out[name] = {"value": defaults.get(name, False), "source": "default"}
    return out


# Flag che NON possono cambiare l'output del classifier: vivono interamente in
# `services/ai_portfolio/` (verificato via grep S42: zero occorrenze in regime/).
# Escluderli evita di invalidare una griglia di regimi quando cambia solo una leva
# di portafoglio — l'harness daily forza deploy_boost=0 mentre il live lo ha a 1.
# Tutto il resto è INCLUSO per default: un flag nuovo entra nel fingerprint da solo.
_NON_CLASSIFIER_FLAGS: frozenset[str] = frozenset({
    "use_ai_portfolio_cash_overnight",
    "use_ai_portfolio_deploy_boost",
})


def classifier_flags_fingerprint() -> str:
    """Hash corto della configurazione che può cambiare l'output del classifier.

    Serve come CHIAVE DI CACHE per tutto ciò che memorizza classificazioni di
    regime (griglia settimanale su disco, WalkForwardCache in memoria).

    S42 — perché esiste: la griglia settimanale era cachata per SOLA DATA. Cambiando
    un flag del classifier, i venerdì risultavano "già calcolati" e si riceveva la
    classificazione di un ALTRO set di flag, in silenzio → ogni A/B su una feature
    del classifier misurava Δ=0 (falso negativo indistinguibile da un vero zero).
    Stessa famiglia del bug cache Yahoo di S41b: *chiave di cache incompleta*.

    Fail-safe per costruzione: i flag sono enumerati per INTROSPEZIONE delle
    `use_*()` di questo modulo, non da una lista a mano. Aggiungere un flag nuovo
    lo include automaticamente. (Le liste a mano si disallineano: quelle di
    `get_all_flags_state()` hanno già perso USE_LEVEL_ACCEL_PILLARS.)
    """
    import hashlib
    import inspect
    import sys

    module = sys.modules[__name__]
    parts: list[str] = []
    for name, fn in inspect.getmembers(module, inspect.isfunction):
        if not name.startswith("use_") or name in _NON_CLASSIFIER_FLAGS:
            continue
        if inspect.signature(fn).parameters:
            continue  # non è un getter di flag semplice
        try:
            parts.append(f"{name}={int(bool(fn()))}")
        except Exception:  # pragma: no cover - un getter rotto non deve rompere la cache
            parts.append(f"{name}=?")
    digest = hashlib.sha256("|".join(sorted(parts)).encode()).hexdigest()
    return digest[:12]


def use_calibrated_scoring() -> bool:
    """Se True, scoring engine usa calibrated_asset_regime.json (Bayesian shrinkage).

    **T10b 2026-05-17 PROMOTED TO DEFAULT ON**: validation walk-forward 15 sim
    real-prices: Sortino delta vs baseline **+0.881** (massivo), Win rate
    11/15 (vs baseline 6/15), alpha **+14%** (vs baseline -42%). SOLO flag
    scoring-side che genera valore real-prices senza data leakage. Disable
    via `USE_CALIBRATED_SCORING=0` se serve baseline puro per debug.
    """
    return _read_flag("USE_CALIBRATED_SCORING", default=True)


def use_dedollar_bonus() -> bool:
    """Se True, secular_bonus dedollar viene applicato a final asset scores
    e DEDOLLAR_REGIME_PRESSURE viene applicato in trajectory.
    Default: False (scoring/trajectory data-driven puri).
    """
    return _read_flag("USE_DEDOLLAR_BONUS")


def use_dedollar_pillar() -> bool:
    """Tier 5.2 LOOP CLOSURE: se True, dedollar_combined entra come pillar
    nel rule_based classifier (`stagflation.dedollar_debasement` weight 0.03,
    `reflation.dedollar_low` weight 0.02). Default: False (back-compat puro).

    Indipendente da `use_dedollar_bonus` (che controlla scoring asset).
    Attivabile separatamente per A/B testing dei due loop.
    """
    return _read_flag("USE_DEDOLLAR_PILLAR")


def use_crisis_modulation() -> bool:
    """Tier 5.1 LOOP CLOSURE: se True, le probabilities regime vengono
    modulate post-blend in base al crisis_type + risk_score di
    `assess_crisis_risk`. Default: False.

    Effetto:
    - `deflation_crash` con risk>0.3 → +deflation prob (alpha × risk).
    - `stagflation_debasement` con risk>0.3 → +stagflation prob.
    - `bubble_goldilocks` con risk>0.3 → small shift verso deflation/stag
      (preludio mean-reversion downside).
    - alpha=0.05 default = max ±5pp shift.

    Distinto da `use_dedollar_pillar` (che agisce sui fit_scores dentro
    classifier). Crisis modulation è POST-classifier, sui probs finali.
    """
    return _read_flag("USE_CRISIS_MODULATION")


def use_news_pillar() -> bool:
    """Tier 5.3 LOOP CLOSURE: se True, news avg sentiment entra come pillar
    nel rule_based classifier (`deflation.news_panic` weight 0.02,
    `reflation.news_positive_strong` weight 0.02). Default: False.

    News sono oggi consumate solo da trajectory (forces) + scoring asset
    (news_signals). Questo flag chiude il loop news → regime ufficiale.

    Pillar è soft (weight 0.02) per gestire rumorosità news intraday.
    """
    return _read_flag("USE_NEWS_PILLAR")


def use_dfm_asset_bonus() -> bool:
    """Tier 5.4 LOOP CLOSURE: se True, gdp_yoy_dfm modula direttamente
    final asset scores (oltre al pillar classifier già attivo).
    Default: False.

    Bonus applicato a:
    - us_equities_growth/value, em_equities: +δ × max(0, gdp_yoy_dfm - 2.5)
      (real-time growth nowcast = pro-cyclical assets bid)
    - us_bonds_long: -δ × max(0, gdp_yoy_dfm - 2.5) (duration risk se yields salgono)
    - Simmetrico negativo se gdp_yoy_dfm < 1.5 (slowdown nowcast).

    δ = 2.0 default → max ±5pp score boost con gdp_yoy_dfm estremo (~5%).

    **T9-AUDIT P0 (council 2026-05-16)**: DFM nowcast NON VALIDATO (solo 2
    records in DB con gdp_yoy_dfm popolato). Validation `dfm_validate.py`
    richiede ≥12 pairs. Council taleb+buffett: "feature non validata in
    produzione è una bomba". **Sconsigliato in production finché backfill
    storico ≥30 records non popolato + LOO R² ≥0.30**.
    """
    return _read_flag("USE_DFM_ASSET_BONUS")


def use_vol_targeting() -> bool:
    """Tier 6.10 VOL TARGETING: se True, scala weights post position sizing
    per matchare target annualized portfolio vol (default 10%).

    Math:
        realized_vol = sqrt(Σ w_i² σ_i² + correlation_terms)
        scale = target_vol / realized_vol
        weights_scaled = weights * scale

    Se `scale > 1`: leverage required (margin). Se `scale < 1`: cash buffer.

    Default vol_target = 10% annualized (conservative). Avg correlation
    assumed 0.3 (Bridgewater empirico) per stima realistic.

    RISK ALTO: leverage = rischio reale. Solo simulazione, NON live trading.
    Default: False.
    """
    return _read_flag("USE_VOL_TARGETING")


def use_position_sizing_layer() -> bool:
    """Tier 6.4 POSITION SIZING LAYER: se True, espone endpoints
    `/portfolio/allocation` con 4 algoritmi di portfolio construction.

    Algoritmi:
    1. equal_weight: top-N asset equipesati (baseline trivial)
    2. score_weighted: weight ∝ score / sum(scores_top_N)
    3. risk_parity: weight ∝ 1/vol_asset normalizzato (Bridgewater AW pattern)
    4. kelly_fractional: weight ∝ score² / vol² fractional (max 25%)

    Black-Litterman view-blending → TODO follow-up (richiede market cap weights).

    Output uniforme: PortfolioAllocation dataclass {asset: weight, sum=1.0}.

    Default: False (back-compat — endpoint disponibile solo opt-in).
    """
    return _read_flag("USE_POSITION_SIZING_LAYER")


def use_live_calibration() -> bool:
    """Tier 6.6 LIVE CALIBRATION: se True, `_calibrated_or_prior()` legge
    calibration da DB table `asset_regime_calibrations` (auto-aggiornata
    mensile dal scheduler) invece di `seed/calibrated_asset_regime.json`.

    Differenze rispetto a `USE_CALIBRATED_SCORING` (T1 legacy):
    - JSON statico → DB versionato (history snapshot).
    - Refresh manuale → job scheduler mensile auto.
    - Singolo file → tabella query-able (audit trail).

    Indipendente da `USE_CALIBRATED_SCORING`: se entrambi ON, DB prevale.

    Default: False (back-compat: usa hardcoded prior in ASSET_REGIME_DATA).
    """
    return _read_flag("USE_LIVE_CALIBRATION")


def use_cross_asset_pillars() -> bool:
    """Tier 6.3 CROSS-ASSET SIGNALS: se True, ratio cross-asset entrano
    come pillar classifier (peso 0.02-0.03 each).

    Bridgewater pattern: relazioni cross-asset come INPUT al regime
    (non solo output analitico). 3 ratio prioritari:

    - copper/gold: cyclical (Cu) vs safe-haven (Au). Alto → reflation,
      basso → deflation. Storica: 2009 bottom 0.20, 2011 peak 0.55.
    - gold/oil: gold strong vs oil weak = deflation. Storica avg ~17,
      2020-Q1 picco 75, 1973 oil shock minimo 9.
    - HY/IG spread ratio: HY_OAS / IG_OAS = credit risk appetite.
      Alto → deflation, basso → reflation.

    6 condition classifier opt-in via questo flag. Default: False.
    """
    return _read_flag("USE_CROSS_ASSET_PILLARS")


# use_correlation_regime() rimosso 2026-05-28 (code audit: module deleted)


def use_adaptive_thresholds() -> bool:
    """Tier 6.1 ADAPTIVE THRESHOLDS: se True, le condizioni del classifier
    per VIX/BAA/CPI/yield_curve/NFCI usano center+scale calcolati come
    quantili rolling 10y, invece di valori hardcoded.

    Filosofia: "VIX > 30" nel 1995 != "VIX > 30" nel 2024. Threshold
    relativi al regime storico recente catturano meglio i regime shift.

    Indicatori NON adattivizzati (anchored a target fissi):
    - fed_funds (Fed target esplicito)
    - breakeven_10y (Fed inflation target 2%)
    - unrate (NAIRU ~4%)

    Default: False (back-compat puro con T1-T5).
    """
    return _read_flag("USE_ADAPTIVE_THRESHOLDS")


def use_discretionary_overrides() -> bool:
    """Tier 6.7 DISCRETIONARY OVERRIDE LAYER: se True, applica override
    discrezionali attivi (da DB table `discretionary_overrides`) al scoring
    asset e/o alle probabilities regime.

    Schema override: (target_type, target_name, delta, expires_at, reason,
    author). Solo overrides con `expires_at > now` sono applicati.

    Default: False (back-compat, 100% mechanical).
    """
    return _read_flag("USE_DISCRETIONARY_OVERRIDES")


def use_freshness_weighting() -> bool:
    """Tier 6.9 FRESHNESS WEIGHTING: se True, ogni condition del classifier
    è scalata per `freshness_score` dell'indicator (0-1).

    Filosofia: GDP quarterly (lag ~60d) NON è informativa quanto VIX daily.
    `effective_weight = base_weight × freshness_score`. Indicatori freschi
    pesano di più, stale meno.

    Default freshness map:
    - daily (VIX, BAA, fed_funds, yield_curve): 1.0
    - weekly (claims): 0.9
    - monthly (CPI, unrate, PMI, etc.): 0.7-0.8
    - quarterly (GDP, GDI): 0.4

    Default: False (back-compat puro).
    """
    return _read_flag("USE_FRESHNESS_WEIGHTING")


# use_sub_regimes() rimosso 2026-05-28 (code audit: module deleted)


# use_regime_2d() rimosso 2026-05-28 (code audit: module deleted, no frontend consumer)


def use_gdp_collapse_override() -> bool:
    """Tier 7.1 GDP COLLAPSE OVERRIDE: se True, attiva un re-pesaggio
    deciso pro-deflation quando l'economia entra in recessione disinflazionistica.

    Trigger composito:
    - gdp_roc < 1.0 (crescita stallata o negativa)
    - unrate_roc > 0.2 (disoccupazione in salita)
    - vix > 22 (paura del mercato)
    - breakeven_10y < 2.3 (forward inflation in collasso — il mercato NON crede
      che l'inflazione corrente persisterà)

    Tutti e 4 i trigger devono fired → `raw_scores["stagflation"] *= 0.4`
    e `raw_scores["deflation"] *= 1.5`. Risolve il problema 2008/2000/1990 dove
    il classifier vede CPI ancora alto (lagging) e classifica stagflation
    invece di transizione disinflazionistica.

    **T10b 2026-05-17 PROMOTED TO DEFAULT ON**: validation walk-forward 15 sim
    real-prices: Sortino delta vs baseline **+0.613**, DD delta +23.5pp.
    Sblocca pattern 2008/2000/1990 recession transitions. Disable via
    `USE_GDP_COLLAPSE_OVERRIDE=0` se serve baseline puro per debug.

    Default: True (post-T10b validation).
    """
    return _read_flag("USE_GDP_COLLAPSE_OVERRIDE", default=True)


# use_rank_percentile_scoring() rimosso 2026-05-28 (council KILL T10b: -0.421 Sortino)


# use_conditional_asset_scoring() rimosso 2026-05-28 (code audit: no validation)


def use_financial_stress_veto() -> bool:
    """T9-AUDIT-FIX: Financial-stress veto layer post-classifier.

    Quando VIX > 28 AND BAA > 2.5% OR HY_OAS > 5%, AND regime predicted in
    {goldilocks, reflation}, il veto si attiva → confidence × 0.5, transition
    mode forzato. Cattura 2018 Q4 sell-off e 2023 SVB dove macro indicators
    erano sani ma financial conditions in stress.

    Default: False (back-compat).
    """
    return _read_flag("USE_FINANCIAL_STRESS_VETO")


def use_defensive_transition_mode() -> bool:
    """Tier 9 DEFENSIVE TRANSITION MODE: se True, in periodi di transizione
    confusionari (regime mixed, entropy alta, confidence bassa) il modello
    suggerisce allocazione DIFENSIVA invece di top-N basato su regime.

    Filosofia (Buffett + Taleb): "Quando non sai cosa fare, NON fare nulla.
    Cash è una posizione." Meglio missare un rally del 5% che subire un crash
    del 30% perché il modello era confuso.

    Detection trigger (TUTTI 2 di 3 devono fired):
    - max(regime_probs) < 0.45 (nessun regime domina chiaramente)
    - shannon_entropy(probs) > 1.40 (incertezza alta — max=2.0 = uniforme)
    - confidence < 0.40 (modello non sicuro)

    Allocazione difensiva (Bridgewater All-Weather conservative):
    - cash_money_market: 35%
    - us_bonds_short: 25%
    - us_bonds_long: 15%
    - gold: 15%
    - tips_inflation_bonds: 10%

    Default: False (back-compat).
    """
    return _read_flag("USE_DEFENSIVE_TRANSITION_MODE")


def use_liquidity_surge_override() -> bool:
    """Council 2026-05-27: liquidity_surge regime override.

    User insight: 'Il V-shape 2020 era predittibile dalla STAMPA Fed (QE)'.
    Council unanime: aggiungere override layer che ri-pesa probabilita
    "deflation" verso "reflation_forced" quando rileva massive liquidity injection.

    Trigger composite (TUTTI 3 must fire):
    - M2 YoY > +10% (Soros threshold, vs Buffett +15% più strict)
    - WALCL ROC 3m > +12%
    - Real rates declining 3m (real_rate_change_3m < 0)

    Effetto quando fires:
    - reflation_prob += 0.20 (boost)
    - deflation_prob *= 0.6 (dampen, NON cancellare)
    - normalize tutto

    Coexists with USE_GDP_COLLAPSE_OVERRIDE (defense trigger) → questo è il
    "all clear" early signal post-stampa.

    Default: True (council priority CRITICAL per fix 2020 disaster).
    """
    return _read_flag("USE_LIQUIDITY_SURGE_OVERRIDE", default=True)


def use_ai_portfolio_cash_overnight() -> bool:
    """AI Portfolio horserace: il capitale NON schierato rende l'overnight.

    2026-07-13 (insight Erik): tenere liquidità è già un posizionamento macro,
    ma il residuo non allocato rendeva 0% mentre i benchmark (SPY, 60/40) sono
    full-invested → doppia penalità (niente upside E niente risk-free), che
    falsava l'horserace svantaggiando le AI del ~risk-free × frazione idle.

    Con flag ON: idle = max(0, 1 - deployed) rende il daily return di BIL
    (SPDR 1-3M T-Bill, l'equivalente USD di XEON/€STR). Fix di FAIRNESS del
    confronto, non un cambio di strategia (i gate di entry restano invariati).

    Default: True. Disattivabile per riprodurre la vecchia contabilità a 0%.
    """
    return _read_flag("USE_AI_PORTFOLIO_CASH_OVERNIGHT", default=True)


def use_ai_portfolio_deploy_boost() -> bool:
    """AI Portfolio: combo "deploy-boost" (deadband + fastramp + floor).

    2026-07-13: la strategia model/data-driven deploya troppo poco capitale
    (~20% nei mercati normali) e cede l'edge del motore macro. La causa non sono
    i gate d'ingresso (abbassarli non aiuta) ma la MECCANICA del ramp. Questa
    combo schiera meglio il capitale già approvato:
    - banda-morta: add-tranche a score≥ENTRY (55) non ≥60
    - fast-ramp: max 2 tranche invece di 2-5
    - floor: deployment minimo 30% sui top del regime

    Backtest A/B (`horserace_ab_matrix.py`): deployed 21→48%, alpha 60/40
    −3.0→−0.9pp (bull) e +8.4→+10.1pp (crisi), severe 30→17%. Migliora ENTRAMBI
    i regimi (non baratta uno per l'altro).

    **2026-07-13 PROMOSSO DEFAULT-ON** (Erik authorized), numeri RICONCILIATI dopo
    audit raydalio + robustezza multi-seed (`scripts/deployboost_robustness.py`,
    5 seed × 4 finestre, 40 sim). Il Δ vs 60/40 (combo − baseline) è ROBUSTO e
    positivo su ogni seed/finestra, ANCHE cost-adjusted (10bps turnover):
      - broad 1995-2023: Δ **+1.8±0.5pp**  (cost-adj +1.6)
      - 2008:            Δ **+0.6±0.3pp**  (cost-adj +0.4)   ← modesto ma stabile
      - 2020 (V-shape):  Δ **+8.0±0.5pp**  (cost-adj +7.7)   ← il caso decisivo
      - all_crisis:      Δ **+2.0±0.3pp**  (cost-adj +1.8)
    ⚠️ COSTO REALE: il max-DD ~RADDOPPIA (broad −2.1→−3.1%, 2008 −2.9→−5.0%,
    2020 −3.6→−6.5%, all_crisis −3.9→−5.5%) — più capitale schierato = più rischio.
    ⚠️ `apply_deployment_floor` NON applica il momentum gate (può schierare su asset
    regime-favorevoli ma BEARISH). Nota: i livelli ASSOLUTI di alpha sono
    seed-sensibili (il broad resta sotto 60/40 di ~5-8pp); è il Δ a essere stabile.
    Reversibile via env=0.

    Default: **True**. Disattivabile con `USE_AI_PORTFOLIO_DEPLOY_BOOST=0`.
    """
    return _read_flag("USE_AI_PORTFOLIO_DEPLOY_BOOST", default=True)


def use_uncertainty_gate() -> bool:
    """Paper trading insight 2026-05-27 PROMOTED DEFAULT-ON: quando confidence
    < 0.30, fallback 60/40 statico (no aggressive bet con modello incerto).

    **Validation paper trading** (120 trades × 4 seeds):
    - Baseline mean alpha vs 60/40: -0.85pp
    - With gate: **+0.97pp** (delta +1.82pp boost)
    - Deflation predictions improvement: -7.57pp → -1.48pp (seed 300)

    Risolve pattern "low-confidence systematic failures" (6/30 baseline = 20%
    delle predictions con conf < 0.30 → 30% win rate vs 50% overall).

    Threshold: confidence < 0.30.

    Default: True (post paper trading validation). Disable via env=0.
    """
    return _read_flag("USE_UNCERTAINTY_GATE", default=True)


def use_momentum_pillars() -> bool:
    """T11 (2026-05-27 Council): se True, aggiunge pillar momentum-based al
    classifier che catturano l'ACCELERAZIONE di CPI/Core PCE/GDP, non solo il
    livello istantaneo.

    Pillar aggiunti:
    - `inflation_accelerating`: cpi_yoy_change_6m > 1.0% (peso 0.06 stagflation)
    - `core_pce_accelerating`: core_pce_yoy_change_6m > 0.8% (peso 0.05 stagflation)
    - `gdp_decelerating`: gdp_roc_change_6m < -0.5% (peso 0.04 stag, 0.05 defla)
    - `inflation_persistent`: min(cpi_yoy_last_6m) > 2.5% (peso 0.04 stagflation)
    - `dfm_growth_decelerating`: gdp_yoy_dfm_change_3m < -0.4% (peso 0.03 stag, 0.04 defla)

    **Paper trading validation 2026-05-27 PROMOTED DEFAULT-ON**:
    - N=100 sims monthly rebalance: Sharpe 1.27 → 1.28
    - α vs All-Weather: +3.07pp → +3.34pp (+0.27pp boost)
    - Win rate vs All-Weather: 70% → 74%
    - Consistente miglioramento (no regression)

    Hybrid level+momentum (Bridgewater all-weather 1996+, raydalio "delta primary").

    Default: True (post paper trading validation). Disable via env=0.
    """
    return _read_flag("USE_MOMENTUM_PILLARS", default=True)


def use_nowcast_pillar() -> bool:
    """T19 (2026-05-31 Council sessione 26): nowcast pillar (Atlanta GDPNow).

    Council brutal feedback sessione 26: highest expected alpha (+0.8-1.5pp).
    Real-time GDP estimate aggiornato weekly, lead 1-3 mesi su BEA print.

    Pattern:
    - GDPNow > 3% = strong growth nowcast (reflation early signal)
    - GDPNow < 1% = weak growth nowcast (deflation early)
    - GDPNow deviation > 1.5pp da consensus = "surprise" trigger

    Pillar aggiunti:
    - `gdpnow_strong` (reflation 0.04): GDPNow > 3%
    - `gdpnow_weak` (deflation 0.04): GDPNow < 1%
    - `gdpnow_moderate` (goldilocks 0.03): GDPNow 1.5-3%

    Default: **False** (post-validation sessione 26: drag -0.22pp vs OFF).
    Council expected +0.8-1.5pp ma realtà n=100 sims 2014-2023: -0.22pp.
    Rebalance pesi -0.02 sui gdp pillars correlati ha penalizzato senza
    compensazione adequata.

    Lezione: features predittive NON automatic alpha boosters. Solo T11
    momentum + T17 labor/credit + T18 cyclical hanno dimostrato edge.
    Atlanta GDPNow disponibile dal 2014 (FRED GDPNOW).
    """
    return _read_flag("USE_NOWCAST_PILLAR", default=False)


def use_level_accel_pillars() -> bool:
    """T20 (2026-07-12 council P2): pillar livello-vs-ACCELERAZIONE.

    Fix strutturale della diagnosi council 5/5: deflation overfit (40.3% vs
    5-15% storico reale) e goldilocks blindness (accuracy 29.9%, 1995-99 e
    2017-19 misclassificati come deflation) nascono dal confondere LIVELLI
    con DINAMICHE: `inflation_low` (CPI < 2%) premiava deflation anche nei
    boom disinflazionistici goldilocks.

    Pacchetto:
    - deflation: labor_collapsing (payrolls YoY < 0) + disinflation_impulse
      (CPI in CADUTA × claims in salita, composito) — deflazione =
      deterioramento, non livello basso. REBALANCE: inflation_low 0.12 → 0.06.
    - goldilocks: labor_momentum_healthy (payrolls ~p50) +
      unit_labor_costs_contained (ULC YoY < p50=2.3) + productivity_healthy.
    - stagflation: unit_labor_costs_spiraling (ULC YoY > p90=7, wage-spiral 70s).

    Soglie da QUANTILI STORICI REALI FRED 1948-2026 (regola anti-hand-tuning):
    ULC p50=2.29 p90=7.00 · productivity p25=0.97 · payrolls p50=1.94 ·
    cpi_chg6 p25=-0.62. Sanity su episodi veri: 1997 gold (ULC 1.68, pay 2.54)
    vs 2008 defl (pay -2.56, cpi_chg6 -4.96) vs 1974 stag (ULC 11.2).

    Serie nuove: ULCNFB + OPHNFB (trimestrali, dal 1947 — piena copertura
    backfill). Default OFF fino a validation gate A/B sui 32 episodi.
    """
    return _read_flag("USE_LEVEL_ACCEL_PILLARS", default=False)


def use_cyclical_pillar() -> bool:
    """T18 (2026-05-31 Council sessione 25): cyclical leading indicators.

    Tesi: T17 ha dimostrato che features predittive del classifier producono
    alpha edge REAL (+0.15pp consistent). Stessa logica: aggiungere indicatori
    cyclical-leading che precedono macro shifts.

    Indicatori aggiunti (early cyclical signals):
    - heavy_truck_sales (HTRUCKSSAAR): freight demand collapse leading
      recession 6-12m. Storia: heavy truck -20% YoY = recession 80% accuracy.
    - building_permits (PERMIT): housing cycle leading indicator (3-6m lead).
    - consumer_credit (TOTALSL): household balance sheet expansion/contraction.
    - durable_goods_orders (DGORDER): business investment leading capex.

    Pillar aggiunti:
    - `truck_freight_strong` (reflation 0.04): heavy_truck_sales YoY > +5%
    - `truck_freight_collapse` (deflation 0.04): heavy_truck YoY < -10%
    - `permits_expansion` (reflation/goldilocks 0.03): permits YoY > +5%
    - `permits_collapse` (deflation 0.03): permits YoY < -20%
    - `durable_orders_expansion` (reflation 0.03): YoY > +5%
    - `durable_orders_contraction` (deflation 0.03): YoY < -5%
    - `consumer_credit_expansion` (reflation/goldilocks 0.02): YoY > +5%

    Default: True. Rebalance leggero (-0.02) sui pillars correlati.
    """
    return _read_flag("USE_CYCLICAL_PILLAR", default=True)


def use_labor_credit_pillar() -> bool:
    """T17 (2026-05-31 Council sessione 23): se True, aggiunge pillar al classifier
    basati su LABOR MARKET TIGHTNESS + CREDIT CONDITIONS per migliorare accuracy.

    Razionale: i 5 council edge (Kelly, Tail Hedge, Events, Forward Inflation,
    Sector ETF) sono tutti DD reducer marginali, nessun vero alpha booster.
    Attaccare alpha alla radice = più feature predittive nel classifier.

    Pillar aggiunti (T17):
    - `wage_growth_high` (stagflation 0.04): Atlanta wage tracker > 4.5%
    - `quits_rate_elevated` (reflation/goldilocks 0.03): JOLTS quits > 2.5%
    - `hours_worked_declining` (deflation 0.03): AWHAETP YoY < 0 (early labor)
    - `hy_credit_stress` (deflation 0.04): HY OAS > 5% (credit conditions)
    - `hy_credit_tight` (reflation/goldilocks 0.03): HY OAS < 3% (risk-on credit)

    Rebalance: piccoli (-0.02) sui pillars esistenti correlati.

    Default: True (council T17 raccomandation).
    """
    return _read_flag("USE_LABOR_CREDIT_PILLAR", default=True)


def use_forward_inflation_pillar() -> bool:
    """T13 (2026-05-31 Council post bug T12): se True, aggiunge pillar al classifier
    basati su FORWARD-LOOKING inflation features per fixare la "inflation blindness"
    diagnosticata dopo il bug deflation-vs-stagflation 2026-05-31.

    Razionale: il classifier vedeva solo CPI/Core PCE level + momentum.
    Mancava feature SET forward-looking che market/consumer pricing già scontano:
    - T5YIFR (5y5y forward): Fed's preferred inflation anchor (>2.5% = break)
    - MICH (UMich 1y consumer): sentiment-driven inflation expectation
    - Sticky-Price CPI (Atlanta Fed): hardest inflation component to break

    Pillar aggiunti:
    - `umich_inflation_high` (stagflation 0.04): MICH > 3.5%
    - `sticky_inflation_high` (stagflation 0.04): sticky_cpi_yoy > 4%
    - `breakeven_5y5y_high` (stagflation 0.03): T5YIFR > 2.5%
    - `umich_inflation_low` (deflation 0.03): MICH < 2.0%
    - `umich_inflation_anchored` (goldilocks 0.03): MICH 2.0-2.5%
    - `sticky_inflation_anchored` (goldilocks 0.03): sticky_cpi 2.0-2.7%

    Pillar rebalance (sottrai pesi dai level pillar per coerenza Σ=1):
    - Stagflation: inflation_high -0.04, core_pce_high -0.04, breakeven_high -0.03
    - Deflation: breakeven_collapse -0.03
    - Goldilocks: breakeven_stable -0.03, core_pce_contained -0.03

    Default: True (council T13 raccomandation post fix T12).
    """
    return _read_flag("USE_FORWARD_INFLATION_PILLAR", default=True)


def use_ml_regime_blend() -> bool:
    """Tier 8 ML REGIME BLEND: se True, le probabilities regime sono blendate
    con un ensemble ML (RandomForest + GradientBoosting trained su 18 episodi
    storici labeled).

    Formula:
        final_probs = (1 - alpha) * rule_based_probs + alpha * ml_probs

    Default alpha=0.4 (40% peso ML, 60% rule-based). Risolve i bias del
    rule-based (es. 2000 dotcom classificato stagflation invece di deflation)
    mantenendo la stabilità del rule-based su scenari noti.

    LOO CV stats: accuracy 77.8% (14/18), Brier 0.097.
    Ensemble blend stats: accuracy ~89% sui 7 audit episodes.

    **T10b 2026-05-17 PROMOTED TO DEFAULT ON**: validation walk-forward 15 sim
    real-prices: Sortino delta vs baseline **+0.271**, DD delta +20.7pp.
    Riduce volatilita downside a costo di mean return (alpha -48% ma più
    consistente). Use case: barbell defensive. Disable via
    `USE_ML_REGIME_BLEND=0` se serve baseline puro per debug.

    Default: True (post-T10b validation).
    """
    return _read_flag("USE_ML_REGIME_BLEND", default=True)


# use_downside_protection_bonus() rimosso 2026-05-28 (code audit: no validation)


# `use_asset_feedback` rimosso 2026-05-28 (code audit): modulo dead, 21 test
# verdi su funzione mai chiamata in produzione. File asset_feedback.py deleted.
