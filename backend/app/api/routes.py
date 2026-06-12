"""API endpoints per il Macro Analyzer."""

import json
from datetime import date, datetime

from fastapi import APIRouter, Body, Depends, HTTPException, Query
from pydantic import BaseModel
from sqlalchemy.orm import Session

from app.database import get_db
from app.models import DailySignal, RegimeClassification
from app.models.news_signals import NewsSignal
from app.models.secular_trends import SecularTrend
from app.services.regime.classifier import classify_regime
from app.services.scoring.engine import calculate_final_scores, ASSET_CLASSES

router = APIRouter(prefix="/api/v1")


# --- Schemas ---


class RegimeResponse(BaseModel):
    date: date
    regime: str
    probabilities: dict[str, float]
    confidence: float
    fit_scores: dict[str, float] = {}


class SignalResponse(BaseModel):
    date: date
    asset_class: str
    final_score: float
    regime_component: float
    secular_trend_bonus: float
    news_signal: float
    momentum_penalty: float


class ScoreboardResponse(BaseModel):
    date: date
    regime: str
    confidence: float
    scores: dict[str, float]


class DedollarHistoryItem(BaseModel):
    date: date
    combined_score: float
    cyclical_score: float
    structural_score: float
    decade_score: float
    twenty_year_score: float | None = None
    geopolitical_score: float = 0.0
    acceleration: float = 0.0


class SignalsHistoryItem(BaseModel):
    date: date
    scores: dict[str, float]


class MacroIndicatorsHistoryItem(BaseModel):
    date: date
    indicators: dict[str, float]


class PlayerHistoryItem(BaseModel):
    date: date
    players: dict[str, float]


class RegimeConditionMeta(BaseModel):
    name: str
    weight: float
    description: str


class AssetRegimeStats(BaseModel):
    hit_rate: float
    avg_return: float
    vol: float
    sharpe: float


class DataSnapshotResponse(BaseModel):
    date: date | None
    regime: str | None
    regime_conditions: dict[str, list[RegimeConditionMeta]]
    asset_regime_data: dict[str, dict[str, AssetRegimeStats]]
    asset_dedollar_sensitivity: dict[str, float]
    current_indicators: dict[str, float]
    current_dedollar_indicators: dict[str, float]
    current_probabilities: dict[str, float]
    current_fit_scores: dict[str, float]
    current_asset_scores: dict[str, float]
    current_dedollar_combined: float | None


class TransitionMatrixResponse(BaseModel):
    horizon_days: int
    regimes: list[str]
    counts: dict[str, dict[str, int]]
    probabilities: dict[str, dict[str, float]]
    avg_duration_days: dict[str, float]
    self_transition_probability: dict[str, float]
    total_observations: int
    date_from: date | None
    date_to: date | None
    projected_probabilities: dict[str, float] | None = None


class HMMResponse(BaseModel):
    regimes: list[str]
    probabilities: dict[str, float]
    current_state: int
    state_to_regime: dict[int, str]
    n_training: int
    log_likelihood: float
    feature_means: dict[str, float]
    feature_stds: dict[str, float]


class SmoothedPointResponse(BaseModel):
    date: date
    raw: dict[str, float]
    smoothed: dict[str, float]


class SmoothedHistoryResponse(BaseModel):
    points: list[SmoothedPointResponse]
    transition_horizon_days: int
    total_observations: int


class AssetRegimeMeasured(BaseModel):
    asset: str
    regime: str
    n_observations: int
    hit_rate: float | None
    real_return: float | None
    volatility: float | None
    sharpe: float | None


class AssetValidationResponse(BaseModel):
    """Confronto tra ASSET_REGIME_DATA hardcoded e metriche misurate dai prezzi reali."""
    horizon_months: int
    regime_threshold: float
    measured: list[AssetRegimeMeasured]
    hardcoded: list[AssetRegimeMeasured]
    notes: list[str]


class CalibrationDiagnostic(BaseModel):
    asset: str
    regime: str
    n_observations: int
    weight_measured: float
    prior: dict[str, float]
    measured: dict[str, float | None] | None
    calibrated: dict[str, float]


class FOMCAnalysisResponse(BaseModel):
    url: str
    doc_type: str
    published_date: str
    title: str
    hawkish_dovish_score: float
    confidence: float
    key_topics: list[str]
    forward_guidance: str
    regime_implication: dict[str, float]
    summary: str
    provider: str
    analyzed_at: str


class FOMCReportResponse(BaseModel):
    analyses: list[FOMCAnalysisResponse]
    latest_score: float | None
    avg_score_3last: float | None
    trend: str
    n_documents: int


class TermPremiumPointResponse(BaseModel):
    date: date
    fitted_yield: float
    term_premium: float
    expected_path: float


class TermPremiumStatResponse(BaseModel):
    regime: str
    n_observations: int
    mean_fitted: float | None
    mean_term_premium: float | None
    mean_expected_path: float | None
    pct_term_premium_positive: float | None


class TermPremiumReportResponse(BaseModel):
    common_period: tuple[str, str]
    threshold: float
    n_observations: int
    points: list[TermPremiumPointResponse]
    by_regime: list[TermPremiumStatResponse]


class FactorRegimeStatResponse(BaseModel):
    factor: str
    regime: str
    n_observations: int
    mean_annual: float | None
    vol_annual: float | None
    sharpe: float | None
    win_rate: float | None


class FactorRegimeReportResponse(BaseModel):
    threshold: float
    n_months_analyzed: int
    factor_keys: list[str]
    regimes: list[str]
    common_period: tuple[str, str]
    stats: list[FactorRegimeStatResponse]


class SmoothedSeriesPoint(BaseModel):
    date: date
    raw: float
    filtered: float
    smoothed: float


class SmoothedIndicatorResponse(BaseModel):
    series_name: str
    description: str
    lambda_used: float
    n_points: int
    variance_reduction: float
    points: list[SmoothedSeriesPoint]


class ScoreComparisonItem(BaseModel):
    asset: str
    pure_score: float           # senza dedollar bonus (puro data-driven)
    adjusted_score: float       # con dedollar bonus
    dedollar_delta: float       # adjusted - pure (puo' essere negativo)
    sensitivity: float          # ASSET_DEDOLLAR_SENSITIVITY[asset]


class ScoreboardDedollarComparison(BaseModel):
    date: date
    regime: str
    probabilities: dict[str, float]
    dedollar_combined_score: float
    use_dedollar_bonus_active: bool      # stato corrente env var
    items: list[ScoreComparisonItem]     # ordinati per |dedollar_delta| desc


class RegimeBandResponse(BaseModel):
    regime: str
    median: list[float]
    p10: list[float]
    p25: list[float]
    p75: list[float]
    p90: list[float]
    mean: list[float]


class AssetBandResponse(BaseModel):
    asset: str
    median: list[float]
    p10: list[float]
    p25: list[float]
    p75: list[float]
    p90: list[float]
    mean: list[float]


class MonteCarloResponse(BaseModel):
    n_paths: int
    n_steps: int
    horizon_days: int
    initial_distribution: dict[str, float]
    initial_source: str                          # rule_based | ensemble | explicit
    step_dates_offsets: list[int]
    transition_matrix_observations: int
    regime_bands: list[RegimeBandResponse]
    asset_bands: list[AssetBandResponse]
    notes: list[str]
    ensemble_disagreement: float | None = None
    ensemble_confidence: float | None = None


class ScenarioResponse(BaseModel):
    scenario_key: str
    label: str
    description: str
    baseline_indicators: dict[str, float]
    shocked_indicators: dict[str, float]
    baseline_regime: str
    baseline_probabilities: dict[str, float]
    shocked_regime: str
    shocked_probabilities: dict[str, float]
    baseline_scores: dict[str, float]
    shocked_scores: dict[str, float]
    asset_score_deltas: dict[str, float]


class BacktestStrategyResponse(BaseModel):
    name: str
    description: str
    nav: list[dict]               # [{date, value}]
    monthly_returns: list[dict]   # [{date, value}]
    stats: dict                   # serializzato da PerformanceStats
    alpha_vs_60_40: dict[str, float]


class BacktestResponse(BaseModel):
    common_start: str
    common_end: str
    n_months: int
    strategies: list[BacktestStrategyResponse]


class RecessionLeadResponse(BaseModel):
    recession_start: str
    recession_end: str
    duration_months: int
    signal_date: str | None
    lead_months: float | None
    max_prob_during: float
    pre_recession_max_prob: float


class LeadTimeResponse(BaseModel):
    threshold: float
    lookback_months: int
    n_recessions_analyzed: int
    hit_rate: float
    avg_lead_months: float | None
    median_lead_months: float | None
    recessions: list[RecessionLeadResponse]


class ModelViewResponse(BaseModel):
    name: str
    probabilities: dict[str, float]
    error: str | None = None
    metadata: dict | None = None


class EnsembleResponse(BaseModel):
    weights: dict[str, float]
    views: list[ModelViewResponse]
    ensemble_probabilities: dict[str, float]
    confidence: float
    disagreement_score: float
    high_disagreement: bool
    dominant_regime: str
    notes: list[str]


class CalibrationResponse(BaseModel):
    version: int
    calibrated_on: str
    n_classifications: int
    params: dict[str, float | int]
    diagnostics: list[CalibrationDiagnostic]
    use_calibrated_scoring: bool


class NewsItemResponse(BaseModel):
    date: date
    source: str
    title: str
    summary: str
    sentiment: float
    relevance: float
    affected_assets: dict[str, float]


class PlayerSignal(BaseModel):
    key: str
    label: str
    value: float | None
    score: float | None
    interpret: str
    red_flag: bool
    weight: float


class PlayerScore(BaseModel):
    label: str
    score: float
    coverage: float
    signals: list[PlayerSignal]


class DedollarizationResponse(BaseModel):
    date: date
    score: float
    structural_score: float
    decade_score: float
    twenty_year_score: float | None = None
    acceleration: float
    combined_score: float
    geopolitical_score: float = 0.0
    components: dict[str, float]
    structural: dict[str, float]
    decade: dict[str, float]
    twenty_year: dict[str, float] = {}
    by_player: dict[str, PlayerScore] = {}
    player_history: dict[str, dict[str, float]] = {}
    player_acceleration: dict[str, float] = {}
    explanation: str | None = None


class ConditionDetail(BaseModel):
    name: str
    description: str
    raw_score: float
    weight: float
    signal: str  # "strong", "moderate", "weak", "neutral"


class TrajectoryForce(BaseModel):
    name: str
    type: str
    description: str
    pushes_toward: str
    strength: float


class TrajectoryDrift(BaseModel):
    regime: str
    current: float
    projected: float
    delta: float


class TrajectoryResponse(BaseModel):
    current_regime: str
    projected_regime: str
    projected_probabilities: dict[str, float]
    projected_fit_scores: dict[str, float] = {}
    forces: list[TrajectoryForce]
    drift: list[TrajectoryDrift]
    transition_risk: float
    summary: str
    projected_scores: dict[str, float] = {}


class RegimeExplainResponse(BaseModel):
    date: date
    regime: str
    probabilities: dict[str, float]
    fit_scores: dict[str, float] = {}
    confidence: float
    indicators: dict[str, float]
    top_drivers: list[ConditionDetail]
    dedollar_indicators: dict[str, float]
    trajectory: TrajectoryResponse | None = None


class HealthResponse(BaseModel):
    status: str
    version: str


# --- Endpoints ---


@router.get("/health", response_model=HealthResponse)
def health_check():
    """Health check endpoint."""
    return {"status": "ok", "version": "1.0.0"}


def _extract_fit_scores(record: RegimeClassification) -> dict[str, float]:
    """Estrae fit_scores dal JSON conditions_met, se presenti."""
    if not record.conditions_met:
        return {}
    try:
        payload = json.loads(record.conditions_met)
        fit = payload.get("fit_scores") or {}
        return {k: float(v) for k, v in fit.items()} if isinstance(fit, dict) else {}
    except Exception:
        return {}


@router.get("/regime/current", response_model=RegimeResponse)
def get_current_regime(db: Session = Depends(get_db)):
    """Regime macro corrente (ultima classificazione)."""
    latest = (
        db.query(RegimeClassification)
        .order_by(RegimeClassification.date.desc())
        .first()
    )
    if not latest:
        raise HTTPException(status_code=404, detail="Nessuna classificazione disponibile")

    return RegimeResponse(
        date=latest.date,
        regime=latest.regime,
        probabilities={
            "reflation": latest.probability_reflation,
            "stagflation": latest.probability_stagflation,
            "deflation": latest.probability_deflation,
            "goldilocks": latest.probability_goldilocks,
        },
        confidence=latest.confidence,
        fit_scores=_extract_fit_scores(latest),
    )


@router.get("/regime/explain", response_model=RegimeExplainResponse)
def get_regime_explain(db: Session = Depends(get_db)):
    """Spiegazione dettagliata del regime corrente: indicatori + driver principali."""
    from app.services.regime.classifier import REGIME_CONDITIONS

    latest = (
        db.query(RegimeClassification)
        .order_by(RegimeClassification.date.desc())
        .first()
    )
    if not latest:
        raise HTTPException(status_code=404, detail="Nessuna classificazione disponibile")

    payload = json.loads(latest.conditions_met) if latest.conditions_met else {}

    # Supporta sia il vecchio formato (solo conditions) che il nuovo (conditions + indicators)
    if "conditions" in payload:
        conditions = payload["conditions"]
        indicators = payload.get("indicators", {})
        dedollar_indicators = payload.get("dedollar_indicators", {})
        fit_scores = payload.get("fit_scores", {}) or {}
    else:
        conditions = payload
        indicators = {}
        dedollar_indicators = {}
        fit_scores = {}

    # Estrai i top driver per il regime dominante
    regime_conditions = conditions.get(latest.regime, {})
    regime_meta = REGIME_CONDITIONS.get(latest.regime, {})

    drivers: list[ConditionDetail] = []
    for cond_name, cond_data in regime_conditions.items():
        raw = cond_data.get("raw_score", 0)
        if raw > 0.6:
            signal = "strong"
        elif raw > 0.4:
            signal = "moderate"
        elif raw > 0.2:
            signal = "weak"
        else:
            signal = "neutral"

        desc = regime_meta.get(cond_name, {}).get("description", cond_name)
        drivers.append(ConditionDetail(
            name=cond_name,
            description=desc,
            raw_score=round(raw, 3),
            weight=cond_data.get("weight", 0),
            signal=signal,
        ))

    # Ordina per weighted_score decrescente
    drivers.sort(key=lambda d: d.raw_score * d.weight, reverse=True)

    # Trajectory (salvata nello stesso payload)
    trajectory_data = payload.get("trajectory")
    trajectory = None
    if trajectory_data and "projected_regime" in trajectory_data:
        trajectory = TrajectoryResponse(
            current_regime=trajectory_data["current_regime"],
            projected_regime=trajectory_data["projected_regime"],
            projected_probabilities=trajectory_data["projected_probabilities"],
            projected_fit_scores=trajectory_data.get("projected_fit_scores", {}) or {},
            forces=[TrajectoryForce(**f) for f in trajectory_data.get("forces", [])],
            drift=[TrajectoryDrift(**d) for d in trajectory_data.get("drift", [])],
            transition_risk=trajectory_data["transition_risk"],
            summary=trajectory_data["summary"],
            projected_scores=trajectory_data.get("projected_scores", {}) or {},
        )

    return RegimeExplainResponse(
        date=latest.date,
        regime=latest.regime,
        probabilities={
            "reflation": latest.probability_reflation,
            "stagflation": latest.probability_stagflation,
            "deflation": latest.probability_deflation,
            "goldilocks": latest.probability_goldilocks,
        },
        fit_scores=fit_scores,
        confidence=latest.confidence,
        indicators=indicators,
        top_drivers=drivers,
        dedollar_indicators=dedollar_indicators,
        trajectory=trajectory,
    )


@router.get("/regime/transition-matrix", response_model=TransitionMatrixResponse)
def get_transition_matrix(
    horizon_days: int = Query(default=30, ge=1, le=365),
    project_steps: int = Query(default=0, ge=0, le=24,
        description="Se >0, proietta le probabilita correnti moltiplicando la matrice N volte"),
    db: Session = Depends(get_db),
):
    """Matrice di transizione empirica tra regimi macro.

    Conta le transizioni osservate in `RegimeClassification` a orizzonte `horizon_days`.
    Se `project_steps` > 0, restituisce anche la proiezione delle probabilita correnti
    dopo N passi (ogni passo = `horizon_days` giorni).
    """
    from app.services.regime.transition_matrix import (
        compute_transition_matrix,
        project_probabilities,
    )

    result = compute_transition_matrix(db, horizon_days=horizon_days)

    projected: dict[str, float] | None = None
    if project_steps > 0 and result.total_observations > 0:
        latest = (
            db.query(RegimeClassification)
            .order_by(RegimeClassification.date.desc())
            .first()
        )
        if latest:
            current = {
                "reflation": latest.probability_reflation,
                "stagflation": latest.probability_stagflation,
                "deflation": latest.probability_deflation,
                "goldilocks": latest.probability_goldilocks,
            }
            projected = project_probabilities(
                result.probabilities, current, steps=project_steps
            )

    return TransitionMatrixResponse(
        horizon_days=result.horizon_days,
        regimes=result.regimes,
        counts=result.counts,
        probabilities=result.probabilities,
        avg_duration_days=result.avg_duration_days,
        self_transition_probability=result.self_transition_probability,
        total_observations=result.total_observations,
        date_from=result.date_range[0],
        date_to=result.date_range[1],
        projected_probabilities=projected,
    )


@router.get("/regime/hmm", response_model=HMMResponse)
def get_regime_hmm(
    n_states: int = Query(default=4, ge=2, le=8),
    db: Session = Depends(get_db),
):
    """Classificatore HMM gaussiano sulle feature macro storiche.

    Addestra on-demand un GaussianHMM e mappa gli stati latenti ai 4 regimi
    rule-based via majority vote. Restituisce la distribuzione posteriore
    sullo stato corrente. Richiede >= 60 osservazioni storiche: se il DB e'
    vuoto, invoca prima /regime/backfill/historical.
    """
    from app.services.regime.hmm_classifier import fit_and_predict_hmm

    try:
        result = fit_and_predict_hmm(db, n_states=n_states)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    return HMMResponse(
        regimes=result.regimes,
        probabilities=result.probabilities,
        current_state=result.current_state,
        state_to_regime=result.state_to_regime,
        n_training=result.n_training,
        log_likelihood=result.log_likelihood,
        feature_means=result.feature_means,
        feature_stds=result.feature_stds,
    )


@router.get("/regime/forecast/monte-carlo", response_model=MonteCarloResponse)
def get_regime_monte_carlo(
    n_paths: int = Query(default=500, ge=100, le=5000),
    n_steps: int = Query(default=12, ge=1, le=36),
    horizon_days: int = Query(default=30, ge=7, le=90),
    initial_source: str = Query(
        default="rule_based",
        regex="^(rule_based|ensemble)$",
        description="Sorgente della distribuzione iniziale: 'rule_based' "
        "(default, ultima classification del DB) o 'ensemble' "
        "(weighted average dei 3 modelli — riflette il vero stato di "
        "incertezza quando i modelli sono in disaccordo).",
    ),
    include_dedollar: bool | None = Query(default=None),
    db: Session = Depends(get_db),
):
    """Monte Carlo regime trajectories: simula N path dalla transition matrix
    empirica e ritorna percentili (p10/p25/p50/p75/p90) per regime + asset score.
    """
    from app.services.regime.monte_carlo import run_monte_carlo

    try:
        r = run_monte_carlo(
            db, n_paths=n_paths, n_steps=n_steps, horizon_days=horizon_days,
            initial_source=initial_source,
            force_include_dedollar=include_dedollar,
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    return MonteCarloResponse(
        n_paths=r.n_paths,
        n_steps=r.n_steps,
        horizon_days=r.horizon_days,
        initial_distribution=r.initial_distribution,
        initial_source=r.initial_source,
        step_dates_offsets=r.step_dates_offsets,
        transition_matrix_observations=r.transition_matrix_observations,
        regime_bands=[
            RegimeBandResponse(
                regime=b.regime,
                median=b.median, p10=b.p10, p25=b.p25, p75=b.p75, p90=b.p90, mean=b.mean,
            )
            for b in r.regime_bands
        ],
        asset_bands=[
            AssetBandResponse(
                asset=b.asset,
                median=b.median, p10=b.p10, p25=b.p25, p75=b.p75, p90=b.p90, mean=b.mean,
            )
            for b in r.asset_bands
        ],
        notes=r.notes,
        ensemble_disagreement=r.ensemble_disagreement,
        ensemble_confidence=r.ensemble_confidence,
    )


@router.get("/scenarios/list")
def list_scenarios():
    """Lista scenari preset disponibili."""
    from app.services.regime.shock_scenarios import list_preset_scenarios
    return list_preset_scenarios()


@router.get("/scenarios/run", response_model=ScenarioResponse)
def run_scenario_endpoint(
    scenario_key: str = Query(..., description="key dal /scenarios/list"),
    include_dedollar: bool | None = Query(default=None),
    db: Session = Depends(get_db),
):
    """Esegue uno scenario preset: confronto baseline vs shocked (regime + asset scores)."""
    from app.services.regime.shock_scenarios import run_scenario

    try:
        r = run_scenario(db, scenario_key, force_include_dedollar=include_dedollar)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    return ScenarioResponse(
        scenario_key=r.scenario_key,
        label=r.label,
        description=r.description,
        baseline_indicators=r.baseline_indicators,
        shocked_indicators=r.shocked_indicators,
        baseline_regime=r.baseline_regime,
        baseline_probabilities=r.baseline_probabilities,
        shocked_regime=r.shocked_regime,
        shocked_probabilities=r.shocked_probabilities,
        baseline_scores=r.baseline_scores,
        shocked_scores=r.shocked_scores,
        asset_score_deltas=r.asset_score_deltas,
    )


@router.get("/backtest/run", response_model=BacktestResponse)
def run_backtest_endpoint(
    start_year: int = Query(default=2003, ge=1990, le=2025),
    end_year: int = Query(default=2026, ge=2000, le=2030),
    top_n: int = Query(default=5, ge=1, le=15),
    score_threshold: float = Query(default=30.0, ge=0.0, le=100.0),
    cost_bps: float = Query(default=10.0, ge=0.0, le=100.0),
    include_dedollar: bool | None = Query(default=None),
    db: Session = Depends(get_db),
):
    """Backtest portfolio score-weighted vs benchmark (60/40, SPY, equal-weight)."""
    from datetime import date as _date
    from dataclasses import asdict
    from app.services.backtest.runner import run_full_backtest

    try:
        result = run_full_backtest(
            db,
            start=_date(start_year, 1, 1),
            end=_date(end_year, 12, 31),
            top_n=top_n,
            score_threshold=score_threshold,
            cost_bps=cost_bps,
            force_include_dedollar=include_dedollar,
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    strategies_out = []
    for s in result.strategies:
        nav_list = [{"date": str(idx.date()), "value": float(v)} for idx, v in s.nav.items()]
        ret_list = [{"date": str(idx.date()), "value": float(v)} for idx, v in s.monthly_returns.items()]
        strategies_out.append(BacktestStrategyResponse(
            name=s.name,
            description=s.description,
            nav=nav_list,
            monthly_returns=ret_list,
            stats=asdict(s.stats),
            alpha_vs_60_40=s.alpha_vs_60_40,
        ))

    return BacktestResponse(
        common_start=result.common_start,
        common_end=result.common_end,
        n_months=result.n_months,
        strategies=strategies_out,
    )


@router.get("/backtest/lead-time", response_model=LeadTimeResponse)
def lead_time_endpoint(
    threshold: float = Query(default=0.35, ge=0.10, le=0.80),
    lookback_months: int = Query(default=60, ge=3, le=240),
    min_recession_year: int = Query(default=1970, ge=1854, le=2020),
    db: Session = Depends(get_db),
):
    """Lead time NBER recessions: di quanti mesi il sistema anticipa ogni recessione storica.

    Default lookback 60 mesi (5 anni) per catturare segnali long-cycle.
    Cap 240 mesi (20 anni) per Kuznets/Kondratieff analysis.
    """
    from app.services.backtest.lead_time import compute_lead_time_report

    r = compute_lead_time_report(
        db,
        threshold=threshold,
        lookback_months=lookback_months,
        min_recession_year=min_recession_year,
    )
    return LeadTimeResponse(
        threshold=r.threshold,
        lookback_months=r.lookback_months,
        n_recessions_analyzed=r.n_recessions_analyzed,
        hit_rate=r.hit_rate,
        avg_lead_months=r.avg_lead_months,
        median_lead_months=r.median_lead_months,
        recessions=[
            RecessionLeadResponse(
                recession_start=rec.recession_start,
                recession_end=rec.recession_end,
                duration_months=rec.duration_months,
                signal_date=rec.signal_date,
                lead_months=rec.lead_months,
                max_prob_during=rec.max_prob_during,
                pre_recession_max_prob=rec.pre_recession_max_prob,
            )
            for rec in r.recessions
        ],
    )


@router.get("/regime/ensemble", response_model=EnsembleResponse)
def get_regime_ensemble(db: Session = Depends(get_db)):
    """Ensemble di 3 modelli regime: rule-based + HMM-Market + MS-VAR.

    Ritorna posterior pesata + flag `high_disagreement` quando i modelli divergono
    (Jensen-Shannon divergence media pairwise). Useful per evidenziare incertezza:
    quando high_disagreement=True non c'e' consensus, l'allocation dovrebbe essere
    piu' conservativa.
    """
    from app.services.regime.ensemble import compute_ensemble

    r = compute_ensemble(db)
    return EnsembleResponse(
        weights=r.weights,
        views=[
            ModelViewResponse(
                name=v.name, probabilities=v.probabilities,
                error=v.error, metadata=v.metadata,
            )
            for v in r.views
        ],
        ensemble_probabilities=r.ensemble_probabilities,
        confidence=r.confidence,
        disagreement_score=r.disagreement_score,
        high_disagreement=r.high_disagreement,
        dominant_regime=r.dominant_regime,
        notes=r.notes,
    )


@router.get("/regime/smoothed-history", response_model=SmoothedHistoryResponse)
def get_regime_smoothed_history(
    days: int = Query(default=365 * 5, ge=30, le=365 * 60),
    transition_horizon_days: int = Query(default=30, ge=7, le=180),
    db: Session = Depends(get_db),
):
    """Storico probabilita regime con smoothing temporale forward-backward.

    Usa la transition matrix empirica come prior di persistenza e le posteriori
    rule-based come emission. Restituisce per ogni data sia il posterior raw sia
    quello smoothed: confronto visibile tra segnale istantaneo e filtrato.
    """
    from app.services.regime.smoothing import smooth_history

    result = smooth_history(db, days=days, transition_horizon_days=transition_horizon_days)
    return SmoothedHistoryResponse(
        points=[
            SmoothedPointResponse(date=p.date, raw=p.raw, smoothed=p.smoothed)
            for p in result.points
        ],
        transition_horizon_days=result.transition_horizon_days,
        total_observations=result.total_observations,
    )


@router.get("/asset-validation", response_model=AssetValidationResponse)
def get_asset_validation(
    horizon_months: int = Query(default=12, ge=3, le=36),
    regime_threshold: float = Query(default=0.40, ge=0.25, le=0.70),
    db: Session = Depends(get_db),
):
    """Phase 2 (MVP): confronto ASSET_REGIME_DATA hardcoded vs metriche misurate
    da rendimenti reali (Yahoo Finance + CPI deflazionato)."""
    from app.services.prices.asset_universe import ASSET_TICKERS
    from app.services.prices.returns import metrics_by_regime, regime_probs_dataframe
    from app.services.scoring.engine import ASSET_REGIME_DATA

    notes: list[str] = []
    rows = (
        db.query(RegimeClassification)
        .order_by(RegimeClassification.date.asc())
        .all()
    )
    probs_df = regime_probs_dataframe(rows)
    if probs_df.empty:
        raise HTTPException(status_code=400, detail="Nessuna classification in DB. Esegui /regime/backfill/historical.")

    measured: list[AssetRegimeMeasured] = []
    hardcoded: list[AssetRegimeMeasured] = []
    for asset in ASSET_TICKERS.keys():
        try:
            metrics = metrics_by_regime(
                asset, probs_df, horizon_months=horizon_months, threshold=regime_threshold,
            )
            for m in metrics:
                # NaN-safe per Pydantic
                def _nz(x):
                    import math
                    return None if (x is None or (isinstance(x, float) and math.isnan(x))) else float(x)
                measured.append(AssetRegimeMeasured(
                    asset=m.asset, regime=m.regime, n_observations=m.n_observations,
                    hit_rate=_nz(m.hit_rate), real_return=_nz(m.real_return),
                    volatility=_nz(m.volatility), sharpe=_nz(m.sharpe),
                ))
        except Exception as e:
            notes.append(f"{asset}: skip ({e})")

        # Hardcoded reference dallo scoring engine
        if asset in ASSET_REGIME_DATA:
            for regime, stats in ASSET_REGIME_DATA[asset].items():
                hardcoded.append(AssetRegimeMeasured(
                    asset=asset, regime=regime, n_observations=0,
                    hit_rate=stats.get("hit_rate"), real_return=stats.get("avg_return"),
                    volatility=stats.get("vol"), sharpe=stats.get("sharpe"),
                ))

    if not notes:
        notes.append("OK: validazione completata. NB: cash/bonds long pre-2007 senza backfill TR (yield≠prezzo).")

    return AssetValidationResponse(
        horizon_months=horizon_months,
        regime_threshold=regime_threshold,
        measured=measured,
        hardcoded=hardcoded,
        notes=notes,
    )


@router.get("/asset-calibration", response_model=CalibrationResponse)
def get_asset_calibration():
    """Restituisce la calibrazione persistita (seed/calibrated_asset_regime.json).

    Mostra il confronto prior/measured/calibrated con il peso shrinkage applicato.
    Per attivare la calibrazione nello scoring corrente, settare env var
    USE_CALIBRATED_SCORING=1 e riavviare il backend.
    """
    import os
    from app.services.scoring.calibration import load_calibration

    payload = load_calibration()
    if payload is None:
        raise HTTPException(
            status_code=404,
            detail="Calibrazione assente. Esegui POST /asset-calibration/run per generarla.",
        )

    use_calibrated = os.getenv("USE_CALIBRATED_SCORING", "0") in ("1", "true", "yes")
    return CalibrationResponse(
        version=payload["version"],
        calibrated_on=payload["calibrated_on"],
        n_classifications=payload["n_classifications"],
        params=payload["params"],
        diagnostics=[
            CalibrationDiagnostic(
                asset=d["asset"],
                regime=d["regime"],
                n_observations=d["n_observations"],
                weight_measured=d["weight_measured"],
                prior=d["prior"],
                measured=d["measured"],
                calibrated=d["calibrated"],
            )
            for d in payload["diagnostics"]
        ],
        use_calibrated_scoring=use_calibrated,
    )


@router.post("/asset-calibration/run")
def run_asset_calibration(
    horizon_months: int = Query(default=12, ge=3, le=36),
    regime_threshold: float = Query(default=0.35, ge=0.25, le=0.70),
    n_min: int = Query(default=8, ge=3, le=30),
    n_full: int = Query(default=40, ge=10, le=200),
    db: Session = Depends(get_db),
):
    """Rigenera la calibrazione e la persiste su disco.

    Triggerare manualmente dopo:
      - backfill storico modificato
      - aggiunta nuovi indicatori al classifier
      - cambio asset universe
    """
    from app.services.scoring.calibration import (
        CalibrationParams, calibrate, save_calibration,
    )
    from app.services.scoring.engine import reload_calibration

    params = CalibrationParams(
        horizon_months=horizon_months,
        regime_threshold=regime_threshold,
        n_min=n_min,
        n_full=n_full,
    )
    try:
        payload = calibrate(db, params)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    save_calibration(payload)
    reload_calibration()
    return {
        "status": "ok",
        "n_classifications": payload["n_classifications"],
        "n_diagnostics": len(payload["diagnostics"]),
        "calibrated_on": payload["calibrated_on"],
    }


@router.get("/regime/history", response_model=list[RegimeResponse])
def get_regime_history(
    days: int = Query(default=365 * 5, ge=1, le=365 * 60),
    db: Session = Depends(get_db),
):
    """Storico classificazioni regime."""
    records = (
        db.query(RegimeClassification)
        .order_by(RegimeClassification.date.desc())
        .limit(days)
        .all()
    )
    return [
        RegimeResponse(
            date=r.date,
            regime=r.regime,
            probabilities={
                "reflation": r.probability_reflation,
                "stagflation": r.probability_stagflation,
                "deflation": r.probability_deflation,
                "goldilocks": r.probability_goldilocks,
            },
            confidence=r.confidence,
            fit_scores=_extract_fit_scores(r),
        )
        for r in records
    ]


@router.get("/signals/current", response_model=list[SignalResponse])
def get_current_signals(db: Session = Depends(get_db)):
    """Score correnti per tutte le asset class."""
    latest_date = db.query(DailySignal.date).order_by(DailySignal.date.desc()).first()
    if not latest_date:
        raise HTTPException(status_code=404, detail="Nessun segnale disponibile")

    signals = (
        db.query(DailySignal)
        .filter(DailySignal.date == latest_date[0])
        .order_by(DailySignal.final_score.desc())
        .all()
    )
    return [
        SignalResponse(
            date=s.date,
            asset_class=s.asset_class,
            final_score=s.final_score,
            regime_component=s.regime_component,
            secular_trend_bonus=s.secular_trend_bonus,
            news_signal=s.news_signal,
            momentum_penalty=s.momentum_penalty,
        )
        for s in signals
    ]


@router.get("/scoreboard", response_model=ScoreboardResponse)
def get_scoreboard(
    include_dedollar: bool | None = Query(default=None, description="Override env: include dedollar bonus"),
    db: Session = Depends(get_db),
):
    """Dashboard completa: regime + tutti gli scores ordinati.

    Se `include_dedollar` e' specificato, gli score vengono ricalcolati on-the-fly
    con/senza il secular bonus dedollar. Se omesso, usa i valori in DB (calcolati
    dal scheduler in base all'env USE_DEDOLLAR_BONUS).
    """
    regime = (
        db.query(RegimeClassification)
        .order_by(RegimeClassification.date.desc())
        .first()
    )
    if not regime:
        raise HTTPException(status_code=404, detail="Nessun dato disponibile")

    if include_dedollar is None:
        # Path veloce: leggi gli score precalcolati dal DB
        signals = (
            db.query(DailySignal)
            .filter(DailySignal.date == regime.date)
            .order_by(DailySignal.final_score.desc())
            .all()
        )
        scores = {s.asset_class: s.final_score for s in signals}
    else:
        # Path on-the-fly: ricalcola applicando o saltando il bonus dedollar
        from app.services.dedollarization.scorer import calculate_secular_bonus

        probabilities = {
            "reflation": regime.probability_reflation,
            "stagflation": regime.probability_stagflation,
            "deflation": regime.probability_deflation,
            "goldilocks": regime.probability_goldilocks,
        }
        latest_sec = (
            db.query(SecularTrend)
            .filter(SecularTrend.trend_name == "dedollarization")
            .order_by(SecularTrend.date.desc())
            .first()
        )
        if latest_sec is not None:
            try:
                payload = json.loads(latest_sec.metadata_json) if latest_sec.metadata_json else {}
                dedollar_score = payload.get("combined_score", float(latest_sec.score))
            except Exception:
                dedollar_score = float(latest_sec.score)
        else:
            dedollar_score = 0.0
        secular_bonus = calculate_secular_bonus(dedollar_score)
        scores = calculate_final_scores(
            probabilities, secular_bonus,
            force_include_dedollar=bool(include_dedollar),
        )

    return ScoreboardResponse(
        date=regime.date,
        regime=regime.regime,
        confidence=regime.confidence,
        scores=scores,
    )


@router.get("/llm-settings")
def get_llm_settings():
    """Stato corrente settings LLM (API key masked, modello, source resolution)."""
    from app.services.llm import settings as llm_settings
    return llm_settings.get_current_state()


@router.post("/llm-settings")
def update_llm_settings(payload: dict = Body(...)):
    """Update runtime settings LLM (process-local + persistent file disco).

    Body:
      {
        "api_key": "AIza...new key..." | "" (= clear),
        "model": "gemini-2.5-pro" | "" (= clear)
      }

    Settings persistite in `.cache/llm_settings.json`. Override env var
    `GEMINI_API_KEY_MACRO` e `GEMINI_MODEL_MACRO` + global config.
    """
    from app.services.llm import settings as llm_settings

    if "api_key" in payload:
        new_key = payload["api_key"]
        if new_key is None or new_key == "":
            llm_settings.set_api_key(None)
        else:
            llm_settings.set_api_key(str(new_key).strip())

    if "model" in payload:
        new_model = payload["model"]
        if new_model is None or new_model == "":
            llm_settings.set_model(None)
        else:
            llm_settings.set_model(str(new_model).strip())

    return llm_settings.get_current_state()


@router.get("/macro-llm-analysis")
def get_macro_llm_analysis(
    force_refresh: bool = Query(default=False, description="Bypass cache 24h"),
    db: Session = Depends(get_db),
):
    """LLM macro analysis: regime + indicators → Gemini structured analysis.

    Cache disco TTL 24h per evitare Gemini call ad ogni reload (cost + latency).
    Cache invalidata su regime change (file key = `date_regime.json`).

    Returns:
        {headline, regime_thesis, key_drivers, risks, opportunities,
         time_horizon, confidence_qualitative, regime, classification_date,
         cached_at, provider, cache_hit}
        OR HTTP 503 se Gemini API key missing e nessuna cache disponibile.
    """
    from app.services.llm.macro_analyzer import get_macro_analysis

    latest = (
        db.query(RegimeClassification)
        .order_by(RegimeClassification.date.desc())
        .first()
    )
    if not latest:
        raise HTTPException(status_code=404, detail="Nessuna classificazione in DB")

    # Indicators + fit scores dal JSON payload
    payload = json.loads(latest.conditions_met) if latest.conditions_met else {}
    indicators = payload.get("indicators", {}) or {}
    fit_scores = payload.get("fit_scores", {}) or {}

    # Scoreboard corrente da DailySignal
    signals = (
        db.query(DailySignal)
        .filter(DailySignal.date == latest.date)
        .order_by(DailySignal.final_score.desc())
        .all()
    )
    scoreboard = {s.asset_class: s.final_score for s in signals}

    probabilities = {
        "reflation": float(latest.probability_reflation or 0),
        "stagflation": float(latest.probability_stagflation or 0),
        "deflation": float(latest.probability_deflation or 0),
        "goldilocks": float(latest.probability_goldilocks or 0),
    }

    result = get_macro_analysis(
        regime=latest.regime,
        probabilities=probabilities,
        confidence=float(latest.confidence or 0.5),
        indicators=indicators,
        scoreboard=scoreboard,
        classification_date=str(latest.date),
        fit_scores=fit_scores,
        force_refresh=force_refresh,
    )

    if result is None:
        raise HTTPException(
            status_code=503,
            detail="LLM analysis unavailable (Gemini API key missing or call failed)",
        )

    return result


# ============================================================================
# AI Portfolio (paper trading autonomo daily, sessione 2026-06-10)
# ============================================================================


@router.get("/ai-portfolio/positions")
def get_ai_portfolio_positions(
    strategy: str = Query(default="model_driven"),
    db: Session = Depends(get_db),
):
    """Posizioni attuali AI Portfolio + metriche per strategy_type."""
    from app.models.ai_portfolio import AiPortfolioPosition
    positions = (
        db.query(AiPortfolioPosition)
        .filter(AiPortfolioPosition.strategy_type == strategy)
        .order_by(AiPortfolioPosition.current_weight_pct.desc())
        .all()
    )
    return [
        {
            "asset_class": p.asset_class,
            "strategy_type": p.strategy_type,
            "target_weight_pct": p.target_weight_pct,
            "current_weight_pct": p.current_weight_pct,
            "tranches_filled": p.tranches_filled,
            "tranches_total": p.tranches_total,
            "entry_regime": p.entry_regime,
            "avg_entry_score": p.avg_entry_score,
            "estimated_pnl_pct": p.estimated_pnl_pct,
            "took_partial_tp": p.took_partial_tp,
            "opened_at": str(p.opened_at),
            "last_action": p.last_action,
            "last_action_date": str(p.last_action_date),
            "days_held": (date.today() - p.opened_at).days,
        }
        for p in positions
    ]


@router.get("/ai-portfolio/decisions")
def get_ai_portfolio_decisions(
    days: int = Query(default=30, ge=1, le=365),
    strategy: str = Query(default="model_driven"),
    db: Session = Depends(get_db),
):
    """Log decisioni AI Portfolio ultimi N giorni per strategy_type."""
    from datetime import timedelta
    from app.models.ai_portfolio import AiPortfolioDecision

    cutoff = date.today() - timedelta(days=days)
    decisions = (
        db.query(AiPortfolioDecision)
        .filter(
            AiPortfolioDecision.date >= cutoff,
            AiPortfolioDecision.strategy_type == strategy,
        )
        .order_by(AiPortfolioDecision.date.desc(), AiPortfolioDecision.id.desc())
        .limit(500)
        .all()
    )
    return [
        {
            "date": str(d.date),
            "asset_class": d.asset_class,
            "strategy_type": d.strategy_type,
            "action": d.action,
            "size_pct": d.size_pct,
            "tranche_index": d.tranche_index,
            "tranches_total": d.tranches_total,
            "score": d.score,
            "confidence": d.confidence,
            "regime": d.regime,
            "momentum_signal": d.momentum_signal,
            "vol_60d": d.vol_60d,
            "estimated_pnl_pct": d.estimated_pnl_pct,
            "reason_short": d.reason_short,
        }
        for d in decisions
    ]


@router.get("/ai-portfolio/performance")
def get_ai_portfolio_performance(
    days: int = Query(default=180, ge=1, le=1825),
    strategy: str = Query(default="model_driven"),
    db: Session = Depends(get_db),
):
    """Equity curve + DD + benchmarks (60/40 + S&P500) per strategy_type."""
    from datetime import timedelta
    from app.models.ai_portfolio import AiPortfolioPerformance

    cutoff = date.today() - timedelta(days=days)
    snaps = (
        db.query(AiPortfolioPerformance)
        .filter(
            AiPortfolioPerformance.date >= cutoff,
            AiPortfolioPerformance.strategy_type == strategy,
        )
        .order_by(AiPortfolioPerformance.date)
        .all()
    )
    latest = snaps[-1] if snaps else None
    return {
        "current": (
            {
                "date": str(latest.date),
                "nav": latest.nav,
                "cumulative_return_pct": latest.cumulative_return_pct,
                "drawdown_pct": latest.drawdown_pct,
                "peak_nav": latest.peak_nav,
                "n_positions": latest.n_positions,
                "deployed_pct": latest.deployed_pct,
                "top_position_asset": latest.top_position_asset,
                "top_position_weight_pct": latest.top_position_weight_pct,
                "benchmark_60_40_return_pct": latest.benchmark_60_40_return_pct,
                "benchmark_sp500_return_pct": latest.benchmark_sp500_return_pct,
            }
            if latest
            else None
        ),
        "history": [
            {
                "date": str(s.date),
                "nav": s.nav,
                "cumulative_return_pct": s.cumulative_return_pct,
                "drawdown_pct": s.drawdown_pct,
                "benchmark_60_40_return_pct": s.benchmark_60_40_return_pct,
                "benchmark_sp500_return_pct": s.benchmark_sp500_return_pct,
            }
            for s in snaps
        ],
    }


@router.post("/ai-portfolio/manual-scan")
def trigger_ai_portfolio_scan(db: Session = Depends(get_db)):
    """Trigger manuale daily scan ENTRAMBE le strategie + performance + LLM reasoning."""
    from app.services.ai_portfolio import (
        strategist,
        data_strategist,
        performance as perf,
        llm_reasoner,
    )
    results = {}
    for mod, label in (
        (strategist, "model_driven"),
        (data_strategist, "data_driven"),
    ):
        summary = mod.daily_scan(db)
        snap = perf.snapshot(db, strategy_type=label)
        try:
            llm_reasoner.generate_reasoning(db, strategy_type=label)
        except Exception:
            pass
        results[label] = {
            "scan_summary": summary,
            "performance": (
                {
                    "date": str(snap.date),
                    "nav": snap.nav,
                    "cumulative_return_pct": snap.cumulative_return_pct,
                    "n_positions": snap.n_positions,
                }
                if snap
                else None
            ),
        }
    return results


@router.get("/ai-portfolio/reasoning")
def get_ai_portfolio_reasoning(
    force_refresh: bool = Query(default=False),
    strategy: str = Query(default="model_driven"),
    db: Session = Depends(get_db),
):
    """LLM reasoning + regime challenge per ultima data disponibile per strategy."""
    from app.services.ai_portfolio import llm_reasoner
    import json as _json

    if force_refresh:
        r = llm_reasoner.generate_reasoning(db, force_refresh=True, strategy_type=strategy)
    else:
        r = llm_reasoner.latest_reasoning(db, strategy_type=strategy) or \
            llm_reasoner.generate_reasoning(db, strategy_type=strategy)

    if r is None:
        raise HTTPException(
            status_code=503,
            detail="AI Portfolio reasoning non disponibile",
        )

    try:
        dec_reasoning = _json.loads(r.decisions_reasoning_json or "{}")
    except Exception:
        dec_reasoning = {}

    return {
        "date": str(r.date),
        "daily_summary": r.daily_summary,
        "decisions_reasoning": dec_reasoning,
        "regime_classifier_says": r.regime_classifier_says,
        "ai_agrees": r.ai_agrees,
        "ai_alternative_regime": r.ai_alternative_regime,
        "regime_challenge_reasoning": r.regime_challenge_reasoning,
        "provider": r.provider,
        "cached_at": r.created_at.isoformat() if r.created_at else None,
    }


@router.get("/ai-portfolio/learnings")
def get_ai_portfolio_learnings(
    limit: int = Query(default=20, ge=1, le=100),
    strategy: str = Query(default="model_driven"),
    db: Session = Depends(get_db),
):
    """Lista learnings post-mortem aggregati per pattern per strategy."""
    from app.services.ai_portfolio import learning
    return learning.list_top_learnings(db, limit=limit, strategy_type=strategy)


@router.get("/ai-portfolio/leaderboard")
def get_ai_portfolio_leaderboard(db: Session = Depends(get_db)):
    """Confronto rapido: model_driven vs data_driven (NAV + alpha vs benchmarks)."""
    from app.models.ai_portfolio import AiPortfolioPerformance

    out = {}
    for strategy in ("model_driven", "data_driven"):
        latest = (
            db.query(AiPortfolioPerformance)
            .filter(AiPortfolioPerformance.strategy_type == strategy)
            .order_by(AiPortfolioPerformance.date.desc())
            .first()
        )
        if latest is None:
            out[strategy] = None
            continue
        out[strategy] = {
            "date": str(latest.date),
            "nav": latest.nav,
            "cumulative_return_pct": latest.cumulative_return_pct,
            "drawdown_pct": latest.drawdown_pct,
            "peak_nav": latest.peak_nav,
            "n_positions": latest.n_positions,
            "deployed_pct": latest.deployed_pct,
            "benchmark_60_40_return_pct": latest.benchmark_60_40_return_pct,
            "benchmark_sp500_return_pct": latest.benchmark_sp500_return_pct,
            "alpha_vs_60_40": (
                latest.cumulative_return_pct - (latest.benchmark_60_40_return_pct or 0)
            ),
            "alpha_vs_sp500": (
                latest.cumulative_return_pct - (latest.benchmark_sp500_return_pct or 0)
            ),
        }

    # Head-to-head: chi vince
    md = out.get("model_driven")
    dd = out.get("data_driven")
    head_to_head = None
    if md and dd:
        diff = md["cumulative_return_pct"] - dd["cumulative_return_pct"]
        head_to_head = {
            "winner": "model_driven" if diff > 0 else ("data_driven" if diff < 0 else "tie"),
            "spread_pct": diff,
            "spread_text": f"{abs(diff)*100:.2f}pp",
        }
    return {"by_strategy": out, "head_to_head": head_to_head}


@router.get("/fomc/report", response_model=FOMCReportResponse)
def get_fomc_report(
    limit: int = Query(default=6, ge=1, le=12),
    force_refresh: bool = Query(default=False),
):
    """FOMC statements/minutes con sentiment hawkish/dovish via LLM (Gemini → Groq fallback).

    Cache aggressiva su disco per URL. force_refresh=true forza ri-analisi.
    Trend = differenza tra primo e ultimo score (hawkening/dovening/stable).
    """
    from app.services.fomc.service import build_fomc_report, serialize_analysis

    report = build_fomc_report(limit=limit, force_refresh=force_refresh)
    return FOMCReportResponse(
        analyses=[FOMCAnalysisResponse(**serialize_analysis(a)) for a in report.analyses],
        latest_score=report.latest_score,
        avg_score_3last=report.avg_score_3last,
        trend=report.trend,
        n_documents=report.n_documents,
    )


@router.get("/indicators/term-premium", response_model=TermPremiumReportResponse)
def get_term_premium_report(
    threshold: float = Query(default=0.40, ge=0.20, le=0.80),
    days: int = Query(default=365 * 30, ge=365, le=365 * 60),
    db: Session = Depends(get_db),
):
    """Adrian-Crump-Moench term premium decomposition del 10Y yield.

    Decompone yield_10y_fitted ≈ expected_path + term_premium per ogni mese,
    poi aggrega per regime corrente. Permette di distinguere "Fed prevista
    hawkish" (path alto) da "mercato vuole risk premium" (term_premium alto).
    Fonte: NY Fed ACM model via FRED (THREEFY10, THREEFYTP10).
    """
    import math
    from app.services.indicators.term_premium import compute_term_premium_report

    try:
        r = compute_term_premium_report(db, threshold=threshold, days=days)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    def _nz(x: float) -> float | None:
        if x is None or (isinstance(x, float) and math.isnan(x)):
            return None
        return float(x)

    return TermPremiumReportResponse(
        common_period=r.common_period,
        threshold=r.threshold,
        n_observations=r.n_observations,
        points=[
            TermPremiumPointResponse(
                date=p.date, fitted_yield=p.fitted_yield,
                term_premium=p.term_premium, expected_path=p.expected_path,
            )
            for p in r.points
        ],
        by_regime=[
            TermPremiumStatResponse(
                regime=s.regime, n_observations=s.n_observations,
                mean_fitted=_nz(s.mean_fitted),
                mean_term_premium=_nz(s.mean_term_premium),
                mean_expected_path=_nz(s.mean_expected_path),
                pct_term_premium_positive=_nz(s.pct_term_premium_positive),
            )
            for s in r.by_regime
        ],
    )


@router.get("/factors/regime-mapping", response_model=FactorRegimeReportResponse)
def get_factor_regime_mapping(
    threshold: float = Query(default=0.40, ge=0.20, le=0.80,
                              description="Soglia prob_regime per considerare il regime 'attivo'"),
    db: Session = Depends(get_db),
):
    """Performance dei fattori Fama-French (Mkt-RF, SMB, HML, Mom) per regime macro.

    Per ogni fattore × regime, calcola mean_annual, vol_annual, Sharpe e win_rate
    sui mesi dove `prob(regime) >= threshold`. Fonte dati: Kenneth French Data Library
    (1926+, free). Permette ranking sub-equity (size/value/momentum) regime-conditional.
    """
    import math
    from app.services.factors.regime_mapping import compute_factor_regime_report

    try:
        r = compute_factor_regime_report(db, threshold=threshold)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    def _nz(x: float) -> float | None:
        if x is None or (isinstance(x, float) and math.isnan(x)):
            return None
        return float(x)

    return FactorRegimeReportResponse(
        threshold=r.threshold,
        n_months_analyzed=r.n_months_analyzed,
        factor_keys=r.factor_keys,
        regimes=r.regimes,
        common_period=r.common_period,
        stats=[
            FactorRegimeStatResponse(
                factor=s.factor, regime=s.regime, n_observations=s.n_observations,
                mean_annual=_nz(s.mean_annual), vol_annual=_nz(s.vol_annual),
                sharpe=_nz(s.sharpe), win_rate=_nz(s.win_rate),
            )
            for s in r.stats
        ],
    )


@router.get("/indicators/smoothed", response_model=SmoothedIndicatorResponse)
def get_smoothed_indicator(
    series_name: str = Query(..., description="Nome interno serie (es. unrate, initial_claims, lei)"),
    lam: float = Query(default=10.0, ge=0.5, le=200.0, alias="lambda",
                        description="R/Q ratio: alto=smoothing aggressivo, basso=segue raw"),
    days: int = Query(default=365 * 5, ge=180, le=365 * 60),
):
    """Kalman 1D smoothing su un indicatore macro rumoroso.

    Restituisce raw + filtered (real-time, causale) + smoothed (retrospective, RTS).
    Usato per validare visivamente quanto il rumore di una serie compromette la
    classificazione regime e per debug indicatori sospetti (outlier mensili).
    """
    from datetime import date as _date, timedelta
    import pandas as pd
    from app.services.indicators.kalman import (
        NOISY_INDICATORS, smooth_macro_series,
    )

    try:
        result = smooth_macro_series(series_name, lam=lam)
    except KeyError:
        raise HTTPException(status_code=400, detail=f"Serie sconosciuta: {series_name}")
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    cutoff = _date.today() - timedelta(days=days)
    raw = result.raw[result.raw.index >= pd.Timestamp(cutoff)]
    filt = result.filtered.loc[raw.index]
    smo = result.smoothed.loc[raw.index]

    points = [
        SmoothedSeriesPoint(
            date=idx.date(),
            raw=float(raw.loc[idx]),
            filtered=float(filt.loc[idx]),
            smoothed=float(smo.loc[idx]),
        )
        for idx in raw.index
    ]

    description = NOISY_INDICATORS.get(series_name, "")

    return SmoothedIndicatorResponse(
        series_name=series_name,
        description=description,
        lambda_used=result.lambda_used,
        n_points=len(points),
        variance_reduction=result.variance_reduction,
        points=points,
    )


@router.get("/indicators/smoothed/list")
def list_smoothed_indicators():
    """Lista degli indicatori macro per cui il Kalman smoothing e' particolarmente utile."""
    from app.services.indicators.kalman import NOISY_INDICATORS
    return [{"key": k, "description": v} for k, v in NOISY_INDICATORS.items()]


@router.get("/scoreboard/dedollar-comparison", response_model=ScoreboardDedollarComparison)
def get_scoreboard_dedollar_comparison(db: Session = Depends(get_db)):
    """Confronto side-by-side asset scores PURE (data-driven) vs ADJUSTED (con dedollar bonus).

    Ricalcola entrambe le viste on-the-fly dalle prob regime correnti + dedollar score.
    Indipendente dall'env var USE_DEDOLLAR_BONUS — l'utente vede sempre il delta cosi' da
    poter giudicare quanto la dedollarizzazione sposta il ranking.
    """
    from app.services.config_flags import use_dedollar_bonus
    from app.services.dedollarization.scorer import (
        ASSET_DEDOLLAR_SENSITIVITY,
        calculate_secular_bonus,
    )

    regime = (
        db.query(RegimeClassification)
        .order_by(RegimeClassification.date.desc())
        .first()
    )
    if not regime:
        raise HTTPException(status_code=404, detail="Nessuna classificazione in DB")

    probabilities = {
        "reflation": regime.probability_reflation,
        "stagflation": regime.probability_stagflation,
        "deflation": regime.probability_deflation,
        "goldilocks": regime.probability_goldilocks,
    }

    # Dedollar combined score corrente
    latest_sec = (
        db.query(SecularTrend)
        .filter(SecularTrend.trend_name == "dedollarization")
        .order_by(SecularTrend.date.desc())
        .first()
    )
    if latest_sec is None:
        dedollar_score = 0.0
    else:
        try:
            payload = json.loads(latest_sec.metadata_json) if latest_sec.metadata_json else {}
            dedollar_score = payload.get("combined_score", latest_sec.score)
        except Exception:
            dedollar_score = float(latest_sec.score)

    secular_bonus = calculate_secular_bonus(dedollar_score)
    pure_scores = calculate_final_scores(probabilities, secular_bonus, force_include_dedollar=False)
    adjusted_scores = calculate_final_scores(probabilities, secular_bonus, force_include_dedollar=True)

    items: list[ScoreComparisonItem] = []
    for asset in ASSET_CLASSES:
        pure = pure_scores.get(asset, 0.0)
        adj = adjusted_scores.get(asset, 0.0)
        items.append(ScoreComparisonItem(
            asset=asset,
            pure_score=pure,
            adjusted_score=adj,
            dedollar_delta=round(adj - pure, 2),
            sensitivity=ASSET_DEDOLLAR_SENSITIVITY.get(asset, 0.0),
        ))

    items.sort(key=lambda it: abs(it.dedollar_delta), reverse=True)

    return ScoreboardDedollarComparison(
        date=regime.date,
        regime=regime.regime,
        probabilities=probabilities,
        dedollar_combined_score=float(dedollar_score),
        use_dedollar_bonus_active=use_dedollar_bonus(),
        items=items,
    )


@router.post("/regime/classify")
def classify_from_indicators(
    indicators: dict[str, float],
    include_dedollar: bool | None = Query(default=None),
):
    """Classifica regime on-demand da indicatori forniti manualmente."""
    result = classify_regime(indicators)
    scores = calculate_final_scores(result["probabilities"], force_include_dedollar=include_dedollar)

    return {
        "regime": result["regime"],
        "probabilities": result["probabilities"],
        "confidence": result["confidence"],
        "scores": scores,
    }


@router.post("/refresh")
def trigger_refresh():
    """Trigger manuale del refresh dati."""
    from app.scheduler.jobs import daily_refresh

    try:
        daily_refresh()
        return {"status": "ok", "message": "Refresh completato"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Errore refresh: {str(e)}")


# ─────────────────────────────────────────────────────────────────────────
# CRITICAL #1+#2 council 2026-05-27: prediction log + model snapshot lock
# Endpoint per frontend "Live Track Record" + audit OOS validation.
# ─────────────────────────────────────────────────────────────────────────

@router.get("/prediction-log/stats")
def get_prediction_log_stats(
    horizon: str = Query(default="3m", regex="^(1m|3m|6m)$"),
    db: Session = Depends(get_db),
):
    """Aggregate stats P&L live model vs benchmark 60/40.

    Horizon: '1m' | '3m' | '6m'. Mostra mean_alpha, hit_rate, n_evaluated.
    Empty quando il tracking è appena iniziato (council: l'onestà è la feature).
    """
    from app.services.validation.prediction_logger import (
        stats_by_regime, summary_stats,
    )
    summary = summary_stats(db, horizon=horizon)
    by_regime = stats_by_regime(db, horizon=horizon)
    return {
        "horizon": horizon,
        "summary": summary,
        "by_regime": by_regime,
    }


@router.get("/prediction-log/recent")
def list_recent_predictions(
    limit: int = Query(default=30, ge=1, le=500),
    db: Session = Depends(get_db),
):
    """Lista predictions recenti (default 30 più nuove) con realized returns."""
    from app.models import PredictionLog
    rows = (
        db.query(PredictionLog)
        .order_by(PredictionLog.date_predicted.desc())
        .limit(limit)
        .all()
    )
    return [
        {
            "id": r.id,
            "date_predicted": r.date_predicted.isoformat() if r.date_predicted else None,
            "regime": r.regime,
            "confidence": r.confidence,
            "model_version": r.model_version,
            "allocation_mode": r.allocation_mode,
            "realized_return_1m": r.realized_return_1m,
            "realized_return_3m": r.realized_return_3m,
            "realized_return_6m": r.realized_return_6m,
            "alpha_1m": r.alpha_1m,
            "alpha_3m": r.alpha_3m,
            "alpha_6m": r.alpha_6m,
            "evaluated_at": r.evaluated_at.isoformat() if r.evaluated_at else None,
        }
        for r in rows
    ]


@router.get("/model/active")
def get_active_model_version():
    """Ritorna versione del modello ATTIVO + classifier hash + flag count."""
    from app.services.validation.model_lock import (
        compute_classifier_hash, get_active_version, list_snapshots,
    )
    return {
        "active_version": get_active_version(),
        "current_classifier_hash": compute_classifier_hash(),
        "snapshots": list_snapshots(),
    }


@router.get("/prediction-log/post-mortem/{prediction_id}")
def get_post_mortem(
    prediction_id: int,
    horizon: str = Query(default="3m", regex="^(1m|3m|6m)$"),
    db: Session = Depends(get_db),
):
    """NEW council 2026-05-27: post-mortem analysis singola failure.

    Reverse engineering: identifica root cause, findings interpretati,
    suggestions per migliorare il modello.
    """
    from app.services.validation.reverse_engineering import post_mortem
    pm = post_mortem(db, prediction_id, horizon=horizon)
    if pm is None:
        raise HTTPException(status_code=404, detail="Prediction non trovata o non valutata")
    return {
        "id": pm.prediction_id,
        "date_predicted": pm.date_predicted.isoformat() if pm.date_predicted else None,
        "regime": pm.regime,
        "confidence": pm.confidence,
        "horizon": pm.horizon,
        "alpha": pm.alpha,
        "severity": pm.severity,
        "regime_was_boundary": pm.regime_was_boundary,
        "confidence_was_low": pm.confidence_was_low,
        "top5_assets": pm.top5_assets,
        "findings": pm.findings,
        "suggestions": pm.suggestions,
    }


@router.post("/ml/retrain")
def trigger_ml_retrain():
    """NEW council 2026-05-27 #5: trigger ML blend retrain.

    Re-fitta RandomForest + GradientBoosting su HISTORICAL_EPISODES corrente.
    Use case: dopo aver aggiunto nuovi episodi 2025-2026 (COVID-aftermath,
    AI capex), retrain per riflettere distribution drift.

    Output: metriche LOO CV + Brier + n_episodes addestrato.
    """
    from app.services.regime.historical_episodes import HISTORICAL_EPISODES
    from app.services.regime.ml_classifier import train_classifier
    try:
        bundle = train_classifier(episodes=list(HISTORICAL_EPISODES))
        return {
            "status": "ok",
            "n_episodes_train": len(HISTORICAL_EPISODES),
            "n_features": len(bundle.feature_names) if hasattr(bundle, "feature_names") else None,
            "loo_accuracy": getattr(bundle, "loo_accuracy", None),
            "brier_mean": getattr(bundle, "brier_mean", None),
            "model_classes": getattr(bundle, "classes", []),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"ML retrain failed: {str(e)}")


@router.get("/prediction-log/learnings")
def get_learnings(
    horizon: str = Query(default="3m", regex="^(1m|3m|6m)$"),
    db: Session = Depends(get_db),
):
    """NEW council 2026-05-27: aggregate failure patterns + top actions.

    Use case: quarterly review console. L'utente legge questa lista per
    decidere quali patterns indagare manualmente per migliorare il modello.
    """
    from app.services.validation.reverse_engineering import (
        generate_improvement_suggestions, post_mortem_recent_failures,
    )
    summary = generate_improvement_suggestions(db, horizon=horizon)
    recent = post_mortem_recent_failures(db, horizon=horizon, limit=10)
    return {"summary": summary, "recent_failures": recent}


@router.post("/model/lock")
def lock_model_endpoint(
    version: str = Query(..., min_length=3, max_length=64),
    description: str | None = Query(default=None),
):
    """Lock manualmente un nuovo snapshot del modello.

    Use case: dopo update flag importante o classifier change, lock una
    nuova versione (es. "v1.1-sealed-2026-07-01") per separare le predictions
    future dallo storico v1.0.
    """
    from app.services.validation.model_lock import lock_current_model
    try:
        info = lock_current_model(version=version, description=description)
        return {"status": "ok", **info}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Lock failed: {str(e)}")


@router.post("/regime/backfill")
def trigger_regime_backfill(days: int = Query(default=365, ge=1, le=3650)):
    """Ricostruisce lo storico classificazioni regime per gli ultimi N giorni."""
    from app.services.regime.backfill import backfill_regime_history

    try:
        stats = backfill_regime_history(days=days)
        return {
            "status": "ok",
            "classified": stats["classified"],
            "skipped": stats["skipped"],
            "errors": stats["errors"],
            "start": stats["start"].isoformat(),
            "end": stats["end"].isoformat(),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Errore backfill: {str(e)}")


@router.post("/regime/backfill/historical")
def trigger_regime_backfill_historical(
    start: date = Query(default=date(1970, 1, 1)),
    end: date | None = Query(default=None),
    step_days: int = Query(default=30, ge=1, le=365),
):
    """Backfill storico a lungo raggio (training HMM / transition matrix).

    Default 1970-01-01 → oggi con step mensile. Usa rule-based classifier
    sulle serie FRED troncate as-of. I pillar minimi ({gdp_roc, cpi_yoy, unrate})
    coprono l'intervallo 1948-oggi; indicatori piu recenti (LEI '82, BAA '86)
    vengono semplicemente ignorati quando non disponibili.
    """
    from app.services.regime.backfill import backfill_regime_history_long

    try:
        stats = backfill_regime_history_long(
            start_date=start, end_date=end, step_days=step_days
        )
        return {
            "status": "ok",
            "classified": stats["classified"],
            "skipped": stats["skipped"],
            "errors": stats["errors"],
            "start": stats["start"].isoformat(),
            "end": stats["end"].isoformat(),
            "step_days": stats["step_days"],
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Errore backfill storico: {str(e)}")


@router.post("/backfill/all")
def trigger_full_backfill(days: int = Query(default=365, ge=30, le=3650)):
    """Backfill completo: regime + dedollarization + asset scores + prune."""
    from app.services.backfill import backfill_all

    try:
        stats = backfill_all(days=days)
        return {
            "status": "ok",
            "days": stats["days"],
            "regime": {
                "classified": stats["regime"]["classified"],
                "skipped": stats["regime"]["skipped"],
                "errors": stats["regime"]["errors"],
            },
            "dedollar": {
                "classified": stats["dedollar"]["classified"],
                "skipped": stats["dedollar"]["skipped"],
                "errors": stats["dedollar"]["errors"],
            },
            "asset_scores": {
                "written": stats["asset_scores"]["written"],
                "errors": stats["asset_scores"]["errors"],
            },
            "pruned": stats["pruned"],
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Errore backfill completo: {str(e)}")


@router.get("/regime/crisis-risk")
def get_crisis_risk(db: Session = Depends(get_db)):
    """Crisis Risk Indicator (Tier 3 dashboard): livello rischio + tipo crisi
    + asset positioning.

    Identifica 3 tipi di crisi (NON solo crash deflattivi):
    - deflation_crash (2008/2020): GDP↓ + CPI↓ + credit spread↑ + VIX spike
    - stagflation_debasement (1973/2022): CPI↑ + dedollar↑ + bonds crollano
    - bubble_goldilocks (2000/2021): VIX compresso + complacency mean-reversion

    Per ogni tipo restituisce positioning suggerito (overweight/neutral/underweight).
    """
    import json
    from app.models import RegimeClassification
    from app.services.regime.crisis_indicator import assess_crisis_risk

    last = (
        db.query(RegimeClassification)
        .order_by(RegimeClassification.date.desc()).first()
    )
    if last is None:
        raise HTTPException(status_code=404, detail="Nessuna classificazione in DB")
    meta = json.loads(last.conditions_met) if last.conditions_met else {}
    indicators = meta.get("indicators", {}) or {}

    # Pesca dedollar combined dall'ultimo SecularTrend record (= stesso storage
    # usato da /dedollarization endpoint).
    dedollar_combined = None
    try:
        latest_dedollar = (
            db.query(SecularTrend)
            .filter(SecularTrend.trend_name == "dedollarization")
            .order_by(SecularTrend.date.desc())
            .first()
        )
        if latest_dedollar is not None:
            payload = json.loads(latest_dedollar.components) if latest_dedollar.components else {}
            dedollar_combined = float(
                payload.get("combined_score") or latest_dedollar.score or 0.0
            )
    except Exception:
        # Graceful: se dedollar non disponibile, continua senza (crisis_indicator
        # accetta None come default).
        pass

    result = assess_crisis_risk(
        indicators,
        dedollar_combined=dedollar_combined,
        date=str(last.date),
    )

    return {
        "date": result.date,
        "risk_level": result.risk_level,
        "risk_score": result.risk_score,
        "crisis_type": result.crisis_type,
        "crisis_type_confidence": result.crisis_type_confidence,
        "summary": result.summary,
        "historical_analog": result.historical_analog,
        "positioning": result.positioning,
        "triggers_fired": [t for t in result.triggers if t["fired"]],
        "triggers_all": result.triggers,
        "dedollar_combined": dedollar_combined,
        "indicators_snapshot": result.indicators_snapshot,
        "notes": result.notes,
    }


@router.get("/config/flags")
def get_config_flags():
    """Snapshot di tutti i feature flag noti con source (runtime/env/default).

    Frontend FeatureFlagToggle usa questo per popolare lo stato corrente.
    """
    from app.services.config_flags import get_all_flags_state
    return get_all_flags_state()


class FlagUpdate(BaseModel):
    name: str
    value: bool | None  # None = clear runtime override


@router.post("/config/flags")
def set_config_flag(update: FlagUpdate):
    """Set/clear runtime override per un feature flag (process-local).

    Persiste finchè backend resta up. NO file/DB write — restart = reset.
    Per persistenza permanente: usa env var.
    """
    from app.services.config_flags import get_all_flags_state, set_runtime_flag
    known = set(get_all_flags_state().keys())
    if update.name not in known:
        raise HTTPException(status_code=400, detail=f"Unknown flag: {update.name}")
    set_runtime_flag(update.name, update.value)
    return {"ok": True, "flags": get_all_flags_state()}


@router.get("/portfolio/allocation")
def get_portfolio_allocation(
    method: str = Query("equal_weight", description="equal_weight | score_weighted | risk_parity | kelly_fractional"),
    top_n: int = Query(5, ge=1, le=15),
    fraction: float = Query(0.25, ge=0.05, le=1.0, description="Solo per kelly_fractional"),
    vol_target: float | None = Query(None, ge=0.02, le=0.40, description="T6.10: target annualized vol (es. 0.10 = 10%). Se None, no vol targeting."),
    avg_correlation: float = Query(0.3, ge=0.0, le=1.0, description="T6.10: correlazione media assumed cross-asset"),
    max_leverage: float = Query(2.0, ge=1.0, le=5.0, description="T6.10: cap leverage scale factor"),
    db: Session = Depends(get_db),
):
    """Tier 6.4 — Position sizing layer.

    Converte gli ultimi asset scores in pesi portfolio % via 4 algoritmi.
    Output uniforme: weights dict (sum=1), method, top_n, metadata.

    Risk-parity e Kelly_fractional richiedono vols → estratti da
    ASSET_REGIME_DATA (vol media weighted regime).
    """
    from app.services.config_flags import use_position_sizing_layer
    if not use_position_sizing_layer():
        raise HTTPException(
            status_code=403,
            detail="USE_POSITION_SIZING_LAYER flag richiesto (opt-in T6.4)",
        )

    from app.models import DailySignal, RegimeClassification
    from app.services.portfolio.position_sizing import compute_allocation
    from app.services.scoring.engine import ASSET_REGIME_DATA

    # Carica ultimi asset scores da DailySignal
    last_classification = (
        db.query(RegimeClassification)
        .order_by(RegimeClassification.date.desc())
        .first()
    )
    if last_classification is None:
        raise HTTPException(status_code=404, detail="No classification available")

    signals = (
        db.query(DailySignal)
        .filter(DailySignal.date == last_classification.date)
        .all()
    )
    if not signals:
        raise HTTPException(status_code=404, detail="No daily signals available")

    scores = {s.asset_class: float(s.final_score) for s in signals if s.final_score is not None}
    if not scores:
        raise HTTPException(status_code=404, detail="No valid scores")

    # Vols: media weighted regime per asset
    vols = {}
    for asset in scores:
        regime_data = ASSET_REGIME_DATA.get(asset, {})
        if regime_data:
            avg_vol = sum(r.get("vol", 0.15) for r in regime_data.values()) / len(regime_data)
            vols[asset] = avg_vol

    try:
        allocation = compute_allocation(
            method=method,
            scores=scores,
            vols=vols if method in {"risk_parity", "kelly_fractional"} else None,
            top_n=top_n,
            fraction=fraction,
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    base_response = {
        "method": allocation.method,
        "top_n": allocation.top_n,
        "weights": allocation.weights,
        "total_weight": allocation.total_weight,
        "excluded_assets": allocation.excluded_assets,
        "description": allocation.description,
        "metadata": allocation.metadata,
        "as_of": str(last_classification.date),
    }

    # T6.10: vol targeting + leverage post position-sizing.
    # Attivo se vol_target esplicitamente passato OR flag globale ON.
    from app.services.config_flags import use_vol_targeting
    if vol_target is not None or use_vol_targeting():
        from app.services.portfolio.position_sizing import apply_vol_targeting
        target = vol_target if vol_target is not None else 0.10
        vt = apply_vol_targeting(
            allocation,
            vols=vols,
            target_vol=target,
            avg_correlation=avg_correlation,
            max_leverage=max_leverage,
        )
        base_response["vol_targeting"] = {
            "target_vol": vt.target_vol,
            "realized_vol_pre": vt.realized_vol_pre,
            "realized_vol_post": vt.realized_vol_post,
            "scale_factor": vt.scale_factor,
            "weights_scaled": vt.weights,
            "leverage_used": vt.leverage_used,
            "cash_buffer": vt.cash_buffer,
            "description": vt.description,
        }

    return base_response


@router.get("/portfolio/methods")
def list_portfolio_methods():
    """Lista algoritmi position sizing disponibili (T6.4)."""
    from app.services.portfolio.position_sizing import list_methods
    return {"methods": list_methods()}


@router.get("/scoring/calibration/snapshots")
def get_calibration_snapshots(
    limit: int = Query(24, ge=1, le=200),
    db: Session = Depends(get_db),
):
    """Tier 6.6 — Lista snapshot history calibration `asset_regime_data`.

    Ogni snapshot = run mensile scheduler. Solo l'ultimo è `is_active=True`.
    """
    from app.services.scoring.live_calibration import list_snapshots
    return list_snapshots(db, limit=limit)


@router.post("/scoring/calibration/refresh")
def trigger_calibration_refresh(db: Session = Depends(get_db)):
    """Tier 6.6 — Force refresh manuale della live calibration.

    Utile per testing/dev. Il scheduler la chiama auto il 1° del mese.
    """
    from app.services.scoring.live_calibration import calibrate_and_persist
    result = calibrate_and_persist(db)
    if result is None:
        raise HTTPException(status_code=500, detail="Calibration refresh failed")
    return {
        "version": result.version,
        "calibrated_on": str(result.calibrated_on),
        "n_classifications": result.n_classifications,
        "n_assets": result.n_assets,
    }


class OverrideCreate(BaseModel):
    target_type: str
    target_name: str
    delta: float
    expires_at: datetime
    reason: str
    author: str = "anonymous"


@router.get("/scoring/overrides")
def list_overrides(db: Session = Depends(get_db)):
    """Tier 6.7 — Lista override discrezionali attivi (expires_at > now)."""
    from app.services.scoring.discretionary import load_active_overrides
    overrides = load_active_overrides(db)
    return [
        {
            "id": o.id,
            "target_type": o.target_type,
            "target_name": o.target_name,
            "delta": o.delta,
            "expires_at": o.expires_at.isoformat() if o.expires_at else None,
            "reason": o.reason,
            "author": o.author,
            "created_at": o.created_at.isoformat() if o.created_at else None,
        }
        for o in overrides
    ]


@router.post("/scoring/overrides")
def post_override(payload: OverrideCreate, db: Session = Depends(get_db)):
    """Tier 6.7 — Crea nuovo override discrezionale (audit-trail).

    target_type: 'asset' (additivo a score 0-100) | 'regime' (shift probs Σ=1).
    delta: range tipico ±30 per asset, ±0.20 per regime.
    expires_at: timestamp UTC dopo cui l'override è inattivo (auto-rollback).
    """
    from app.services.scoring.discretionary import create_override
    try:
        rec = create_override(
            db,
            target_type=payload.target_type,
            target_name=payload.target_name,
            delta=payload.delta,
            expires_at=payload.expires_at,
            reason=payload.reason,
            author=payload.author,
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    return {
        "id": rec.id,
        "target_type": rec.target_type,
        "target_name": rec.target_name,
        "delta": rec.delta,
        "expires_at": rec.expires_at.isoformat(),
        "reason": rec.reason,
        "author": rec.author,
    }


@router.get("/regime/transition-check")
def get_transition_check(db: Session = Depends(get_db)):
    """Tier 9 — Verifica se il regime corrente è in transizione confusionaria
    e suggerisce allocazione difensiva.

    Triggers (2 di 3 fired → transizione):

    - max(regime_probs) < 0.45 (no dominant regime)
    - entropy(probs) > 1.40 (high uncertainty)
    - confidence < 0.40 (model unsure)

    Quando in transizione, Buffett/Taleb/raydalio dicono: meglio cash + bonds +
    gold che inseguire un regime ambiguo.
    """
    # Performance fix 2026-05-28: usa DB cached regime invece di ri-classificare.
    # classify_regime() ricalcola tutti i pillar + ML blend + override = lento.
    # DB ha già le probs dell'ultima classification (scheduler giornaliero).
    from app.services.regime.transition_detector import (
        assess_transition,
        get_defensive_allocation,
    )

    latest = db.query(RegimeClassification).order_by(
        RegimeClassification.date.desc()
    ).first()
    if latest is None:
        raise HTTPException(status_code=404, detail="No regime classification found")

    probs = {
        "reflation": float(latest.probability_reflation or 0),
        "stagflation": float(latest.probability_stagflation or 0),
        "deflation": float(latest.probability_deflation or 0),
        "goldilocks": float(latest.probability_goldilocks or 0),
    }
    confidence = float(latest.confidence or 0.5)

    assessment = assess_transition(probs=probs, confidence=confidence)
    defensive = get_defensive_allocation()

    return {
        "date": str(latest.date),
        "current_regime": latest.regime,
        "current_probs": {k: round(v, 4) for k, v in probs.items()},
        "confidence": confidence,
        "is_transition": assessment.is_transition,
        "triggers_fired": assessment.triggers_fired,
        "triggers_count": assessment.triggers_count,
        "max_prob": assessment.max_prob,
        "entropy": assessment.entropy,
        "description": assessment.description,
        "defensive_allocation": defensive if assessment.is_transition else None,
    }


@router.get("/regime/ml-forecast")
def get_ml_forecast(
    horizons: str = Query("1,3,6", description="Comma-separated months (e.g. '1,3,6,12')"),
    top_n: int = Query(5, ge=1, le=15),
    use_percentile: bool = Query(False, description="Use T7.2 rank percentile scoring"),
    db: Session = Depends(get_db),
):
    """Tier 8.4 — Forecast regime + asset allocation prossimi N mesi.

    Pipeline:
    1. Legge latest RegimeClassification + indicators.
    2. Se USE_ML_REGIME_BLEND ON: blend rule+ML per probs t0.
    3. Stima transition matrix monthly da DB (1027 records storici).
    4. Propaga probs t0 via T^k per ogni horizon.
    5. Calcola top-N asset weighted by forecasted regime probs.

    Risposta inclusiva: ottieni dove allocare oggi E dove sarà profittevole
    allocare tra 1/3/6 mesi se il regime evolve come storicamente.
    """
    from app.services.regime.classifier import classify_regime
    from app.services.regime.ml_forecast import (
        asset_allocation_forecast,
        estimate_transition_matrix,
        forecast_horizons,
    )
    from app.services.scoring.engine import (
        ASSET_CLASSES,
        _asset_regime_score,
        _asset_regime_score_percentile,
    )

    latest = db.query(RegimeClassification).order_by(
        RegimeClassification.date.desc()
    ).first()
    if latest is None:
        raise HTTPException(status_code=404, detail="No regime classification found")

    indicators_json = latest.conditions_met or {}
    if isinstance(indicators_json, str):
        try:
            indicators_json = json.loads(indicators_json)
        except (ValueError, TypeError):
            indicators_json = {}
    indicators = (
        indicators_json.get("indicators", {})
        if isinstance(indicators_json, dict)
        else {}
    )

    # Re-classify per ottenere probs aggiornate (con eventual flag ML attivo)
    out = classify_regime(indicators)
    current_probs = out["probabilities"]

    # Transition matrix da DB
    T_matrix = estimate_transition_matrix(db)

    # Parse horizons
    try:
        h_list = tuple(sorted(set(int(h.strip()) for h in horizons.split(",") if h.strip())))
        h_list = tuple(h for h in h_list if 1 <= h <= 24)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid horizons format")
    if not h_list:
        h_list = (1, 3, 6)

    forecasts = forecast_horizons(current_probs, T_matrix, h_list)

    # Asset allocation forecast
    score_fn = _asset_regime_score_percentile if use_percentile else _asset_regime_score
    asset_top = asset_allocation_forecast(forecasts, score_fn, ASSET_CLASSES, top_n)

    return {
        "date": str(latest.date),
        "current_regime": out["regime"],
        "current_probs": {k: round(v, 4) for k, v in current_probs.items()},
        "ml_blend_active": "ml_blend" in out,
        "ml_blend_metadata": out.get("ml_blend"),
        "forecasts": [
            {
                "horizon_months": fc.horizon_months,
                "probs": fc.probs,
                "expected_regime": fc.expected_regime,
                "entropy": fc.entropy,
                "top_assets": [
                    {"asset": a, "weighted_score": s}
                    for a, s in asset_top.get(fc.horizon_months, [])
                ],
            }
            for fc in forecasts
        ],
    }


# /regime/regime-2d endpoint rimosso 2026-05-28 (code audit, no frontend consumer)


# /regime/sub-regime endpoint rimosso 2026-05-28 (code audit, no frontend consumer)


@router.delete("/scoring/overrides/{override_id}")
def remove_override(override_id: int, db: Session = Depends(get_db)):
    """Tier 6.7 — Cancella override per id."""
    from app.services.scoring.discretionary import delete_override
    ok = delete_override(db, override_id)
    if not ok:
        raise HTTPException(status_code=404, detail="Override not found")
    return {"deleted": override_id}


# /regime/correlation-regime endpoint rimosso 2026-05-28 (code audit, no consumer)


@router.get("/regime/vol-regime")
def get_vol_regime(db: Session = Depends(get_db)):
    """Tier 5.7 LOOP CLOSURE — Vol regime separato + change-point detection.

    Classifica VIX corrente in 4 vol regimes (calm/normal/elevated/spike) +
    detect change-point via rolling z-score 20gg. Sub-regime informativo,
    ortogonale al regime 4-quadranti.
    """
    import json
    from app.models import RegimeClassification, MacroIndicator
    from app.services.regime.vol_regime import assess_vol_regime

    last = (
        db.query(RegimeClassification)
        .order_by(RegimeClassification.date.desc()).first()
    )
    if last is None:
        raise HTTPException(status_code=404, detail="Nessuna classificazione in DB")
    meta = json.loads(last.conditions_met) if last.conditions_met else {}
    indicators = meta.get("indicators", {}) or {}
    vix = indicators.get("vix")
    if vix is None:
        raise HTTPException(status_code=404, detail="VIX non disponibile in indicators")

    # Calcola ma20 + std20 dalla MacroIndicator history (VIX daily)
    vix_ma20 = None
    vix_std20 = None
    try:
        recent_vix = (
            db.query(MacroIndicator)
            .filter(MacroIndicator.code == "vix")
            .order_by(MacroIndicator.date.desc())
            .limit(20)
            .all()
        )
        values = [float(r.value) for r in recent_vix if r.value is not None]
        if len(values) >= 5:
            vix_ma20 = sum(values) / len(values)
            mean = vix_ma20
            var = sum((v - mean) ** 2 for v in values) / len(values)
            vix_std20 = var ** 0.5
    except Exception:
        pass

    result = assess_vol_regime(float(vix), vix_ma20=vix_ma20, vix_std20=vix_std20)
    return {
        "date": str(last.date),
        "vol_regime": result.vol_regime,
        "vol_score": result.vol_score,
        "vix": result.vix,
        "vix_ma20": result.vix_ma20,
        "vix_std20": result.vix_std20,
        "change_point": result.change_point,
        "change_magnitude": result.change_magnitude,
        "description": result.description,
    }


@router.get("/regime/crisis-risk/history")
def get_crisis_risk_history(
    days: int = Query(365 * 5, ge=30, le=365 * 30),
    limit: int | None = Query(None, ge=1, le=5000),
    db: Session = Depends(get_db),
):
    """Backtest storico Crisis Risk Indicator.

    Applica `assess_crisis_risk` su ogni record `RegimeClassification`
    della finestra `days` (default 5y, max 30y). Restituisce time series
    + statistiche aggregate.

    Validazione: il crisis indicator dovrebbe accendersi durante episodi
    storici noti (2008-09 GFC, 2020 COVID, 1973-74 oil shock, 2022 stag,
    2000-01 dotcom). Indicatori pre-2000 hanno copertura parziale
    (no VIX/NFCI/breakeven) — segnale meno affidabile.
    """
    from app.services.regime.crisis_indicator_history import (
        aggregate_crisis_stats,
        compute_crisis_history,
        serialize_points,
    )

    try:
        points = compute_crisis_history(db, days=days, limit=limit)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Crisis history failed: {str(e)}")

    stats = aggregate_crisis_stats(points)

    return {
        "n_points": stats["n_points"],
        "date_from": stats.get("date_from"),
        "date_to": stats.get("date_to"),
        "level_distribution": stats["level_distribution"],
        "type_distribution": stats["type_distribution"],
        "notable_episodes": stats["notable_episodes"],
        "points": serialize_points(points),
    }


@router.get("/alt-data/insider-activity")
def get_insider_activity(
    days: int = Query(30, ge=1, le=90),
    max_filings_per_day: int = Query(40, ge=5, le=200),
    end_date: str | None = Query(None, description="Override end date (YYYY-MM-DD), default=oggi"),
):
    """SEC EDGAR Form 4 insider trading aggregate (smart money signal).

    Aggrega open-market purchases (code P) e sales (code S) di ufficiali,
    direttori e holder >10% di tutte le aziende US filing Form 4 nella
    finestra `days`. Esclude option exercises (M), grants (A) e tax
    withholdings (F).

    Output:
    - `total_buy_usd` / `total_sell_usd` / `buy_sell_ratio` / `net_flow_usd`
    - `insider_score` ∈ [-1, +1] (tanh-bounded). Score > 0.3 = bullish strong,
      < -0.3 = bearish, ~0 = neutral.
    - `top_buy_tickers` / `top_sell_tickers`

    NOTE: prima chiamata può essere lenta (fino a 30+ min full fetch per 30
    giorni × 40 filings). I dati cached su disco sono permanenti (SEC filings
    immutable). Scheduler aggiorna cache in background.
    """
    from datetime import datetime
    from app.services.alt_data.sec_form4 import compute_insider_activity

    end: date | None = None
    if end_date:
        try:
            end = datetime.strptime(end_date, "%Y-%m-%d").date()
        except ValueError:
            raise HTTPException(status_code=400, detail="end_date deve essere YYYY-MM-DD")

    try:
        report = compute_insider_activity(
            days=days,
            end_date=end,
            max_filings_per_day=max_filings_per_day,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Insider activity fetch failed: {str(e)}")

    return {
        "period_start": str(report.period_start) if report.period_start else None,
        "period_end": str(report.period_end) if report.period_end else None,
        "n_transactions": report.n_transactions,
        "n_buy_filings": report.n_buy_filings,
        "n_sell_filings": report.n_sell_filings,
        "total_buy_usd": report.total_buy_usd,
        "total_sell_usd": report.total_sell_usd,
        "buy_sell_ratio": report.buy_sell_ratio,
        "net_flow_usd": report.net_flow_usd,
        "insider_score": report.insider_score,
        "top_buy_tickers": [{"ticker": t, "usd": v} for t, v in report.top_buy_tickers],
        "top_sell_tickers": [{"ticker": t, "usd": v} for t, v in report.top_sell_tickers],
        "notes": report.notes,
    }


@router.get("/backtest/lead-time-sensitivity")
def get_lead_time_sensitivity(db: Session = Depends(get_db)):
    """Sensitivity grid del lead-time NBER: hit_rate × avg_lead variando
    threshold (0.30-0.50) e lookback (6-24m). Test contro data mining:
    se cambiando i parametri il risultato resta stabile, segnale robusto.
    """
    from app.services.backtest.robustness import lead_time_sensitivity_grid

    try:
        return lead_time_sensitivity_grid(db)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Sensitivity grid failed: {str(e)}")


@router.get("/backtest/walk-forward")
def get_walk_forward(
    test_window_months: int = Query(default=36, ge=24, le=120),
    step_months: int = Query(default=12, ge=6, le=24),
    db: Session = Depends(get_db),
):
    """Walk-forward OOS evaluation della macro_score_weighted strategy.
    Rolling windows + statistiche stability (sharpe_std, verdict).
    Test contro overfitting: se Sharpe oscilla poco, robusto; molto, fragile.
    """
    from datetime import date as _date
    from app.services.backtest.robustness import walk_forward_backtest

    try:
        return walk_forward_backtest(
            db,
            window_start=_date(2005, 1, 1),
            window_end=_date.today(),
            test_window_months=test_window_months,
            step_months=step_months,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Walk-forward failed: {str(e)}")


@router.get("/indicators/gdp-nowcast")
def get_gdp_nowcast(n_factors: int = Query(default=1, ge=1, le=3)):
    """DFM nowcast GDP YoY (Tier 2.5 roadmap Bridgewater).

    Estrae fattore latente da 5 coincident indicators FRED (industrial production,
    nonfarm payrolls, retail sales, consumer sentiment, housing starts), mappa a
    GDP YoY via OLS. Output mensile vs trimestrale lagged del classico `gdp_roc`.
    """
    from app.services.indicators.dfm_nowcast import compute_gdp_nowcast

    try:
        r = compute_gdp_nowcast(n_factors=n_factors)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"GDP nowcast failed: {str(e)}")

    if r is None:
        raise HTTPException(status_code=503, detail="DFM fit failed (insufficient data)")

    return {
        "latest_nowcast_date": r.latest_nowcast_date,
        "latest_factor_value": r.latest_factor_value,
        "latest_gdp_yoy_implied": r.latest_gdp_yoy_implied,
        "ols_alpha": r.ols_alpha,
        "ols_beta": r.ols_beta,
        "ols_r_squared": r.ols_r_squared,
        "series_used": r.series_used,
        "n_history_points": len(r.nowcast_history),
        "history_tail": r.nowcast_history[-24:],   # ultimi 2 anni in payload
        "notes": r.notes,
    }


@router.get("/regime/meta")
def get_regime_meta(
    lookback_months: int = Query(default=60, ge=12, le=240),
    min_weight: float = Query(default=0.10, ge=0.0, le=0.5),
    prob_floor: float = Query(default=0.02, ge=0.0, le=0.25),
    db: Session = Depends(get_db),
):
    """Meta-regime classifier: Brier-weighted blend di rule-based + HMM-Market + MS-VAR.

    Tier 1.1 della roadmap Bridgewater. Sostituisce il regime "single rule-based"
    con un ensemble pesato per Brier score storico. Output = blend dei 3 modelli
    con pesi calibrati sulla loro accuratezza.

    Caveat: rule-based è ground truth → suo Brier ~0 → peso dominante.
    `min_weight` (default 0.10) garantisce gli altri modelli almeno 10% per
    preservare diversification of belief.
    """
    from app.services.regime.meta_classifier import compute_meta_regime

    try:
        r = compute_meta_regime(
            db,
            lookback_months=lookback_months,
            min_weight=min_weight,
            prob_floor=prob_floor,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Meta-regime failed: {str(e)}")

    return {
        "date": r.date,
        "regime": r.regime,
        "final_probabilities": r.final_probabilities,
        "ground_truth_window_months": r.ground_truth_window_months,
        "contributions": [
            {
                "name": c.name,
                "current_probabilities": c.current_probabilities,
                "brier_score": c.brier_score,
                "weight": c.weight,
                "n_history_points": c.n_history_points,
                "error": c.error,
            }
            for c in r.contributions
        ],
        "notes": r.notes,
    }


@router.post("/admin/repair-lei")
def trigger_repair_lei(
    dry_run: bool = Query(default=False, description="Anteprima senza commit"),
    db: Session = Depends(get_db),
):
    """Riallinea i record con `lei_roc` corrotto (`abs > 5`) sostituendo col
    valore CFNAIMA3 reale + ricalcolando regime/probabilita'/confidence.

    Idempotente. Bug originale: scheduler pre-2026-04-27 applicava `roc_6m`
    su CFNAI z-score, generando valori 30%/-86% invece dei livelli ±3.
    """
    from app.services.admin.data_repair import repair_lei_history

    try:
        report = repair_lei_history(db, dry_run=dry_run, sample_size=10)
        return {
            "status": "ok",
            "total_records": report.total_records,
            "corrupted": report.corrupted,
            "repaired": report.repaired,
            "skipped": report.skipped,
            "dry_run": report.dry_run,
            "sample_changes": report.sample_changes,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Repair failed: {str(e)}")


@router.get("/assets", response_model=list[str])
def list_asset_classes():
    """Lista di tutte le asset class monitorate."""
    return ASSET_CLASSES


@router.get("/data-snapshot", response_model=DataSnapshotResponse)
def get_data_snapshot(db: Session = Depends(get_db)):
    """Snapshot completo: formule (pesi condizioni + tabelle asset×regime) + valori raw correnti."""
    from app.services.regime.classifier import REGIME_CONDITIONS
    from app.services.scoring.engine import ASSET_REGIME_DATA
    from app.services.dedollarization.scorer import ASSET_DEDOLLAR_SENSITIVITY

    regime_conditions = {
        regime: [
            RegimeConditionMeta(name=name, weight=cfg["weight"], description=cfg["description"])
            for name, cfg in conds.items()
        ]
        for regime, conds in REGIME_CONDITIONS.items()
    }

    asset_regime_data = {
        asset: {
            regime: AssetRegimeStats(**stats)
            for regime, stats in per_regime.items()
        }
        for asset, per_regime in ASSET_REGIME_DATA.items()
    }

    latest_regime = (
        db.query(RegimeClassification)
        .order_by(RegimeClassification.date.desc())
        .first()
    )
    current_indicators: dict[str, float] = {}
    current_dedollar_indicators: dict[str, float] = {}
    current_probabilities: dict[str, float] = {}
    current_fit_scores: dict[str, float] = {}
    snapshot_date = None
    regime_name: str | None = None
    if latest_regime:
        snapshot_date = latest_regime.date
        regime_name = latest_regime.regime
        current_probabilities = {
            "reflation": latest_regime.probability_reflation,
            "stagflation": latest_regime.probability_stagflation,
            "deflation": latest_regime.probability_deflation,
            "goldilocks": latest_regime.probability_goldilocks,
        }
        if latest_regime.conditions_met:
            try:
                payload = json.loads(latest_regime.conditions_met)
                raw_ind = payload.get("indicators") or {}
                if isinstance(raw_ind, dict):
                    current_indicators = {
                        k: float(v) for k, v in raw_ind.items()
                        if isinstance(v, (int, float))
                    }
                raw_ded = payload.get("dedollar_indicators") or {}
                if isinstance(raw_ded, dict):
                    current_dedollar_indicators = {
                        k: float(v) for k, v in raw_ded.items()
                        if isinstance(v, (int, float))
                    }
                fs = payload.get("fit_scores") or {}
                if isinstance(fs, dict):
                    current_fit_scores = {
                        k: float(v) for k, v in fs.items()
                        if isinstance(v, (int, float))
                    }
            except Exception:
                pass

    # Asset scores e dedollar combined dalla data più recente
    current_asset_scores: dict[str, float] = {}
    if snapshot_date:
        signals = (
            db.query(DailySignal)
            .filter(DailySignal.date == snapshot_date)
            .all()
        )
        current_asset_scores = {s.asset_class: s.final_score for s in signals}

    current_dedollar_combined: float | None = None
    latest_sec = (
        db.query(SecularTrend)
        .filter(SecularTrend.trend_name == "dedollarization")
        .order_by(SecularTrend.date.desc())
        .first()
    )
    if latest_sec:
        try:
            payload = json.loads(latest_sec.components) if latest_sec.components else {}
            current_dedollar_combined = payload.get("combined_score", latest_sec.score)
        except Exception:
            current_dedollar_combined = latest_sec.score

    return DataSnapshotResponse(
        date=snapshot_date,
        regime=regime_name,
        regime_conditions=regime_conditions,
        asset_regime_data=asset_regime_data,
        asset_dedollar_sensitivity=dict(ASSET_DEDOLLAR_SENSITIVITY),
        current_indicators=current_indicators,
        current_dedollar_indicators=current_dedollar_indicators,
        current_probabilities=current_probabilities,
        current_fit_scores=current_fit_scores,
        current_asset_scores=current_asset_scores,
        current_dedollar_combined=current_dedollar_combined,
    )


@router.get("/dedollarization", response_model=DedollarizationResponse)
def get_dedollarization(db: Session = Depends(get_db)):
    """Score corrente di dedollarizzazione e componenti."""
    latest = (
        db.query(SecularTrend)
        .filter(SecularTrend.trend_name == "dedollarization")
        .order_by(SecularTrend.date.desc())
        .first()
    )
    if not latest:
        raise HTTPException(status_code=404, detail="Nessun dato dedollarizzazione disponibile")

    payload = json.loads(latest.components) if latest.components else {}

    # Supporta sia vecchio formato (solo components dict) che nuovo (nested)
    if "components" in payload:
        components = payload["components"]
        structural = payload.get("structural", {})
        decade = payload.get("decade", {})
        twenty_year = payload.get("twenty_year", {}) or {}
        structural_score = payload.get("structural_score", 0.0)
        decade_score = payload.get("decade_score", 0.0)
        twenty_year_score = payload.get("twenty_year_score")
        acceleration = payload.get("acceleration", 0.0)
        combined_score = payload.get("combined_score", latest.score)
        by_player = payload.get("by_player", {}) or {}
        geopolitical_score = payload.get("geopolitical_score", 0.0)
    else:
        components = payload
        structural = {}
        decade = {}
        twenty_year = {}
        structural_score = 0.0
        decade_score = 0.0
        twenty_year_score = None
        acceleration = 0.0
        combined_score = latest.score
        by_player = {}
        geopolitical_score = 0.0

    player_history = payload.get("player_history", {}) or {}
    player_acceleration_data = payload.get("player_acceleration", {}) or {}
    explanation = payload.get("explanation")

    return DedollarizationResponse(
        date=latest.date,
        score=latest.score,
        structural_score=structural_score,
        decade_score=decade_score,
        twenty_year_score=twenty_year_score,
        acceleration=acceleration,
        combined_score=combined_score,
        geopolitical_score=geopolitical_score,
        components=components,
        structural=structural,
        decade=decade,
        twenty_year=twenty_year,
        by_player=by_player,
        player_history=player_history,
        player_acceleration=player_acceleration_data,
        explanation=explanation,
    )


@router.get("/dedollarization/history", response_model=list[DedollarHistoryItem])
def get_dedollarization_history(
    days: int = Query(default=365, ge=1, le=3650),
    db: Session = Depends(get_db),
):
    """Storico score dedollarizzazione (combined + per orizzonte)."""
    records = (
        db.query(SecularTrend)
        .filter(SecularTrend.trend_name == "dedollarization")
        .order_by(SecularTrend.date.desc())
        .limit(days)
        .all()
    )

    items: list[DedollarHistoryItem] = []
    for r in records:
        payload = json.loads(r.components) if r.components else {}
        if "components" in payload:
            cyclical = payload.get("score", r.score)
            structural = payload.get("structural_score", 0.0)
            decade = payload.get("decade_score", 0.0)
            twenty = payload.get("twenty_year_score")
            geopolitical = payload.get("geopolitical_score", 0.0)
            acceleration = payload.get("acceleration", 0.0)
            combined = payload.get("combined_score", r.score)
        else:
            cyclical = r.score
            structural = 0.0
            decade = 0.0
            twenty = None
            geopolitical = 0.0
            acceleration = 0.0
            combined = r.score

        items.append(DedollarHistoryItem(
            date=r.date,
            combined_score=combined,
            cyclical_score=cyclical,
            structural_score=structural,
            decade_score=decade,
            twenty_year_score=twenty,
            geopolitical_score=geopolitical,
            acceleration=acceleration,
        ))
    return items


@router.get("/macro-indicators/history", response_model=list[MacroIndicatorsHistoryItem])
def get_macro_indicators_history(
    days: int = Query(default=365, ge=1, le=3650),
    db: Session = Depends(get_db),
):
    """Storico indicatori macro (estratti dal payload conditions_met)."""
    records = (
        db.query(RegimeClassification)
        .order_by(RegimeClassification.date.desc())
        .limit(days)
        .all()
    )

    out: list[MacroIndicatorsHistoryItem] = []
    for r in reversed(records):
        if not r.conditions_met:
            continue
        try:
            payload = json.loads(r.conditions_met)
        except Exception:
            continue
        indicators = payload.get("indicators") or {}
        if not isinstance(indicators, dict):
            continue
        clean: dict[str, float] = {}
        for k, v in indicators.items():
            try:
                clean[k] = float(v)
            except (TypeError, ValueError):
                continue
        if clean:
            out.append(MacroIndicatorsHistoryItem(date=r.date, indicators=clean))
    return out


@router.get("/dedollarization/player-history", response_model=list[PlayerHistoryItem])
def get_dedollar_player_history(
    days: int = Query(default=365, ge=1, le=3650),
    db: Session = Depends(get_db),
):
    """Storico daily score per macro-player (by_player[id].score)."""
    records = (
        db.query(SecularTrend)
        .filter(SecularTrend.trend_name == "dedollarization")
        .order_by(SecularTrend.date.desc())
        .limit(days)
        .all()
    )

    out: list[PlayerHistoryItem] = []
    for r in reversed(records):
        if not r.components:
            continue
        try:
            payload = json.loads(r.components)
        except Exception:
            continue
        by_player = payload.get("by_player") or {}
        if not isinstance(by_player, dict):
            continue
        players: dict[str, float] = {}
        for pid, pdata in by_player.items():
            if isinstance(pdata, dict) and "score" in pdata:
                try:
                    players[pid] = float(pdata["score"])
                except (TypeError, ValueError):
                    continue
        if players:
            out.append(PlayerHistoryItem(date=r.date, players=players))
    return out


@router.get("/signals/history", response_model=list[SignalsHistoryItem])
def get_signals_history(
    days: int = Query(default=365, ge=1, le=3650),
    db: Session = Depends(get_db),
):
    """Storico score finali per asset class (formato pivot: {date, scores})."""
    from datetime import timedelta

    cutoff = date.today() - timedelta(days=days)
    records = (
        db.query(DailySignal)
        .filter(DailySignal.date >= cutoff)
        .order_by(DailySignal.date.asc())
        .all()
    )

    by_date: dict[date, dict[str, float]] = {}
    for s in records:
        by_date.setdefault(s.date, {})[s.asset_class] = s.final_score

    return [SignalsHistoryItem(date=d, scores=scores) for d, scores in by_date.items()]


@router.post("/dedollarization/explanation")
def generate_dedollarization_explanation(db: Session = Depends(get_db)):
    """Genera on-demand l'analisi AI (Gemini) dell'ultimo stato dedollarizzazione.

    Salva il risultato nel record corrente di SecularTrend così che le successive
    GET /dedollarization lo restituiscano subito senza ri-chiamare Gemini.
    """
    from app.services.dedollarization.explainer import generate_explanation

    latest = (
        db.query(SecularTrend)
        .filter(SecularTrend.trend_name == "dedollarization")
        .order_by(SecularTrend.date.desc())
        .first()
    )
    if not latest:
        raise HTTPException(status_code=404, detail="Nessun dato dedollarizzazione disponibile")

    payload = json.loads(latest.components) if latest.components else {}
    raw_indicators = payload.get("raw_dedollar_indicators") or {}

    # Ricostruisco il dict attesi da generate_explanation (firma: dedollar, raw_indicators)
    dedollar_for_prompt = {
        "by_player": payload.get("by_player", {}),
    }

    text = generate_explanation(dedollar_for_prompt, raw_indicators=raw_indicators)
    if not text:
        raise HTTPException(
            status_code=502,
            detail="Gemini non ha restituito alcun testo. Verifica GEMINI_API_KEY o riprova.",
        )

    # Persisto l'explanation nel payload
    payload["explanation"] = text
    latest.components = json.dumps(payload)
    db.commit()

    return {"explanation": text, "date": latest.date.isoformat()}


@router.get("/news", response_model=list[NewsItemResponse])
def get_news(
    days: int = Query(default=3, ge=1, le=30),
    db: Session = Depends(get_db),
):
    """Notizie macro recenti con sentiment scoring."""
    from datetime import timedelta

    cutoff = date.today() - timedelta(days=days)
    news_items = (
        db.query(NewsSignal)
        .filter(NewsSignal.date >= cutoff)
        .order_by(NewsSignal.date.desc(), NewsSignal.relevance_score.desc())
        .limit(50)
        .all()
    )

    return [
        NewsItemResponse(
            date=n.date,
            source=n.source,
            title=n.title,
            summary=n.content_summary or "",
            sentiment=n.sentiment_score,
            relevance=n.relevance_score,
            affected_assets=json.loads(n.affected_assets) if n.affected_assets else {},
        )
        for n in news_items
    ]
