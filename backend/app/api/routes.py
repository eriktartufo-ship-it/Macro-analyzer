"""API endpoints per il Macro Analyzer."""

import json
from datetime import date

from fastapi import APIRouter, Depends, HTTPException, Query
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


@router.get("/regime/history", response_model=list[RegimeResponse])
def get_regime_history(
    days: int = Query(default=30, ge=1, le=365),
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
def get_scoreboard(db: Session = Depends(get_db)):
    """Dashboard completa: regime + tutti gli scores ordinati."""
    regime = (
        db.query(RegimeClassification)
        .order_by(RegimeClassification.date.desc())
        .first()
    )
    if not regime:
        raise HTTPException(status_code=404, detail="Nessun dato disponibile")

    signals = (
        db.query(DailySignal)
        .filter(DailySignal.date == regime.date)
        .order_by(DailySignal.final_score.desc())
        .all()
    )

    scores = {s.asset_class: s.final_score for s in signals}

    return ScoreboardResponse(
        date=regime.date,
        regime=regime.regime,
        confidence=regime.confidence,
        scores=scores,
    )


@router.post("/regime/classify")
def classify_from_indicators(indicators: dict[str, float]):
    """Classifica regime on-demand da indicatori forniti manualmente."""
    result = classify_regime(indicators)
    scores = calculate_final_scores(result["probabilities"])

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
