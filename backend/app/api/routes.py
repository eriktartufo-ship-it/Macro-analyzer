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
def trigger_regime_backfill(days: int = Query(default=180, ge=1, le=3650)):
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


@router.get("/assets", response_model=list[str])
def list_asset_classes():
    """Lista di tutte le asset class monitorate."""
    return ASSET_CLASSES


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
