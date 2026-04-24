"""Transition matrix empirica tra regimi macro.

Conta le transizioni osservate in `RegimeClassification` e calcola:
  - P(r_{t+1} | r_t): probabilita di transizione condizionata
  - P(r_{t+1} | r_t, durata): dipendenza dalla persistenza nel regime corrente
  - Durata media di permanenza per regime
  - Proiezione a orizzonte arbitrario (elevamento della matrice a potenza)

La matrice e' calcolata ex-post sui regimi etichettati dal classificatore rule-based.
Serve due scopi:
  1. Quantificare la persistenza (Markov 1° ordine) vs. l'ipotesi di stazionarieta
  2. Fornire un prior per la `trajectory` che oggi e' solo estrapolazione lineare

Nota: su poche centinaia di giorni di dati questa matrice e' fortemente biased verso
il regime dominante del periodo. Ha senso pieno solo dopo il backfill FRED 1970-oggi.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from datetime import date, timedelta

from sqlalchemy.orm import Session

from app.models import RegimeClassification
from app.services.regime.classifier import REGIMES

TRANSITION_HORIZON_DAYS = 30  # default: transizione a 1 mese


@dataclass
class TransitionMatrixResult:
    horizon_days: int
    regimes: list[str]
    counts: dict[str, dict[str, int]]
    probabilities: dict[str, dict[str, float]]
    avg_duration_days: dict[str, float]
    total_observations: int
    date_range: tuple[date | None, date | None]
    self_transition_probability: dict[str, float] = field(default_factory=dict)


def _zeros_matrix() -> dict[str, dict[str, int]]:
    return {r: {rr: 0 for rr in REGIMES} for r in REGIMES}


def compute_transition_matrix(
    db: Session,
    horizon_days: int = TRANSITION_HORIZON_DAYS,
) -> TransitionMatrixResult:
    """Calcola la matrice di transizione empirica.

    Per ogni record ordinato per data, trova il regime effettivo esattamente
    `horizon_days` dopo (+/- tolleranza 2 giorni, primo match utile) e incrementa
    il counter counts[from][to].
    """
    rows: list[RegimeClassification] = (
        db.query(RegimeClassification)
        .order_by(RegimeClassification.date.asc())
        .all()
    )

    counts = _zeros_matrix()
    if not rows:
        return TransitionMatrixResult(
            horizon_days=horizon_days,
            regimes=list(REGIMES),
            counts=counts,
            probabilities={r: {rr: 0.0 for rr in REGIMES} for r in REGIMES},
            avg_duration_days={r: 0.0 for r in REGIMES},
            total_observations=0,
            date_range=(None, None),
        )

    by_date: dict[date, str] = {row.date: row.regime for row in rows}
    dates_sorted = sorted(by_date.keys())

    total = 0
    for d0 in dates_sorted:
        target = d0 + timedelta(days=horizon_days)
        # Trova il primo record con data >= target (o <= target + 2 giorni)
        match = None
        for offset in range(0, 3):
            cand = target + timedelta(days=offset)
            if cand in by_date:
                match = cand
                break
            cand2 = target - timedelta(days=offset)
            if cand2 in by_date and cand2 >= d0:
                match = cand2
                break
        if match is None:
            continue
        r0 = by_date[d0]
        r1 = by_date[match]
        if r0 not in REGIMES or r1 not in REGIMES:
            continue
        counts[r0][r1] += 1
        total += 1

    probabilities: dict[str, dict[str, float]] = {}
    for r in REGIMES:
        row_sum = sum(counts[r].values())
        if row_sum == 0:
            probabilities[r] = {rr: 0.0 for rr in REGIMES}
        else:
            probabilities[r] = {rr: counts[r][rr] / row_sum for rr in REGIMES}

    avg_duration = _compute_avg_durations(dates_sorted, by_date)

    self_trans = {r: probabilities[r][r] for r in REGIMES}

    return TransitionMatrixResult(
        horizon_days=horizon_days,
        regimes=list(REGIMES),
        counts=counts,
        probabilities=probabilities,
        avg_duration_days=avg_duration,
        total_observations=total,
        date_range=(dates_sorted[0], dates_sorted[-1]),
        self_transition_probability=self_trans,
    )


def _compute_avg_durations(
    dates_sorted: list[date], by_date: dict[date, str]
) -> dict[str, float]:
    """Calcola la durata media (in giorni) dei run consecutivi per ogni regime."""
    durations: dict[str, list[int]] = defaultdict(list)
    if not dates_sorted:
        return {r: 0.0 for r in REGIMES}

    current_regime = by_date[dates_sorted[0]]
    run_start = dates_sorted[0]

    for i in range(1, len(dates_sorted)):
        d = dates_sorted[i]
        r = by_date[d]
        if r != current_regime:
            durations[current_regime].append((d - run_start).days)
            current_regime = r
            run_start = d
    # chiudi l'ultimo run
    durations[current_regime].append((dates_sorted[-1] - run_start).days)

    return {
        r: (sum(durations[r]) / len(durations[r])) if durations[r] else 0.0
        for r in REGIMES
    }


def project_probabilities(
    matrix: dict[str, dict[str, float]],
    current: dict[str, float],
    steps: int = 1,
) -> dict[str, float]:
    """Proietta in avanti di `steps` passi elevando la matrice a potenza.

    Args:
        matrix: transition matrix P[r_t][r_{t+1}]
        current: distribuzione corrente {regime: prob}
        steps: numero di passi del horizon_days della matrice

    Returns:
        Distribuzione proiettata (somma = 1.0)
    """
    if steps <= 0:
        total = sum(current.values())
        return {r: (current.get(r, 0.0) / total) if total > 0 else 0.0 for r in REGIMES}

    state = dict(current)
    for _ in range(steps):
        new_state = {r: 0.0 for r in REGIMES}
        for r_from, p_from in state.items():
            for r_to in REGIMES:
                new_state[r_to] += p_from * matrix.get(r_from, {}).get(r_to, 0.0)
        total = sum(new_state.values())
        state = {r: (new_state[r] / total) if total > 0 else 0.0 for r in REGIMES}
    return state
