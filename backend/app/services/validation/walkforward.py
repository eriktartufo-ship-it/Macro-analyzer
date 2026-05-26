"""T9-AUDIT — Walk-forward retraining annuale.

Council finding (buffett): "Walk-forward retraining annuale: simulare cosa
avrebbe predetto il sistema con solo dati fino a t-1. Oggi è ancora pseudo-OOS."

Vero OOS = simulare un PM che AL TEMPO T usa solo dati fino a T per predire T+1.
Niente leakage temporale (es. parametri calibrati su 2024 non possono essere
usati per predire 2008).

Pipeline:
1. Per ogni cutoff_year da 1995 a 2024:
   a. Training set = episodi con end_year <= cutoff
   b. Test = episodi che iniziano nell'anno successivo
2. Train ML su training set (no shared params con test).
3. Predict test, registra accuracy.
4. Aggregate metrics over all cutoffs.

Output: per-year accuracy + global OOS accuracy + Brier + stability metric.
"""
from __future__ import annotations

from dataclasses import dataclass

from app.services.regime.historical_episodes import HISTORICAL_EPISODES, HistoricalEpisode
from app.services.regime.ml_classifier import predict_regime, train_classifier


@dataclass
class WalkForwardResult:
    """Singolo cutoff: train su pre-cutoff, predict post-cutoff."""
    cutoff_year: int
    n_train: int
    n_test: int
    correct: int
    total: int
    accuracy: float
    per_episode: list[dict]  # name, true, predicted, correct


@dataclass
class WalkForwardAggregate:
    """Risultato aggregato across cutoffs."""
    cutoffs_evaluated: int
    total_predictions: int
    total_correct: int
    global_accuracy: float
    accuracy_by_year: dict[int, float]
    accuracy_std: float  # stabilità del modello
    min_accuracy: float
    max_accuracy: float


def walk_forward_evaluation(
    cutoff_start: int = 1995,
    cutoff_end: int = 2024,
    min_train_size: int = 8,
    episodes: list[HistoricalEpisode] | None = None,
) -> WalkForwardAggregate:
    """Esegue walk-forward su tutti i cutoff possibili.

    Args:
        cutoff_start: primo cutoff_year (default 1995 per avere training >=8 ep).
        cutoff_end: ultimo cutoff_year.
        min_train_size: minimum episodi training prima di valutare cutoff.
        episodes: opzionale, override HISTORICAL_EPISODES (per test).

    Returns:
        WalkForwardAggregate con metrics aggregate.
    """
    if episodes is None:
        episodes = list(HISTORICAL_EPISODES)

    results_by_cutoff: list[WalkForwardResult] = []

    for cutoff in range(cutoff_start, cutoff_end + 1):
        train_eps = [ep for ep in episodes if ep.end_year <= cutoff]
        test_eps = [
            ep for ep in episodes
            if ep.start_year == cutoff + 1  # solo episodi del anno IMMEDIATAMENTE successivo
        ]
        if len(train_eps) < min_train_size or len(test_eps) == 0:
            continue

        # Train ML su pre-cutoff
        try:
            bundle = train_classifier(episodes=train_eps)
        except Exception:
            continue

        per_episode = []
        correct_count = 0
        for ep in test_eps:
            probs = predict_regime(ep.indicators, bundle)
            predicted = max(probs, key=probs.get)
            is_correct = predicted == ep.true_regime
            if is_correct:
                correct_count += 1
            per_episode.append({
                "name": ep.name,
                "true": ep.true_regime,
                "predicted": predicted,
                "correct": is_correct,
                "max_prob": round(max(probs.values()), 3),
            })

        accuracy = correct_count / len(test_eps) if test_eps else 0.0
        results_by_cutoff.append(WalkForwardResult(
            cutoff_year=cutoff,
            n_train=len(train_eps),
            n_test=len(test_eps),
            correct=correct_count,
            total=len(test_eps),
            accuracy=accuracy,
            per_episode=per_episode,
        ))

    # Aggregate
    if not results_by_cutoff:
        return WalkForwardAggregate(
            cutoffs_evaluated=0, total_predictions=0, total_correct=0,
            global_accuracy=0.0, accuracy_by_year={},
            accuracy_std=0.0, min_accuracy=0.0, max_accuracy=0.0,
        )

    total_correct = sum(r.correct for r in results_by_cutoff)
    total_pred = sum(r.total for r in results_by_cutoff)
    global_acc = total_correct / total_pred if total_pred > 0 else 0.0

    accuracy_by_year = {r.cutoff_year + 1: r.accuracy for r in results_by_cutoff}
    accs = list(accuracy_by_year.values())
    import statistics
    acc_std = statistics.stdev(accs) if len(accs) > 1 else 0.0

    return WalkForwardAggregate(
        cutoffs_evaluated=len(results_by_cutoff),
        total_predictions=total_pred,
        total_correct=total_correct,
        global_accuracy=global_acc,
        accuracy_by_year=accuracy_by_year,
        accuracy_std=acc_std,
        min_accuracy=min(accs),
        max_accuracy=max(accs),
    )


def walk_forward_per_year_report(
    cutoff_start: int = 1995,
    cutoff_end: int = 2024,
) -> list[WalkForwardResult]:
    """Restituisce risultato dettagliato per ogni cutoff (per UI/report)."""
    episodes = list(HISTORICAL_EPISODES)
    out: list[WalkForwardResult] = []
    for cutoff in range(cutoff_start, cutoff_end + 1):
        train_eps = [ep for ep in episodes if ep.end_year <= cutoff]
        test_eps = [ep for ep in episodes if ep.start_year == cutoff + 1]
        if len(train_eps) < 8 or len(test_eps) == 0:
            continue
        try:
            bundle = train_classifier(episodes=train_eps)
        except Exception:
            continue
        per_episode = []
        correct = 0
        for ep in test_eps:
            probs = predict_regime(ep.indicators, bundle)
            predicted = max(probs, key=probs.get)
            ok = predicted == ep.true_regime
            if ok:
                correct += 1
            per_episode.append({
                "name": ep.name, "true": ep.true_regime,
                "predicted": predicted, "correct": ok,
                "max_prob": round(max(probs.values()), 3),
            })
        out.append(WalkForwardResult(
            cutoff_year=cutoff, n_train=len(train_eps),
            n_test=len(test_eps), correct=correct, total=len(test_eps),
            accuracy=correct / len(test_eps),
            per_episode=per_episode,
        ))
    return out
