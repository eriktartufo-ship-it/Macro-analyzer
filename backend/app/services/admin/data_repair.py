"""Operazioni di riparazione dati su `regime_classifications`.

Idempotenti, safe da rieseguire. Esposte come endpoint admin (`/admin/...`)
e come CLI script (`backend/scripts/repair_lei_history.py`).
"""

from __future__ import annotations

import json
from dataclasses import dataclass

import pandas as pd
from loguru import logger
from sqlalchemy.orm import Session

from app.models import RegimeClassification
from app.services.indicators.fetcher import FredFetcher
from app.services.regime.classifier import classify_regime


# Range storico CFNAI z-score: ±3 tipico, ±5 estremo (es. recessioni).
# Soglia 5 e' conservativa: oltre 5 in valore assoluto e' quasi sicuramente
# bug del calcolo `roc_6m` applicato sopra uno z-score (genera valori >>5).
LEI_CORRUPTION_THRESHOLD = 5.0


@dataclass
class RepairLEIReport:
    total_records: int
    corrupted: int
    repaired: int
    skipped: int
    dry_run: bool
    sample_changes: list[dict]   # primi N esempi pre/post per audit


def _load_cfnai_history(fred: FredFetcher) -> pd.Series:
    raw = fred.fetch_series("lei").copy()
    raw.index = pd.to_datetime(raw.index)
    return raw.sort_index()


def _value_as_of(series: pd.Series, target: pd.Timestamp) -> float | None:
    sub = series.loc[:target]
    if sub.empty:
        return None
    return float(sub.iloc[-1])


def repair_lei_history(
    db: Session,
    *,
    dry_run: bool = False,
    sample_size: int = 10,
) -> RepairLEIReport:
    """Riallinea i record di `regime_classifications` con `lei_roc` corrotto.

    Bug pre-2026-04-27: lo scheduler giornaliero applicava `roc_6m()` sopra
    CFNAIMA3 (gia' z-score), salvando `lei_roc` come 30%, -86% etc invece
    del livello z-score atteso ±3.

    Per ogni record con `abs(lei_roc) > 5`:
    1. Sostituisce `indicators.lei_roc` col valore CFNAIMA3 reale alla data.
    2. Re-runno `classify_regime()` con indicators corretti.
    3. UPDATE regime/probability_*/confidence/conditions_met.

    Idempotente: record gia' fixati sono saltati.
    """
    fred = FredFetcher()
    cfnai = _load_cfnai_history(fred)
    logger.info(
        f"repair_lei_history: CFNAIMA3 series len={len(cfnai)}, "
        f"range {cfnai.index[0].date()} -> {cfnai.index[-1].date()}"
    )

    all_records = db.query(RegimeClassification).all()

    to_fix: list[tuple[RegimeClassification, dict, dict]] = []
    for r in all_records:
        if not r.conditions_met:
            continue
        try:
            meta = json.loads(r.conditions_met)
        except Exception:
            continue
        ind = meta.get("indicators")
        if not isinstance(ind, dict):
            continue
        lei = ind.get("lei_roc")
        if lei is None:
            continue
        if abs(float(lei)) > LEI_CORRUPTION_THRESHOLD:
            to_fix.append((r, meta, ind))

    repaired = 0
    skipped = 0
    samples: list[dict] = []

    for r, meta, ind in to_fix:
        target = pd.Timestamp(r.date)
        cfnai_val = _value_as_of(cfnai, target)
        if cfnai_val is None:
            skipped += 1
            continue

        old_lei = ind["lei_roc"]
        old_regime = r.regime
        old_conf = r.confidence

        ind["lei_roc"] = cfnai_val
        new = classify_regime(ind)

        if len(samples) < sample_size:
            samples.append({
                "date": r.date.isoformat(),
                "lei_roc_before": old_lei,
                "lei_roc_after": cfnai_val,
                "regime_before": old_regime,
                "regime_after": new["regime"],
                "confidence_before": old_conf,
                "confidence_after": new["confidence"],
            })

        if not dry_run:
            r.regime = new["regime"]
            probs = new["probabilities"]
            r.probability_reflation = float(probs.get("reflation", 0.0))
            r.probability_stagflation = float(probs.get("stagflation", 0.0))
            r.probability_deflation = float(probs.get("deflation", 0.0))
            r.probability_goldilocks = float(probs.get("goldilocks", 0.0))
            r.confidence = float(new["confidence"])

            meta["indicators"] = ind
            if "conditions_detail" in meta and "conditions_detail" in new:
                meta["conditions_detail"] = new["conditions_detail"]
            if "fit_scores" in meta and "fit_scores" in new:
                meta["fit_scores"] = new["fit_scores"]
            r.conditions_met = json.dumps(meta)

        repaired += 1

    if not dry_run and repaired > 0:
        db.commit()
        logger.info(f"repair_lei_history: committed {repaired} updates")
    elif dry_run:
        logger.info(f"repair_lei_history: dry-run, would update {repaired} records")

    return RepairLEIReport(
        total_records=len(all_records),
        corrupted=len(to_fix),
        repaired=repaired,
        skipped=skipped,
        dry_run=dry_run,
        sample_changes=samples,
    )
