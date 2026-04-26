"""Lead time analysis: di quanti mesi prima il classifier ha segnalato la recessione?

Per ogni recessione NBER (USREC = 1) confronta:
  - data inizio recessione (NBER official)
  - data primo mese in cui prob(deflation) >= signal_threshold nella nostra serie
    O prob(stagflation) >= soglia (perche' alcune recessioni sono stagflation, non deflation)

Lead time = months(recession_start - signal_date). Valori positivi = sistema anticipa,
negativi = ritarda. Banche puntano a 3-6 mesi di lead time medio.

Output: tabella per recessione + stat aggregate (mean, median, hit_rate).
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
from loguru import logger
from sqlalchemy.orm import Session

from app.models import RegimeClassification
from app.services.indicators.fetcher import FredFetcher


SIGNAL_THRESHOLD = 0.35  # P(stress regime) sopra cui consideriamo "segnale acceso"


@dataclass
class RecessionLead:
    recession_start: str
    recession_end: str
    duration_months: int
    signal_date: str | None         # primo mese pre-recession con prob>=threshold
    lead_months: float | None       # months(start - signal). Positivo = anticipo.
    max_prob_during: float          # max prob(stress regime) durante recession
    pre_recession_max_prob: float   # max prob nei 12 mesi prima


@dataclass
class LeadTimeReport:
    threshold: float
    lookback_months: int
    recessions: list[RecessionLead]
    avg_lead_months: float | None
    median_lead_months: float | None
    hit_rate: float                 # frazione recessioni con segnale anticipato (lead > 0)
    n_recessions_analyzed: int


def _list_nber_recessions(usrec: pd.Series) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    """Estrae intervalli di recessione (start, end) dalla serie binaria NBER."""
    s = usrec.copy().sort_index()
    s.index = pd.to_datetime(s.index)
    spans: list[tuple[pd.Timestamp, pd.Timestamp]] = []
    in_rec = False
    start = None
    for d, v in s.items():
        if v == 1 and not in_rec:
            in_rec = True
            start = d
        elif v == 0 and in_rec:
            in_rec = False
            spans.append((start, d - pd.Timedelta(days=1)))
    if in_rec and start is not None:
        spans.append((start, s.index[-1]))
    return spans


def compute_lead_time_report(
    db: Session,
    threshold: float = SIGNAL_THRESHOLD,
    lookback_months: int = 12,
    min_recession_year: int = 1970,
) -> LeadTimeReport:
    """Calcola lead time per ogni recessione NBER dal `min_recession_year`."""
    fred = FredFetcher()
    usrec = fred.fetch_series("nber_recession", start_date=pd.Timestamp(f"{min_recession_year}-01-01").date())
    spans = _list_nber_recessions(usrec)

    rows = (
        db.query(RegimeClassification)
        .order_by(RegimeClassification.date.asc())
        .all()
    )
    if not rows:
        return LeadTimeReport(
            threshold=threshold, lookback_months=lookback_months,
            recessions=[], avg_lead_months=None, median_lead_months=None,
            hit_rate=0.0, n_recessions_analyzed=0,
        )

    df = pd.DataFrame([
        {
            "date": pd.Timestamp(r.date),
            "deflation": r.probability_deflation,
            "stagflation": r.probability_stagflation,
        }
        for r in rows
    ]).set_index("date").sort_index().resample("ME").mean().dropna()

    out: list[RecessionLead] = []
    for start, end in spans:
        if start.year < min_recession_year:
            continue
        # Allinea start a fine-mese
        start_m = pd.Timestamp(start).to_period("M").to_timestamp("M")
        end_m = pd.Timestamp(end).to_period("M").to_timestamp("M")

        # Pre-recession window: lookback_months precedenti
        pre_start = start_m - pd.DateOffset(months=lookback_months)
        pre_window = df.loc[(df.index >= pre_start) & (df.index < start_m)]
        # Stress = max(deflation, stagflation)
        if pre_window.empty:
            pre_max = 0.0
            signal_date = None
            lead = None
        else:
            stress = pre_window[["deflation", "stagflation"]].max(axis=1)
            pre_max = float(stress.max())
            triggered = stress[stress >= threshold]
            if triggered.empty:
                signal_date = None
                lead = None
            else:
                first_trigger = triggered.index.min()
                signal_date = str(first_trigger.date())
                lead = float((start_m - first_trigger).days / 30.4375)

        # Max during recession
        during = df.loc[(df.index >= start_m) & (df.index <= end_m)]
        if during.empty:
            max_during = 0.0
        else:
            max_during = float(during[["deflation", "stagflation"]].max(axis=1).max())

        out.append(RecessionLead(
            recession_start=str(start.date()),
            recession_end=str(end.date()),
            duration_months=int(round((end_m - start_m).days / 30.4375)) + 1,
            signal_date=signal_date,
            lead_months=lead,
            max_prob_during=max_during,
            pre_recession_max_prob=pre_max,
        ))

    leads = [r.lead_months for r in out if r.lead_months is not None]
    avg = float(sum(leads) / len(leads)) if leads else None
    if leads:
        s = pd.Series(leads)
        median = float(s.median())
    else:
        median = None
    hit_rate = float(len(leads) / len(out)) if out else 0.0

    logger.info(
        f"Lead-time report: {len(out)} recessioni, hit_rate {hit_rate:.0%}, "
        f"avg lead {avg if avg else 'n/a'} months"
    )

    return LeadTimeReport(
        threshold=threshold,
        lookback_months=lookback_months,
        recessions=out,
        avg_lead_months=avg,
        median_lead_months=median,
        hit_rate=hit_rate,
        n_recessions_analyzed=len(out),
    )
