"""T14 Event-triggered rebalance — Council action #4 (sessione 18).

Razionale: il rebalance calendar mensile può perdere window di opportunità
quando il mercato si muove rapidamente (regime shift, panic, policy surprise).
Aggiunge segnali di rebalance OFF-CALENDAR quando trigger fires.

Trigger types (council T14 2026-05-31):
1. **vix_panic**: VIX > 30 (was < 25 in lookback window) → panic regime shift
2. **m2_surge**: M2 YoY ROC swing > +5pp in 3 mesi → liquidity surge (2020-style)
3. **curve_uninvert**: 10Y-2Y rises from < 0 to > 0 → recession signal release
4. **fomc_dovish_surprise**: FOMC sentiment shift > +0.3 (dovish swing)

Output: dict {trigger_name: bool} per ogni window check.

Usage:
    detector = EventTriggerDetector()
    fired = detector.check_window(current_indicators, previous_indicators)
    if any(fired.values()):
        # Forza rebalance extra
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class TriggerSnapshot:
    """Snapshot indicators per un punto temporale."""
    vix: float
    m2_yoy: float
    yield_curve_10y2y: float
    fomc_sentiment: float = 0.0  # -1 dovish, +1 hawkish, 0 neutral


@dataclass
class TriggerResult:
    """Risultato check trigger + dettagli."""
    fired: bool
    name: str
    description: str
    severity: float = 0.5  # 0-1 (per priorità rebalance)


class EventTriggerDetector:
    """Detector trigger event-driven per rebalance off-calendar.

    Pattern: keep snapshot precedente, confronta con corrente.
    Window suggested: 30 giorni (monthly check) o 7 giorni (weekly).

    Council T14 thresholds (default conservativi):
    - VIX_PANIC_THRESHOLD: 30 (panic level storico 2008/2020/2022)
    - VIX_CALM_THRESHOLD: 25 (was-calm requirement per anti-noise)
    - M2_SURGE_PP: 5.0 (3m swing in ROC)
    - FOMC_DOVISH_SHIFT: 0.30 (sentiment swing significativo)
    """
    VIX_PANIC_THRESHOLD: float = 30.0
    VIX_CALM_THRESHOLD: float = 25.0
    M2_SURGE_PP: float = 5.0
    FOMC_DOVISH_SHIFT: float = 0.30

    def check_window(
        self,
        current: TriggerSnapshot,
        previous: TriggerSnapshot | None,
    ) -> dict[str, TriggerResult]:
        """Valuta tutti i trigger comparing current vs previous snapshot.

        Args:
            current: indicators latest.
            previous: indicators alla check precedente (None = primo check).

        Returns:
            dict {trigger_name: TriggerResult} con `fired=True` se trigger attivo.
        """
        results: dict[str, TriggerResult] = {}

        # 1. VIX panic: spike from calm a panic
        results["vix_panic"] = self._check_vix_panic(current, previous)
        # 2. M2 surge: liquidity acceleration
        results["m2_surge"] = self._check_m2_surge(current, previous)
        # 3. Curve un-invert: from negative to positive
        results["curve_uninvert"] = self._check_curve_uninvert(current, previous)
        # 4. FOMC dovish surprise
        results["fomc_dovish_surprise"] = self._check_fomc_dovish(current, previous)

        return results

    def _check_vix_panic(
        self,
        current: TriggerSnapshot,
        previous: TriggerSnapshot | None,
    ) -> TriggerResult:
        """VIX > 30 con previous < 25 → panic regime shift."""
        if previous is None:
            # No prior context: fire solo se VIX corrente > 35 (panic estremo)
            fired = current.vix > 35.0
            sev = min(1.0, (current.vix - 25.0) / 20.0) if fired else 0.0
        else:
            fired = (
                current.vix > self.VIX_PANIC_THRESHOLD
                and previous.vix < self.VIX_CALM_THRESHOLD
            )
            # Severity proporzionale al jump
            sev = min(1.0, (current.vix - previous.vix) / 20.0) if fired else 0.0
        return TriggerResult(
            fired=fired,
            name="vix_panic",
            description=f"VIX panic spike (current={current.vix:.1f}, prev={(previous.vix if previous else None)})",
            severity=max(0.0, sev),
        )

    def _check_m2_surge(
        self,
        current: TriggerSnapshot,
        previous: TriggerSnapshot | None,
    ) -> TriggerResult:
        """M2 ROC swing > +5pp = liquidity surge (Fed pump)."""
        if previous is None:
            # Primo check: fire solo se M2 > 15% YoY (estremo 2020-style)
            fired = current.m2_yoy > 15.0
            sev = min(1.0, (current.m2_yoy - 10.0) / 10.0) if fired else 0.0
        else:
            swing = current.m2_yoy - previous.m2_yoy
            fired = swing > self.M2_SURGE_PP
            sev = min(1.0, swing / 10.0) if fired else 0.0
        return TriggerResult(
            fired=fired,
            name="m2_surge",
            description=f"M2 surge (current={current.m2_yoy:.1f}%, swing detected)",
            severity=max(0.0, sev),
        )

    def _check_curve_uninvert(
        self,
        current: TriggerSnapshot,
        previous: TriggerSnapshot | None,
    ) -> TriggerResult:
        """10Y-2Y rises from negative to positive = recession signal release.

        Storia: curve un-invert PRECEDE recession start ~6-12 mesi (1990, 2001,
        2008, 2020 inversion → recession). Trigger = entry signal per defensive
        regime.
        """
        if previous is None:
            fired = False
            sev = 0.0
        else:
            fired = previous.yield_curve_10y2y < 0 and current.yield_curve_10y2y > 0
            # Severity: prossima a 0 = early signal, magnitude = late
            sev = min(1.0, current.yield_curve_10y2y / 0.5) if fired else 0.0
        return TriggerResult(
            fired=fired,
            name="curve_uninvert",
            description=(
                f"Curve uninvert (current={current.yield_curve_10y2y:.2f}, "
                f"prev={(previous.yield_curve_10y2y if previous else None)})"
            ),
            severity=max(0.0, sev),
        )

    def _check_fomc_dovish(
        self,
        current: TriggerSnapshot,
        previous: TriggerSnapshot | None,
    ) -> TriggerResult:
        """FOMC sentiment shift > +0.30 dovish = policy pivot."""
        if previous is None:
            # Primo check: fire se FOMC sentiment < -0.5 (very dovish)
            fired = current.fomc_sentiment < -0.5
            sev = min(1.0, abs(current.fomc_sentiment)) if fired else 0.0
        else:
            shift = previous.fomc_sentiment - current.fomc_sentiment
            # shift > 0 = current more dovish than previous (sentiment scale: + hawkish, - dovish)
            fired = shift > self.FOMC_DOVISH_SHIFT
            sev = min(1.0, shift / 1.0) if fired else 0.0
        return TriggerResult(
            fired=fired,
            name="fomc_dovish_surprise",
            description=(
                f"FOMC dovish (current={current.fomc_sentiment:+.2f}, "
                f"prev={(previous.fomc_sentiment if previous else None)})"
            ),
            severity=max(0.0, sev),
        )


def any_trigger_fired(results: dict[str, TriggerResult]) -> bool:
    """Helper: True se almeno un trigger fired nei results."""
    return any(r.fired for r in results.values())


def max_severity(results: dict[str, TriggerResult]) -> float:
    """Helper: severity massima fra i trigger fired (0.0 se nessuno)."""
    fired_severities = [r.severity for r in results.values() if r.fired]
    return max(fired_severities) if fired_severities else 0.0


def fired_triggers_summary(results: dict[str, TriggerResult]) -> list[str]:
    """Helper: lista descrizioni triggers fired."""
    return [r.description for r in results.values() if r.fired]
