"""Test meta-regime Debasement (R2 DREAM).

Il Debasement e' un META-regime monetario (orizzonte lento) SOPRA i 4 regimi ciclici
economici (Investment Clock). Non lo forziamo in una 5a "probabilita'" comparabile
testa-a-testa (sarebbe una scala inventata): e' un'INTENSITA' 0-1 derivata dal segnale
divergenza oro/tassi reali (R1), con:
  - sotto-fasi Early / Mature / Late (dalla traiettoria della divergenza)
  - isteresi ASIMMETRICA: facile accendersi, difficile spegnersi (Vito: "il sistema e'
    molto rigido per evitare falsi rumori, preferisce restare in debasement").
Vista informativa, non trading (spec R2).
"""
from __future__ import annotations

from pytest import approx

from app.services.backtest.debasement_regime import (
    CONFIRM_OFF,
    CONFIRM_ON,
    OFF_THRESHOLD,
    ON_THRESHOLD,
    apply_hysteresis,
    classify_subphase,
    compute_debasement_regime,
    gap_to_intensity,
    months_in_current_state,
)


def test_gap_to_intensity_monotonic_and_bounded():
    lo = gap_to_intensity(-0.25)
    mid = gap_to_intensity(0.10)
    hi = gap_to_intensity(0.40)
    assert 0.0 <= lo < mid < hi <= 1.0
    assert lo < 0.2 and hi > 0.8          # -25% debole, +40% forte
    assert gap_to_intensity(0.10) == approx(0.5, abs=1e-9)  # center


def test_hysteresis_turns_on_after_confirm_on():
    # sale sopra ON e ci resta: si accende dopo CONFIRM_ON mesi consecutivi
    series = [0.1] * 3 + [0.9] * 5
    states = apply_hysteresis(series)
    assert states[:3] == [False, False, False]
    # primo True esattamente al CONFIRM_ON-esimo mese sopra soglia
    first_on = states.index(True)
    assert first_on == 3 + (CONFIRM_ON - 1)
    assert states[-1] is True


def test_hysteresis_is_asymmetric_hard_to_turn_off():
    # acceso, poi scende sotto OFF: NON si spegne subito, servono CONFIRM_OFF mesi.
    # E un singolo scivolone sotto soglia non lo spegne (rigidita').
    series = [0.9] * 6 + [0.3] * (CONFIRM_OFF - 1) + [0.9]
    states = apply_hysteresis(series)
    assert all(states[6:]) is True or states[-1] is True   # torna su prima di spegnersi
    assert states[-1] is True
    assert CONFIRM_OFF > CONFIRM_ON                          # asimmetria strutturale


def test_hysteresis_eventually_turns_off_after_sustained_drop():
    series = [0.9] * 6 + [0.2] * (CONFIRM_OFF + 2)
    states = apply_hysteresis(series)
    assert states[-1] is False
    # si spegne solo dopo CONFIRM_OFF mesi sotto soglia
    assert states[6 + CONFIRM_OFF - 1] is False
    assert states[6 + CONFIRM_OFF - 2] is True


def test_months_in_current_state():
    assert months_in_current_state([False, True, True, True]) == 3
    assert months_in_current_state([True, True, False]) == 1
    assert months_in_current_state([]) == 0


def test_subphase_off_when_not_active():
    assert classify_subphase(on=False, months_in=0, intensity_now=0.1,
                             peak_since_on=0.1, slope=0.0) is None


def test_subphase_early_when_recently_on():
    p = classify_subphase(on=True, months_in=3, intensity_now=0.7,
                          peak_since_on=0.7, slope=0.05)
    assert p == "Early"


def test_subphase_mature_when_high_and_sustained():
    # acceso da tempo, alto, corretto poco dal picco -> Mature (caso attuale Vito)
    p = classify_subphase(on=True, months_in=30, intensity_now=0.90,
                          peak_since_on=0.93, slope=-0.01)
    assert p == "Mature"


def test_subphase_late_when_rolling_over_toward_neutral():
    # alto ma sceso ben sotto il picco e in discesa persistente -> Late
    p = classify_subphase(on=True, months_in=40, intensity_now=0.62,
                          peak_since_on=0.95, slope=-0.04)
    assert p == "Late"


def test_compute_orchestrator_snapshot_shape_and_labels():
    # storia divergenza: sale e resta alta -> debasement acceso, Mature
    gaps = [-0.1] * 4 + [0.35] * 36
    history = [{"date": f"20{20 + i // 12:02d}-{i % 12 + 1:02d}-28", "gap": g}
               for i, g in enumerate(gaps)]
    divergence = {
        "asof": history[-1]["date"], "gap": gaps[-1], "gap_pct": gaps[-1] * 100,
        "gold": 368.0, "fair_value": 265.0, "real_yield": 2.35,
        "window": 60, "n": 100, "history": history,
    }
    probs = {"reflation": 0.4, "stagflation": 0.3, "deflation": 0.1, "goldilocks": 0.2}
    out = compute_debasement_regime(divergence, probs)
    assert out is not None
    # 5-way: 4 prob cicliche + debasement intensita'
    assert set(out["scores"]) == {"reflation", "stagflation", "deflation",
                                  "goldilocks", "debasement"}
    assert out["scores"]["reflation"] == approx(0.4)
    assert 0.0 <= out["scores"]["debasement"] <= 1.0
    assert out["meta"]["active"] == "debasement"          # intensita' alta e confermata
    assert out["meta"]["sub_phase"] in {"Early", "Mature", "Late"}
    assert out["meta"]["months_in"] >= 1
    assert out["divergence"]["gap_pct"] == approx(35.0)
    assert len(out["intensity_history"]) == len(history)
    assert set(out["intensity_history"][0]) >= {"date", "intensity", "state"}


def test_compute_orchestrator_cyclical_when_divergence_low():
    gaps = [-0.15] * 40   # divergenza sempre bassa -> mai debasement
    history = [{"date": f"20{20 + i // 12:02d}-{i % 12 + 1:02d}-28", "gap": g}
               for i, g in enumerate(gaps)]
    divergence = {
        "asof": history[-1]["date"], "gap": gaps[-1], "gap_pct": gaps[-1] * 100,
        "gold": 200.0, "fair_value": 235.0, "real_yield": 1.0,
        "window": 60, "n": 100, "history": history,
    }
    out = compute_debasement_regime(divergence, None)
    assert out["meta"]["active"] == "cyclical"
    assert out["meta"]["sub_phase"] is None


def test_compute_orchestrator_graceful_on_empty():
    assert compute_debasement_regime(None, None) is None
    assert compute_debasement_regime({"history": []}, None) is None


def test_thresholds_sane():
    assert 0.0 < OFF_THRESHOLD < ON_THRESHOLD < 1.0
    assert CONFIRM_ON >= 1 and CONFIRM_OFF > CONFIRM_ON
