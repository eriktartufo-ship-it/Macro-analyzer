"""Test cycle clock (Investment Clock). Spec: S46."""
from __future__ import annotations

import pandas as pd

from app.services.backtest.cycle_clock import (
    PHASE_META,
    compute_cycle_timeline,
    phase_meta_payload,
    project,
)


def _pure(regime):
    return {"reflation": 0.0, "stagflation": 0.0, "deflation": 0.0, "goldilocks": 0.0} | {regime: 1.0}


def test_pure_regime_lands_on_its_corner():
    # goldilocks -> TL (x=-1 infl basso, y=+1 crescita alta) = Recovery
    g = project(_pure("goldilocks"))
    assert g["x"] == -1.0 and g["y"] == 1.0
    assert g["phase"] == "recovery" and g["phase_ml"] == "Recovery"
    # reflation -> TR Overheat
    r = project(_pure("reflation"))
    assert r["x"] == 1.0 and r["y"] == 1.0 and r["phase"] == "overheat"
    # stagflation -> BR
    s = project(_pure("stagflation"))
    assert s["x"] == 1.0 and s["y"] == -1.0 and s["phase"] == "stagflation"
    # deflation -> BL Reflation/Recessione
    d = project(_pure("deflation"))
    assert d["x"] == -1.0 and d["y"] == -1.0 and d["phase"] == "reflation"


def test_pure_regime_is_sharp_not_transition():
    g = project(_pure("goldilocks"))
    assert g["sharpness"] == 1.0        # entropia 0 -> ago netto
    assert g["transition"] is False
    assert g["top_prob"] == 1.0


def test_equiprobable_is_center_blurred_transition():
    p = project({r: 0.25 for r in ["reflation", "stagflation", "deflation", "goldilocks"]})
    assert abs(p["x"]) < 1e-9 and abs(p["y"]) < 1e-9   # centro
    assert p["sharpness"] == 0.0                        # entropia max -> ago sfumato
    assert p["transition"] is True                     # posizione incerta


def test_probs_normalized_and_bounded():
    # input non normalizzato -> rinormalizza, x/y restano nel quadrato [-1,1]
    p = project({"reflation": 2.0, "goldilocks": 2.0, "stagflation": 0.0, "deflation": 0.0})
    assert abs(sum(p["probs"].values()) - 1.0) < 1e-6
    assert -1.0 <= p["x"] <= 1.0 and -1.0 <= p["y"] <= 1.0
    # reflation+goldilocks a pari -> crescita alta (y=+1), inflazione neutra (x=0)
    assert p["y"] == 1.0 and abs(p["x"]) < 1e-9


def test_confidence_overrides_sharpness_for_opacity():
    p = project(_pure("goldilocks"), confidence=0.2)
    assert p["confidence"] == 0.2
    assert p["transition"] is True     # confidence < 0.30 forza transizione


def test_phase_meta_has_4_phases_with_ml_names():
    pm = phase_meta_payload()
    assert len(pm) == 4
    names = {p["ml_name"] for p in pm}
    assert names == {"Recovery", "Overheat", "Stagflation", "Reflation"}
    for p in pm:
        assert p["leaders"] and all(set(l) == {"asset", "label"} for l in p["leaders"])
    # collisione nomi risolta: il regime "reflation" del classifier sta nel quadrante Overheat
    overheat = next(p for p in PHASE_META if p["ml_name"] == "Overheat")
    assert overheat["regime"] == "reflation"


def _frame(weeks, probs_seq, conf=0.7):
    idx = pd.to_datetime(weeks)
    data = {f"p_{r}": [ps[r] for ps in probs_seq]
            for r in ["reflation", "stagflation", "deflation", "goldilocks"]}
    data["confidence"] = [conf] * len(weeks)
    return pd.DataFrame(data, index=idx)


def test_timeline_shape_and_alignment():
    weeks = ["2024-01-05", "2024-01-12", "2024-01-19"]
    seq = [_pure("goldilocks"), _pure("reflation"), _pure("stagflation")]
    tl = compute_cycle_timeline(_frame(weeks, seq), n_weeks=52)
    assert tl["weeks"] == weeks
    assert len(tl["frames"]) == 3
    assert tl["frames"][0]["phase"] == "recovery"
    assert tl["frames"][-1]["phase"] == "stagflation"


def test_agreement_and_anomalies():
    weeks = ["2024-01-05"]
    seq = [_pure("goldilocks")]  # fase Recovery -> leader teorici: growth/tech/financials/value/momentum
    obs = {"2024-01-05": ["us_equities_growth", "gold"]}  # growth confirm, gold anomalia
    tl = compute_cycle_timeline(_frame(weeks, seq), observed_leaders_by_week=obs)
    f = tl["frames"][0]
    agr = {a["asset"] for a in f["agreement"]}
    ano = {a["asset"] for a in f["anomalies"]}
    assert "us_equities_growth" in agr
    assert "gold" in ano


def test_empty_frame_safe():
    tl = compute_cycle_timeline(pd.DataFrame())
    assert tl["frames"] == [] and len(tl["phase_meta"]) == 4
