"""Ogni condizione del classificatore ha l'etichetta italiana, e l'etichetta non tocca
l'impronta del modello (model_lock fa l'md5 di REGIME_CONDITIONS)."""
from __future__ import annotations

from app.services.regime import classifier
from app.services.regime.condition_labels_it import CONDITION_LABEL_IT, condition_label_it


def _all_descriptions() -> set[str]:
    out: set[str] = set()
    for name, value in vars(classifier).items():
        if not isinstance(value, dict) or not (name == "REGIME_CONDITIONS" or name.endswith(("_PILLAR_ADD", "_PILLARS_ADD"))):
            continue
        for conds in value.values():
            for cfg in conds.values():
                out.add(cfg["description"])
    return out


def test_ogni_condizione_ha_la_sua_etichetta_italiana():
    descriptions = _all_descriptions()
    assert len(descriptions) > 90  # il censimento ha trovato davvero le condizioni
    missing = sorted(d for d in descriptions if d not in CONDITION_LABEL_IT)
    assert missing == []


def test_nessuna_etichetta_orfana():
    # Un'etichetta senza condizione è una soglia cambiata nel classificatore e non qui.
    orphans = sorted(set(CONDITION_LABEL_IT) - _all_descriptions())
    assert orphans == []


def test_descrizione_sconosciuta_resta_quella_originale():
    assert condition_label_it("Nuova condizione X > 1") == "Nuova condizione X > 1"


def test_ogni_forza_della_traiettoria_ha_la_sua_etichetta():
    from app.services.regime import trajectory as t

    assert set(t._TREND_LABEL_IT) == set(t.TREND_REGIME_PRESSURE)
    assert set(t._NEWS_LABEL_IT) == set(t.NEWS_REGIME_PRESSURE)
    assert set(t._INSIDER_LABEL_IT) == set(t.INSIDER_REGIME_PRESSURE)
    assert set(t._DEDOLLAR_LABEL_IT) == set(t.DEDOLLAR_REGIME_PRESSURE)


def test_le_condizioni_dei_pilastri_hanno_la_descrizione():
    # Prima il sito mostrava «inflation_persistent» perché cercava solo in REGIME_CONDITIONS.
    from app.services.regime.condition_labels_it import condition_description

    assert condition_description("stagflation", "inflation_persistent") == "min(CPI YoY last 6m) > 2.5%"
    assert condition_description("stagflation", "inflation_high") == "CPI YoY > 4%"
    assert condition_description("stagflation", "non_esiste") is None
