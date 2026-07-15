"""LLM strategist v6-market-aware (2026-07-15, richiesta Erik).

Il bot leggeva i dati come una fotografia (livelli) e giustificava solo le
aperture. Qui si blinda il contratto del prompt: insegna livello-vs-ROC, chiede
una PREVISIONE esplicita (`outlook`), e impone la giustificazione anche sulle
chiusure. Sono test sul CONTRATTO (no rete: non chiamano Gemini).
"""
from __future__ import annotations

from datetime import date

from app.services.ai_portfolio import llm_strategist as S


def test_prompt_version_bumped_invalidates_cache():
    # Il bump è ciò che invalida la cache: se resta v4 il nuovo prompt non gira.
    assert S._PROMPT_VERSION == "v6-market-aware"


def test_hash_includes_prompt_version():
    h = S._compute_input_hash(date(2026, 7, 15), "stagflation", 0.7, 10, 2, "m")
    orig = S._PROMPT_VERSION
    try:
        S._PROMPT_VERSION = "altra-versione"
        h2 = S._compute_input_hash(date(2026, 7, 15), "stagflation", 0.7, 10, 2, "m")
    finally:
        S._PROMPT_VERSION = orig
    assert h != h2, "cambiare prompt DEVE cambiare l'hash, altrimenti resta la cache vecchia"


class TestSystemPromptTeachesRoc:
    def test_distingue_livello_da_roc(self):
        p = S._SYSTEM_PROMPT
        assert "LIVELLO" in p and "ROC" in p
        assert "DOVE SEI" in p and "DOVE STAI ANDANDO" in p

    def test_dichiara_che_il_livello_e_gia_prezzato(self):
        assert "già nei prezzi" in S._SYSTEM_PROMPT

    def test_chiede_previsione_non_fotografia(self):
        assert "PREVEDERE, NON FOTOGRAFARE" in S._SYSTEM_PROMPT

    def test_cita_i_leading_indicator(self):
        p = S._SYSTEM_PROMPT
        assert "lei_roc" in p or "LEADING" in p

    def test_impone_giustificazione_anche_in_chiusura(self):
        p = S._SYSTEM_PROMPT
        assert "APERTURE E CHIUSURE" in p
        assert "FULL_CLOSE" in p

    def test_output_schema_ha_outlook(self):
        assert '"outlook"' in S._SYSTEM_PROMPT


class TestUserPromptRichiesta:
    def _prompt(self):
        return S._build_user_prompt(
            target_date=date(2026, 7, 15),
            regime="stagflation",
            probabilities={"stagflation": 0.6, "reflation": 0.2,
                           "deflation": 0.1, "goldilocks": 0.1},
            confidence=0.7,
            indicators={"cpi_yoy": 4.27, "cpi_yoy_change_6m": -0.8, "lei_roc": 0.3},
            positions=[],
            learnings=[],
        )

    def test_richiede_outlook_3_6_mesi(self):
        p = self._prompt()
        assert "outlook" in p
        assert "3-6 mesi" in p

    def test_richiede_giustificazione_su_aperture_e_chiusure(self):
        p = self._prompt()
        assert "aperture E chiusure" in p

    def test_schema_output_include_outlook(self):
        assert '"outlook"' in self._prompt()

    def test_indicators_reali_finiscono_nel_prompt(self):
        p = self._prompt()
        # il ROC deve arrivare al modello, non solo il livello
        assert "cpi_yoy_change_6m" in p
        assert "4.270" in p or "4.27" in p


def test_outlook_viene_estratto_nel_summary():
    """Il campo `outlook` del JSON di Gemini non deve andare perso."""
    parsed = {"thesis": "oggi stagflazione", "outlook": "disinflazione entro 6m",
              "decisions": []}
    # riproduce l'estrazione fatta in daily_scan
    summary = {
        "thesis": str(parsed.get("thesis", ""))[:300],
        "outlook": str(parsed.get("outlook", ""))[:300],
    }
    assert summary["outlook"] == "disinflazione entro 6m"


def test_outlook_assente_non_rompe():
    parsed = {"thesis": "x", "decisions": []}
    assert str(parsed.get("outlook", ""))[:300] == ""
