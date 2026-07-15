"""LLM strategist v6-market-aware (2026-07-15, richiesta Erik).

BUG STRUTTURALE che questi test blindano: fino alla v5 il bot Gemini sceglieva 24
asset **senza vedere un solo prezzo** — il `price_history` era fetchato DOPO la
chiamata e usato solo per loggare (momentum_signal/vol_60d). Qui si verifica che i
dati Yahoo entrino DAVVERO nel prompt e che il regime sia un input di prima classe
con dissenso motivato. Test sul CONTRATTO (no rete).
"""
from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd

from app.services.ai_portfolio import llm_strategist as S


def _hist():
    """Prezzi sintetici per 3 asset: uno che sale, uno che scende, il benchmark."""
    idx = pd.bdate_range("2025-01-01", periods=200)
    rng = np.random.default_rng(3)
    up = 100 * np.cumprod(1 + rng.normal(0.0015, 0.008, 200))    # forte
    down = 100 * np.cumprod(1 + rng.normal(-0.0010, 0.008, 200))  # debole
    bench = 100 * np.cumprod(1 + rng.normal(0.0003, 0.008, 200))
    return {
        "gold": pd.Series(up, index=idx),
        "sector_technology": pd.Series(down, index=idx),
        "us_equities_growth": pd.Series(bench, index=idx),  # benchmark S&P
    }


class TestMarketBlock:
    def test_blocco_contiene_le_metriche_di_prezzo(self):
        b = S._build_market_block(_hist())
        assert "RSI" in b and "MA50" in b and "vol" in b and "rel_3m" in b

    def test_blocco_contiene_gli_asset(self):
        b = S._build_market_block(_hist())
        assert "gold" in b and "sector_technology" in b

    def test_forza_relativa_ha_segno_sensato(self):
        """L'asset forte deve avere rel_3m positiva, il debole negativa."""
        b = S._build_market_block(_hist())
        riga_gold = [r for r in b.split("\n") if r.strip().startswith("- gold")][0]
        riga_tech = [r for r in b.split("\n") if "sector_technology" in r][0]
        assert "rel_3m vs S&P +" in riga_gold
        assert "rel_3m vs S&P -" in riga_tech

    def test_senza_prezzi_non_inventa(self):
        b = S._build_market_block({})
        assert "non disponibili" in b


class TestUserPromptV6:
    def _prompt(self, price_history):
        return S._build_user_prompt(
            target_date=date(2026, 7, 15),
            regime="stagflation",
            probabilities={"stagflation": 0.6, "reflation": 0.2,
                           "deflation": 0.1, "goldilocks": 0.1},
            confidence=0.7,
            indicators={"cpi_yoy": 4.27, "cpi_yoy_change_6m": -0.8},
            positions=[],
            learnings=[],
            price_history=price_history,
        )

    def test_i_prezzi_yahoo_entrano_nel_prompt(self):
        """IL FIX: prima della v6 questo blocco non esisteva proprio."""
        p = self._prompt(_hist())
        assert "LETTURA DI MERCATO" in p
        assert "gold" in p and "RSI" in p

    def test_regime_e_input_di_prima_classe_non_ignorabile(self):
        p = self._prompt(_hist())
        assert "input di prima classe" in p
        assert "DEVI considerarlo" in p
        # la vecchia formula anti-anchoring non deve sopravvivere
        assert "puoi ignorare" not in p

    def test_dissenso_ammesso_ma_motivato(self):
        p = self._prompt(_hist())
        assert "PUOI dissentire" in p
        assert "regime_dissent" in p

    def test_chiede_di_incrociare_macro_e_prezzo(self):
        p = self._prompt(_hist())
        assert "MACRO e PREZZO" in p

    def test_spiega_la_rotazione(self):
        p = self._prompt(_hist())
        assert "ROTAZIONE" in p

    def test_prompt_regge_senza_price_history(self):
        """Se Yahoo è giù il prompt non deve rompersi (graceful degradation)."""
        p = self._prompt(None)
        assert "LETTURA DI MERCATO" in p
        assert "non disponibili" in p


class TestSystemPromptV6:
    def test_dichiara_le_due_fonti(self):
        assert "LE DUE FONTI" in S._SYSTEM_PROMPT

    def test_schema_ha_regime_dissent(self):
        assert '"regime_dissent"' in S._SYSTEM_PROMPT

    def test_spiega_conflitto_macro_prezzi(self):
        p = S._SYSTEM_PROMPT
        assert "CONCORDANO" in p and "confliggono" in p


def test_summary_estrae_regime_dissent():
    parsed = {"thesis": "x", "outlook": "y", "regime_dissent": "dissento: lei_roc +0.3"}
    assert str(parsed.get("regime_dissent", ""))[:300] == "dissento: lei_roc +0.3"


class TestMonotoneYieldAssets:
    """Cash/T-bill: il prezzo accumula interessi e NON fa trend. Momentum/RSI sono
    degeneri (sempre BULLISH, RSI ~95-100) e un LLM li legge come conviction enorme.
    Osservato sul probe reale: Gemini scrisse "cash_money_market ... RSI a 95, perfetto".
    """

    def _hist_cash(self):
        idx = pd.bdate_range("2025-01-01", periods=200)
        # accumulo monotòno tipo BIL: +0.02%/giorno, nessuna discesa
        monotone = 100 * np.cumprod(np.full(200, 1.0002))
        bench = 100 * np.cumprod(1 + np.random.default_rng(7).normal(0.0003, 0.008, 200))
        return {
            "cash_money_market": pd.Series(monotone, index=idx),
            "us_equities_growth": pd.Series(bench, index=idx),
        }

    def test_cash_non_mostra_momentum_ne_rsi(self):
        b = S._build_market_block(self._hist_cash())
        riga = [r for r in b.split("\n") if "cash_money_market" in r][0]
        assert "RSI" not in riga
        assert "BULLISH" not in riga

    def test_cash_dichiara_perche_non_ha_trend(self):
        b = S._build_market_block(self._hist_cash())
        riga = [r for r in b.split("\n") if "cash_money_market" in r][0]
        assert "NON fa trend" in riga

    def test_cash_mostra_comunque_vol_e_rendimenti(self):
        b = S._build_market_block(self._hist_cash())
        riga = [r for r in b.split("\n") if "cash_money_market" in r][0]
        assert "vol" in riga and "ret 1m" in riga

    def test_asset_normale_mantiene_il_momentum(self):
        """La soppressione vale SOLO per i monotòni, non per tutti."""
        b = S._build_market_block(_hist())
        riga = [r for r in b.split("\n") if r.strip().startswith("- gold")][0]
        assert "RSI" in riga
