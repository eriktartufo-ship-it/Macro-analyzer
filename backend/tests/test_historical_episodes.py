"""Test stress historical replay (Tier 4.9 roadmap Bridgewater).

6 episodi macro storici parametrizzati come delta indicators. L'utente
applica un episodio al regime corrente per vedere "se oggi arriva un
2008 GFC, in quale regime finiamo? Quali asset crollano?".

Episodi:
- 1973_oil_shock     → stagflation (CPI surge + GDP collapse)
- 1987_black_monday  → volatility shock (VIX spike, credit OK)
- 2000_dotcom_bust   → equity crash + soft deflation
- 2008_gfc           → deflation severa (credit + GDP + VIX)
- 2020_covid         → rapid deflation (lockdown shock)
- 2022_inflation_surge → stagflation (post-COVID inflation)
"""
from __future__ import annotations

import pytest

from app.services.regime.shock_scenarios import (
    PRESET_SCENARIOS,
    _apply_deltas,
    list_preset_scenarios,
)


HISTORICAL_KEYS = (
    "historical_1973_oil_shock",
    "historical_1987_black_monday",
    "historical_2000_dotcom_bust",
    "historical_2008_gfc",
    "historical_2020_covid",
    "historical_2022_inflation_surge",
)


class TestHistoricalEpisodesRegistration:
    def test_all_6_episodes_in_preset(self):
        for key in HISTORICAL_KEYS:
            assert key in PRESET_SCENARIOS, f"Missing: {key}"

    def test_all_have_required_fields(self):
        for key in HISTORICAL_KEYS:
            cfg = PRESET_SCENARIOS[key]
            assert "label" in cfg and cfg["label"]
            assert "description" in cfg and cfg["description"]
            assert "deltas" in cfg and isinstance(cfg["deltas"], dict)
            assert len(cfg["deltas"]) >= 2, f"{key} ha pochi deltas (<2)"

    def test_listed_via_endpoint_function(self):
        all_keys = {s["key"] for s in list_preset_scenarios()}
        for key in HISTORICAL_KEYS:
            assert key in all_keys


class TestHistoricalEpisodeShape:
    """Verifica che ogni episodio applicato a uno scenario "neutro" produca
    delta indicators coerenti con il regime atteso storicamente."""

    def _neutral_baseline(self) -> dict[str, float]:
        """Baseline economia goldilocks: regime atteso prima dell'episodio."""
        return {
            "gdp_roc": 2.0, "pmi": 53.0, "cpi_yoy": 2.0, "unrate": 4.0,
            "unrate_roc": 0.0, "yield_curve_10y2y": 1.0, "initial_claims_roc": 0.0,
            "lei_roc": 0.5, "fed_funds_rate": 2.0, "core_pce_yoy": 2.0,
            "payrolls_roc_12m": 1.5, "indpro_roc_12m": 2.0, "baa_spread": 1.8,
            "consumer_sentiment": 85.0, "vix": 15.0, "nfci": -0.3,
            "breakeven_10y": 2.0,
        }

    def test_1973_oil_shock_inflation_jumps(self):
        """Oil shock storico: CPI da ~4% a >10%."""
        base = self._neutral_baseline()
        cfg = PRESET_SCENARIOS["historical_1973_oil_shock"]
        shocked = _apply_deltas(base, cfg["deltas"])
        # CPI deve salire significativamente (almeno +3pp)
        assert shocked["cpi_yoy"] - base["cpi_yoy"] >= 3.0

    def test_2008_gfc_credit_spread_explodes(self):
        """GFC: BAA spread da ~1.5% a ~5%."""
        base = self._neutral_baseline()
        cfg = PRESET_SCENARIOS["historical_2008_gfc"]
        shocked = _apply_deltas(base, cfg["deltas"])
        # Credit spread sale di almeno 200bp
        assert shocked["baa_spread"] - base["baa_spread"] >= 2.0
        # VIX schizza
        assert shocked["vix"] - base["vix"] >= 15.0

    def test_2020_covid_unemployment_explodes(self):
        """COVID: unemp da ~3.5% a ~14% in pochi mesi."""
        base = self._neutral_baseline()
        cfg = PRESET_SCENARIOS["historical_2020_covid"]
        shocked = _apply_deltas(base, cfg["deltas"])
        # Unrate ROC YoY positivo grosso (lavoratori lay-off)
        assert shocked["unrate_roc"] - base["unrate_roc"] >= 5.0

    def test_2022_inflation_surge_cpi_and_rates(self):
        """2022: CPI 9% picco + Fed funds da 0.25% a 5%+."""
        base = self._neutral_baseline()
        cfg = PRESET_SCENARIOS["historical_2022_inflation_surge"]
        shocked = _apply_deltas(base, cfg["deltas"])
        # CPI sale di almeno 4pp
        assert shocked["cpi_yoy"] - base["cpi_yoy"] >= 4.0
        # Fed funds sale di almeno 3pp
        assert shocked["fed_funds_rate"] - base["fed_funds_rate"] >= 3.0

    def test_2000_dotcom_equity_proxy_via_vix(self):
        """Dotcom: equity crash, ma credit relativamente OK (no banking crisis)."""
        base = self._neutral_baseline()
        cfg = PRESET_SCENARIOS["historical_2000_dotcom_bust"]
        shocked = _apply_deltas(base, cfg["deltas"])
        # VIX sale ma meno di 2008 (era ~30 vs 80 di 2008)
        assert shocked["vix"] - base["vix"] >= 10.0
        # GDP rallenta ma non collassa come 2008
        assert -5.0 < shocked["gdp_roc"] - base["gdp_roc"] < 0

    def test_1987_black_monday_pure_volatility(self):
        """1987: equity crash isolato, fondamentali OK (no recession)."""
        base = self._neutral_baseline()
        cfg = PRESET_SCENARIOS["historical_1987_black_monday"]
        shocked = _apply_deltas(base, cfg["deltas"])
        # VIX spike grosso
        assert shocked["vix"] - base["vix"] >= 15.0
        # GDP non collassa drammaticamente (recession leggera 1990 separata)
        assert shocked["gdp_roc"] - base["gdp_roc"] > -2.0
