"""Test per i nuovi dati P1 (gap analysis audit 2026-07-12).

Coprono: net liquidity (unità normalizzate), fed funds futures ZQ (symbol +
implied rate), CFTC COT (net %OI + z-score), derive_pmi_proxy con survey Fed
regionali, igiene fred_codes (no chiavi duplicate a livello di sorgente).
"""
from __future__ import annotations

import re
from datetime import date
from pathlib import Path

import pandas as pd

from app.services.alt_data.cftc_cot import compute_net_pct_oi, zscore_last
from app.services.common.indicator_aliases import derive_pmi_proxy
from app.services.indicators.fed_funds_futures import (
    contract_symbol,
    fetch_implied_12m,
    implied_rate,
    target_month,
)
from app.services.indicators.liquidity import compute_net_liquidity


class TestNetLiquidity:
    """Unità FRED diverse: WALCL $M, RRP $B, TGA $M → output $B."""

    def test_units_normalized_2026_live_values(self):
        # Valori reali 2026-07-08 verificati su FRED
        nl = compute_net_liquidity(6735609.0, 0.545, 774062.0)
        assert nl is not None
        assert abs(nl - (6735.609 - 0.545 - 774.062)) < 0.01  # ~5961 $B

    def test_missing_piece_returns_none(self):
        assert compute_net_liquidity(None, 0.5, 774062.0) is None
        assert compute_net_liquidity(6735609.0, None, 774062.0) is None
        assert compute_net_liquidity(6735609.0, 0.5, None) is None

    def test_wrong_magnitude_walcl_rejected(self):
        # WALCL passato in $B invece che $M → fuori sanity range → None
        assert compute_net_liquidity(6735.6, 0.5, 774062.0) is None


class TestFedFundsFutures:
    def test_contract_symbol_codes(self):
        assert contract_symbol(date(2027, 7, 1)) == "ZQN27.CBT"
        assert contract_symbol(date(2027, 1, 1)) == "ZQF27.CBT"
        assert contract_symbol(date(2026, 12, 1)) == "ZQZ26.CBT"

    def test_target_month_wraps_year(self):
        assert target_month(date(2026, 7, 12), 12) == date(2027, 7, 1)
        assert target_month(date(2026, 2, 28), 11) == date(2027, 1, 1)

    def test_implied_rate_math_and_sanity(self):
        import pytest
        assert implied_rate(95.925) == pytest.approx(4.075)
        assert implied_rate(100.0) == 0.0
        assert implied_rate(120.0) is None  # rate -20 = dato rotto
        assert implied_rate("abc") is None

    def test_fetch_uses_fallback_months(self):
        class FakeFetcher:
            def __init__(self):
                self.calls = []

            def fetch(self, symbol):
                self.calls.append(symbol)
                if len(self.calls) < 2:
                    return pd.Series(dtype=float)  # 12m vuoto
                return pd.Series([95.9])

        f = FakeFetcher()
        rate = fetch_implied_12m(f, today=date(2026, 7, 12))
        assert rate == 100.0 - 95.9
        assert len(f.calls) == 2  # 12m fallito → 11m ok

    def test_fetch_all_fail_returns_none(self):
        class DeadFetcher:
            def fetch(self, symbol):
                raise RuntimeError("no data")

        assert fetch_implied_12m(DeadFetcher(), today=date(2026, 7, 12)) is None


class TestCotMath:
    def test_net_pct_oi(self):
        assert compute_net_pct_oi(300, 100, 1000) == 20.0
        assert compute_net_pct_oi(100, 300, 1000) == -20.0
        assert compute_net_pct_oi(100, 50, 0) is None
        assert compute_net_pct_oi("x", 50, 100) is None

    def test_zscore_last(self):
        flat = [10.0] * 40
        assert zscore_last(flat) is None  # stdev 0
        series = [0.0] * 39 + [10.0]
        z = zscore_last(series)
        assert z is not None and z > 3.0  # ultimo valore = outlier
        assert zscore_last([1.0, 2.0]) is None  # storia corta


class TestPmiProxySurveys:
    """P1: le survey Fed regionali (leading) battono indpro (coincident)."""

    def test_surveys_preferred_over_indpro(self):
        # indpro forte (+5% → vecchio proxy 75 clampato 65) ma survey deboli
        pmi = derive_pmi_proxy({
            "indpro_roc_12m": 5.0,
            "empire_fed_activity": -12.0,
            "philly_fed_activity": -6.0,
        })
        # composite -9 → 50 - 3 = 47 (contrazione, NON 65)
        assert pmi == 47.0

    def test_composite_mapping_and_clamp(self):
        assert derive_pmi_proxy({"empire_fed_activity": 30.0}) == 60.0
        assert derive_pmi_proxy({"empire_fed_activity": 90.0}) == 65.0  # clamp
        assert derive_pmi_proxy({"empire_fed_activity": -90.0}) == 35.0

    def test_fallback_indpro_when_no_surveys(self):
        assert derive_pmi_proxy({"indpro_roc_12m": 2.0}) == 60.0  # 50 + 5*2

    def test_real_pmi_still_wins(self):
        assert derive_pmi_proxy({"pmi": 42.0, "empire_fed_activity": 30.0}) == 42.0


class TestFredCodesHygiene:
    """AUDIT 2026-07-12: 5 chiavi erano definite due volte (override silenzioso)."""

    def test_no_duplicate_keys_in_source(self):
        src = Path(__file__).parent.parent / "app" / "services" / "indicators" / "fred_codes.py"
        text = src.read_text(encoding="utf-8")
        keys = re.findall(r'^    "([a-z0-9_]+)": \{', text, flags=re.MULTILINE)
        dupes = {k for k in keys if keys.count(k) > 1}
        assert not dupes, f"chiavi duplicate in FRED_SERIES: {dupes}"

    def test_new_p1_series_registered(self):
        from app.services.indicators.fred_codes import FRED_SERIES

        for key, fred_id in [
            ("treasury_general_account", "WTREGEN"),
            ("empire_fed_activity", "GACDISA066MSFRBNY"),
            ("philly_fed_activity", "GACDFSA066MSFRBPHI"),
            ("dallas_fed_activity", "BACTSAMFRBDAL"),
            ("korea_exports", "XTEXVA01KRM667S"),
        ]:
            assert key in FRED_SERIES, f"{key} non registrata"
            assert FRED_SERIES[key]["fred_id"] == fred_id

    def test_discontinued_wilshire_removed(self):
        from app.services.indicators.fred_codes import FRED_SERIES

        assert "wilshire_5000" not in FRED_SERIES
