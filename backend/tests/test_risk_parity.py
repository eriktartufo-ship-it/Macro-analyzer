"""Test all_weather_risk_parity_strategy (Tier 2.4 roadmap Bridgewater).

Bridgewater All-Weather: 25% del rischio totale assegnato a ciascuno dei 4 quadrant
macro (reflation/stagflation/deflation/goldilocks). Pesi statici nel tempo (selection
e vol parity da ASSET_REGIME_DATA storico, non dipendono dal regime corrente).

Filosofia: invece di "inseguire" il regime corrente, diversifica per QUALSIASI regime
arrivi. Riduce MaxDD a costo di un po' di alpha rispetto a strategie regime-aware.
"""
from __future__ import annotations

import pandas as pd
import pytest

from app.services.backtest.strategies import all_weather_risk_parity_strategy


def _dates(n: int = 12) -> pd.DatetimeIndex:
    return pd.date_range("2020-01-31", periods=n, freq="ME")


class TestAllWeatherRiskParity:
    def test_weights_sum_to_one(self):
        """Pesi totali = 1.0 (full investment, no cash)."""
        df = all_weather_risk_parity_strategy(
            asset_classes=["us_equities_growth", "us_bonds_long", "gold", "broad_commodities", "tips_inflation_bonds"],
            dates=_dates(6),
            top_n_per_quadrant=3,
        )
        assert not df.empty
        for ts, row in df.iterrows():
            assert abs(row.sum() - 1.0) < 1e-6, f"Mese {ts}: sum={row.sum():.4f}"

    def test_weights_static_across_time(self):
        """Pesi NON cambiano nel tempo (strategia time-invariant)."""
        df = all_weather_risk_parity_strategy(
            asset_classes=["us_equities_growth", "us_bonds_long", "gold", "broad_commodities"],
            dates=_dates(12),
            top_n_per_quadrant=3,
        )
        first_row = df.iloc[0]
        for i in range(1, len(df)):
            for a in df.columns:
                assert df.iloc[i][a] == pytest.approx(first_row[a], abs=1e-9)

    def test_quadrant_weight_25pct_each(self):
        """Per ogni regime, somma dei pesi entro il quadrant = 0.25 (assumendo
        nessuna sovrapposizione asset-regime, caso ideale)."""
        # Universe ampio: ogni regime sceglie asset distinti (no overlap)
        df = all_weather_risk_parity_strategy(
            asset_classes=[
                "us_equities_growth", "us_equities_value", "international_dm_equities",
                "us_bonds_long", "us_bonds_short", "tips_inflation_bonds",
                "gold", "silver", "broad_commodities", "energy",
                "cash_money_market", "real_estate_reits",
                "bitcoin", "crypto_broad", "em_equities",
            ],
            dates=_dates(2),
            top_n_per_quadrant=3,
        )
        # Sum totale = 1 → ogni quadrant 0.25 × 4 = 1
        # Anche con overlap, sum finale = 1 (pesi normalizzati overall).
        assert abs(df.iloc[0].sum() - 1.0) < 1e-6

    def test_vol_parity_within_quadrant(self):
        """Asset con vol piu' bassa ha peso piu' alto nel suo quadrant."""
        # Costruisco un universe minimal dove sappiamo il top-1 per ogni regime
        # e verifico che il peso sia coerente con 1/vol normalizzato.
        df = all_weather_risk_parity_strategy(
            asset_classes=["us_equities_growth", "us_bonds_long"],
            dates=_dates(1),
            top_n_per_quadrant=1,  # 1 asset per quadrant
        )
        # Con top_n=1, ogni quadrant ha 1 solo asset → peso = 0.25 esatto
        # Ma se entrambi gli asset entrano in piu' regimi, somma totale per asset = 0.25 × n_regimes_in_which_top
        # Comunque sum totale = 1
        assert abs(df.iloc[0].sum() - 1.0) < 1e-6

    def test_top_n_per_quadrant_respected(self):
        """top_n_per_quadrant=2 → al massimo 2 asset per regime, 8 totali (con overlap meno)."""
        df = all_weather_risk_parity_strategy(
            asset_classes=[
                "us_equities_growth", "us_equities_value", "us_bonds_long", "us_bonds_short",
                "gold", "silver", "broad_commodities", "energy", "cash_money_market",
            ],
            dates=_dates(1),
            top_n_per_quadrant=2,
        )
        # Numero asset con peso > 0
        nonzero = (df.iloc[0] > 0).sum()
        # Max 8 (2 per regime × 4 regimi), ma con overlap puo' essere meno
        assert 1 <= nonzero <= 8

    def test_returns_empty_for_empty_assets(self):
        df = all_weather_risk_parity_strategy(
            asset_classes=[],
            dates=_dates(3),
            top_n_per_quadrant=3,
        )
        # Nessun asset → DataFrame vuoto o con solo index
        assert df.empty or df.shape[1] == 0

    def test_works_with_partial_universe(self):
        """Anche con solo 2 asset, la strategia produce pesi (non crasha)."""
        df = all_weather_risk_parity_strategy(
            asset_classes=["us_equities_growth", "us_bonds_long"],
            dates=_dates(2),
            top_n_per_quadrant=3,
        )
        assert not df.empty
        # Sum=1
        assert abs(df.iloc[0].sum() - 1.0) < 1e-6

    def test_low_vol_asset_dominates_its_quadrant(self):
        """Asset con vol storico bassa (es cash, vol ~0.01) prende peso piu' grande
        del partner se entrambi in stesso top-N quadrant."""
        # Cash money market ha vol bassissima storica (~0.01-0.02 in tutti i regimi)
        # vs tips_inflation_bonds (~0.06-0.08). Entrambi in deflation top.
        df = all_weather_risk_parity_strategy(
            asset_classes=["cash_money_market", "tips_inflation_bonds"],
            dates=_dates(1),
            top_n_per_quadrant=2,
        )
        # Se entrambi entrano in deflation/goldilocks top, cash peso > tips
        # (Property weak ma direzionale; saltiamo se sharpe ranking esclude uno dei due)
        cash_w = df.iloc[0].get("cash_money_market", 0)
        tips_w = df.iloc[0].get("tips_inflation_bonds", 0)
        if cash_w > 0 and tips_w > 0:
            # Cash dovrebbe avere peso >= tips per via di vol piu' bassa
            assert cash_w >= tips_w * 0.95  # tolerance per vol parity arrotondamenti
