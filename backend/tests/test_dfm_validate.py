"""Test DFM walk-forward validation."""
from __future__ import annotations

from types import SimpleNamespace

import pytest

from app.services.validation.dfm_validate import validate_dfm_walkforward


def _make_row(date_str: str, gdp_yoy_dfm: float, gdp_roc: float):
    return SimpleNamespace(
        date=date_str,
        conditions_met={"indicators": {"gdp_yoy_dfm": gdp_yoy_dfm, "gdp_roc": gdp_roc}},
    )


class TestDFMValidate:
    def test_insufficient_data(self):
        rows = [_make_row(f"2024-{i:02d}-01", 2.5, 0.5) for i in range(1, 6)]
        out = validate_dfm_walkforward(rows)
        assert out.is_reliable is False
        assert "Insufficient" in out.diagnostic
        assert out.n_observations == 5

    def test_perfect_match_high_r2(self):
        """gdp_yoy_dfm == gdp_roc × 4 → R² = 1."""
        rows = [_make_row(f"d-{i}", v * 4, v) for i, v in enumerate([0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 0.0, -0.5, -1.0, 1.2, 1.8, 2.3, 2.8, 0.7])]
        out = validate_dfm_walkforward(rows)
        assert out.r_squared > 0.95

    def test_systematic_bias(self):
        """gdp_yoy_dfm sempre sopra reale di +1.5 → bias positivo."""
        rows = [_make_row(f"d-{i}", v * 4 + 1.5, v) for i, v in enumerate([0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 0.0, -0.5, -1.0, 1.2, 1.8, 2.3, 2.8, 0.7])]
        out = validate_dfm_walkforward(rows)
        assert out.bias > 1.0
        assert out.is_reliable is False  # bias > 1.0 fails reliability gate

    def test_no_predicted_skipped(self):
        rows = [
            _make_row(f"d-{i}", None, 1.0)
            for i in range(15)
        ]
        out = validate_dfm_walkforward(rows)
        assert out.n_observations == 0
