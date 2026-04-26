"""Test factor-regime mapping (Fama-French × regime DB)."""
import pytest

from app.services.factors.fama_french import factor_keys, _parse_ff_csv


class TestFamaFrenchParser:
    def test_parses_3factor_format(self):
        text = (
            "header line\r\n"
            "...metadata...\r\n"
            "\r\n"
            ",Mkt-RF,SMB,HML,RF\r\n"
            "192607,   2.89,  -2.55,  -2.39,   0.22\r\n"
            "192608,   2.64,  -1.14,   3.81,   0.25\r\n"
            "\r\n"
            "annual section ignored\r\n"
        )
        df = _parse_ff_csv(text)
        assert list(df.columns) == ["Mkt-RF", "SMB", "HML", "RF"]
        assert len(df) == 2
        assert df.iloc[0, 0] == pytest.approx(2.89)
        assert df.index[0].year == 1926
        assert df.index[0].month == 7

    def test_parses_momentum_format(self):
        """Il momentum CSV ha header con UNA sola colonna: ',Mom'."""
        text = (
            "metadata line\r\n"
            "\r\n"
            ",Mom\r\n"
            "192701,   0.57\r\n"
            "192702,  -1.50\r\n"
            "\r\n"
        )
        df = _parse_ff_csv(text)
        assert list(df.columns) == ["Mom"]
        assert len(df) == 2

    def test_stops_at_annual_section(self):
        text = (
            ",Mkt-RF,SMB,HML,RF\r\n"
            "192607, 2.89, -2.55, -2.39, 0.22\r\n"
            "1927,  29.47, -2.44, -3.72, 3.04\r\n"  # annual = 4-digit year
        )
        df = _parse_ff_csv(text)
        # Solo la riga monthly, non quella annual
        assert len(df) == 1

    def test_no_header_raises(self):
        with pytest.raises(ValueError):
            _parse_ff_csv("solo metadata\nniente colonne\n")


class TestFactorKeys:
    def test_default_factor_keys(self):
        keys = factor_keys()
        assert "Mkt-RF" in keys
        assert "SMB" in keys
        assert "HML" in keys
        assert "Mom" in keys
