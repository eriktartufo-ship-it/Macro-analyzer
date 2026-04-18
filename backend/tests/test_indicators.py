"""Test TDD per calcolo indicatori: ROC, Z-score, trasformazioni."""

import numpy as np
import pandas as pd
import pytest

from app.services.indicators.transforms import calculate_roc, calculate_zscore, calculate_yoy


class TestRateOfChange:
    """Verifica matematica del calcolo Rate of Change."""

    def test_roc_basic_calculation(self):
        """ROC = (current - past) / past * 100."""
        series = pd.Series([100, 105, 110, 120, 130])
        roc = calculate_roc(series, periods=1)

        assert roc.iloc[1] == pytest.approx(5.0, rel=1e-5)
        assert roc.iloc[2] == pytest.approx(4.7619, rel=1e-3)

    def test_roc_3_periods(self):
        """ROC a 3 periodi su valori noti."""
        series = pd.Series([100, 102, 104, 110, 115, 120])
        roc = calculate_roc(series, periods=3)

        # roc[3] = (110 - 100) / 100 * 100 = 10.0
        assert roc.iloc[3] == pytest.approx(10.0, rel=1e-5)
        # roc[4] = (115 - 102) / 102 * 100 = 12.745
        assert roc.iloc[4] == pytest.approx(12.745, rel=1e-2)

    def test_roc_first_periods_are_nan(self):
        """I primi N valori devono essere NaN."""
        series = pd.Series([100, 105, 110, 115, 120])
        roc = calculate_roc(series, periods=3)

        assert pd.isna(roc.iloc[0])
        assert pd.isna(roc.iloc[1])
        assert pd.isna(roc.iloc[2])
        assert not pd.isna(roc.iloc[3])

    def test_roc_negative_values(self):
        """ROC con valori in calo."""
        series = pd.Series([100, 95, 90, 85])
        roc = calculate_roc(series, periods=1)

        assert roc.iloc[1] == pytest.approx(-5.0, rel=1e-5)
        assert roc.iloc[2] == pytest.approx(-5.2632, rel=1e-3)

    def test_roc_preserves_index(self):
        """L'indice della serie deve essere preservato."""
        dates = pd.date_range("2020-01-01", periods=5, freq="MS")
        series = pd.Series([100, 105, 110, 115, 120], index=dates)
        roc = calculate_roc(series, periods=1)

        assert roc.index.equals(dates)


class TestZScore:
    """Verifica normalizzazione Z-score con rolling window."""

    def test_zscore_known_values(self):
        """Z-score con valori dove sappiamo il risultato."""
        # Serie costante = zscore 0
        series = pd.Series([10.0] * 20)
        zs = calculate_zscore(series, window=10)

        # Dopo il warm-up, z-score di un valore costante = 0 (o NaN per std=0)
        # Con std=0, definiamo zscore=0
        valid = zs.dropna()
        assert all(v == 0.0 or pd.isna(v) for v in valid)

    def test_zscore_positive_deviation(self):
        """Valore sopra la media deve avere zscore > 0."""
        # 10 valori a 100, poi uno spike a 120
        values = [100.0] * 10 + [120.0]
        series = pd.Series(values)
        zs = calculate_zscore(series, window=10)

        # L'ultimo valore (120) e' ben sopra la media rolling
        assert zs.iloc[-1] > 0

    def test_zscore_negative_deviation(self):
        """Valore sotto la media deve avere zscore < 0."""
        values = [100.0] * 10 + [80.0]
        series = pd.Series(values)
        zs = calculate_zscore(series, window=10)

        assert zs.iloc[-1] < 0

    def test_zscore_rolling_window_size(self):
        """I primi (window-1) valori devono essere NaN."""
        series = pd.Series(range(20), dtype=float)
        zs = calculate_zscore(series, window=12)

        # Primi 11 valori NaN (window=12, serve almeno 12 punti)
        assert pd.isna(zs.iloc[0])
        assert not pd.isna(zs.iloc[11])

    def test_zscore_approximate_standard_normal(self):
        """Con dati normali, zscore deve avere media ~0 e std ~1."""
        np.random.seed(42)
        series = pd.Series(np.random.normal(100, 10, 1000))
        zs = calculate_zscore(series, window=252)

        valid = zs.dropna()
        assert abs(valid.mean()) < 0.3
        assert abs(valid.std() - 1.0) < 0.3


class TestYearOverYear:
    """Test calcolo Year-over-Year."""

    def test_yoy_monthly_data(self):
        """YoY su dati mensili (12 periodi indietro)."""
        # 24 mesi: primo anno a 100, secondo anno a 110
        values = [100.0] * 12 + [110.0] * 12
        series = pd.Series(values)
        yoy = calculate_yoy(series, periods=12)

        # Dopo 12 mesi, yoy = (110-100)/100*100 = 10%
        assert yoy.iloc[12] == pytest.approx(10.0, rel=1e-5)

    def test_yoy_first_year_nan(self):
        """Primo anno deve essere NaN."""
        series = pd.Series([100.0] * 24)
        yoy = calculate_yoy(series, periods=12)

        for i in range(12):
            assert pd.isna(yoy.iloc[i])
