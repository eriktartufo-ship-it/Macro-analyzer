"""Test Kalman 1D filter su serie sintetiche e proprieta' matematiche."""
import numpy as np
import pandas as pd
import pytest

from app.services.indicators.kalman import (
    DEFAULT_LAMBDA,
    NOISY_INDICATORS,
    kalman_filter_1d,
)


def _series(values, start="2020-01-31", freq="ME"):
    return pd.Series(values, index=pd.date_range(start, periods=len(values), freq=freq))


class TestKalmanFilter1D:
    def test_too_short_raises(self):
        with pytest.raises(ValueError):
            kalman_filter_1d(_series([1.0, 2.0]))

    def test_filter_reduces_noise(self):
        """Su trend + noise, il filtro deve ridurre il rumore residuo (errore vs trend vero)."""
        rng = np.random.default_rng(0)
        n = 200
        trend = np.linspace(0, 10, n)
        noise = rng.normal(0, 1.0, n)
        y = trend + noise
        r = kalman_filter_1d(_series(y), lam=10.0)
        # Errore residuo = serie - trend vero. Il filtered deve avere errore minore.
        err_raw = float(((y - trend) ** 2).mean())
        err_filt = float(((r.filtered.values - trend) ** 2).mean())
        assert err_filt < err_raw * 0.7, (
            f"filtered MSE residuo {err_filt:.3f} non significativamente < raw {err_raw:.3f}"
        )

    def test_smoothed_better_than_filtered_for_recovering_trend(self):
        """Smoothed (RTS) deve essere piu' vicino al trend vero del filtered."""
        rng = np.random.default_rng(42)
        n = 100
        trend = np.linspace(0, 5, n)
        y = trend + rng.normal(0, 1.5, n)
        r = kalman_filter_1d(_series(y), lam=10.0)
        mse_filt = float(((r.filtered.values - trend) ** 2).mean())
        mse_smooth = float(((r.smoothed.values - trend) ** 2).mean())
        assert mse_smooth <= mse_filt, (
            f"smoothed MSE {mse_smooth} > filtered MSE {mse_filt}"
        )

    def test_high_lambda_smoothes_more(self):
        """Lambda alto -> filtered piu' lontano dalle osservazioni puntuali."""
        rng = np.random.default_rng(1)
        y = rng.normal(0, 1, 100)
        r_low = kalman_filter_1d(_series(y), lam=1.0)
        r_high = kalman_filter_1d(_series(y), lam=50.0)
        # Distanza media tra filtered e raw deve crescere con lam
        diff_low = float((r_low.filtered.values - y).std())
        diff_high = float((r_high.filtered.values - y).std())
        assert diff_high > diff_low, "lam=50 dovrebbe smussare di piu' di lam=1"

    def test_lambda_zero_almost_pass_through(self):
        """Lambda molto basso -> filtered ~ raw (Kalman segue le osservazioni)."""
        rng = np.random.default_rng(2)
        y = rng.normal(0, 1, 50)
        r = kalman_filter_1d(_series(y), lam=0.5)
        # Correlazione alta tra raw e filtered
        corr = float(np.corrcoef(y, r.filtered.values)[0, 1])
        assert corr > 0.85, f"lam=0.5: corr {corr} troppo bassa"

    def test_constant_series_returns_constant(self):
        y = [3.0] * 30
        r = kalman_filter_1d(_series(y))
        assert (r.filtered.round(6) == 3.0).all()
        assert (r.smoothed.round(6) == 3.0).all()

    def test_index_preserved(self):
        rng = np.random.default_rng(0)
        s = _series(rng.normal(0, 1, 30))
        r = kalman_filter_1d(s)
        assert (r.filtered.index == s.index).all()
        assert (r.smoothed.index == s.index).all()
        assert (r.raw.index == s.index).all()


class TestNoisyIndicatorsRegistry:
    def test_registry_non_empty(self):
        assert len(NOISY_INDICATORS) >= 4
        for k, desc in NOISY_INDICATORS.items():
            assert isinstance(k, str)
            assert isinstance(desc, str)
            assert len(desc) > 5

    def test_default_lambda_in_sane_range(self):
        # 5 < default < 30 = bilanciato per macro mensili
        assert 5.0 <= DEFAULT_LAMBDA <= 30.0
