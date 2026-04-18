import pytest
import pandas as pd


@pytest.fixture
def sample_gdp_series():
    """Serie GDP trimestrale realistica per test."""
    dates = pd.date_range("2020-01-01", periods=20, freq="QS")
    # Simula: crescita -> COVID crash -> recovery
    values = [21.7, 21.9, 22.1, 22.3, 22.5, 19.5, 19.0, 20.5, 21.8, 22.0,
              22.4, 22.8, 23.2, 23.5, 23.8, 24.0, 24.2, 24.3, 24.1, 23.9]
    return pd.Series(values, index=dates, name="GDP")


@pytest.fixture
def sample_cpi_series():
    """Serie CPI mensile per test inflation."""
    dates = pd.date_range("2020-01-01", periods=36, freq="MS")
    # CPI crescente con accelerazione nel 2021-2022
    base = 258.0
    values = [base + i * 0.3 + (i**1.5) * 0.02 for i in range(36)]
    return pd.Series(values, index=dates, name="CPIAUCSL")


@pytest.fixture
def sample_unemployment_series():
    """Serie unemployment rate mensile."""
    dates = pd.date_range("2020-01-01", periods=24, freq="MS")
    # Spike COVID poi discesa
    values = [3.5, 3.5, 4.4, 14.7, 13.3, 11.1, 10.2, 8.4, 7.8, 6.9,
              6.7, 6.7, 6.3, 6.2, 6.0, 5.8, 5.4, 5.2, 4.6, 4.2,
              3.9, 3.8, 3.6, 3.5]
    return pd.Series(values, index=dates, name="UNRATE")


@pytest.fixture
def regime_probabilities_reflation():
    """Probabilita tipiche di un regime reflation."""
    return {
        "reflation": 0.55,
        "stagflation": 0.05,
        "deflation": 0.10,
        "goldilocks": 0.30,
    }


@pytest.fixture
def regime_probabilities_stagflation():
    """Probabilita tipiche di stagflation (2022-style)."""
    return {
        "reflation": 0.10,
        "stagflation": 0.60,
        "deflation": 0.15,
        "goldilocks": 0.15,
    }
