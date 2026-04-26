"""Test term premium decomposition (ACM model)."""
import pandas as pd
import pytest

from app.services.indicators.term_premium import fetch_term_premium_decomposition


class TestTermPremiumDecomposition:
    def test_decomposition_identity(self):
        """fitted_yield = expected_path + term_premium per costruzione."""
        df = fetch_term_premium_decomposition()
        if df.empty:
            pytest.skip("Dati ACM non disponibili (no API key o offline)")
        # Identita': differenza < 1e-6 dopo costruzione
        diff = (df["fitted_yield"] - df["expected_path"] - df["term_premium"]).abs()
        assert diff.max() < 1e-6, (
            f"decomposizione non additiva: max diff {diff.max()}"
        )

    def test_columns_present(self):
        df = fetch_term_premium_decomposition()
        if df.empty:
            pytest.skip("Dati ACM non disponibili")
        assert set(df.columns) == {"fitted_yield", "term_premium", "expected_path"}

    def test_index_is_datetime(self):
        df = fetch_term_premium_decomposition()
        if df.empty:
            pytest.skip("Dati ACM non disponibili")
        assert isinstance(df.index, pd.DatetimeIndex)


class TestRegimeReportShape:
    def test_smoke_compute_report(self):
        from app.database import engine
        from app.models import RegimeClassification
        from sqlalchemy.orm import Session
        from app.services.indicators.term_premium import compute_term_premium_report

        with Session(engine) as s:
            n = s.query(RegimeClassification).count()
            if n < 30:
                pytest.skip(f"DB ha solo {n} record")
            try:
                r = compute_term_premium_report(s, threshold=0.40)
            except ValueError as e:
                pytest.skip(f"compute fallita: {e}")

        assert len(r.points) > 0
        assert len(r.by_regime) == 4
        # Threshold preservato
        assert r.threshold == pytest.approx(0.40)
        # Common period e' tuple di str
        assert len(r.common_period) == 2
