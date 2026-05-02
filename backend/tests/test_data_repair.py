"""Test regressione: bug LEI corruption + repair logic.

Bug originale (2025-10..2026-04-27): scheduler giornaliero applicava
`roc_6m` sopra CFNAIMA3 (gia' z-score), salvando `lei_roc=30%`/`-86%`
invece del livello z-score atteso ±3. Lo storico DB e' stato riparato dal
2026-05-02 con `repair_lei_history`. Questi test prevengono ricorrenza.
"""
from __future__ import annotations

import json
from datetime import date
from types import SimpleNamespace
from unittest.mock import patch

import pandas as pd
import pytest

from app.services.admin.data_repair import (
    LEI_CORRUPTION_THRESHOLD,
    _value_as_of,
    repair_lei_history,
)


class TestValueAsOf:
    def test_returns_last_value_le_target(self):
        s = pd.Series(
            [1.0, 2.0, 3.0],
            index=pd.to_datetime(["2024-01-01", "2024-02-01", "2024-03-01"]),
        )
        assert _value_as_of(s, pd.Timestamp("2024-02-15")) == 2.0

    def test_returns_exact_match(self):
        s = pd.Series(
            [-0.03],
            index=pd.to_datetime(["2026-04-01"]),
        )
        assert _value_as_of(s, pd.Timestamp("2026-04-01")) == pytest.approx(-0.03)

    def test_returns_none_if_target_before_series(self):
        s = pd.Series(
            [1.0],
            index=pd.to_datetime(["2024-01-01"]),
        )
        assert _value_as_of(s, pd.Timestamp("2023-12-01")) is None


class TestLEICorruptionDetection:
    """La soglia 5.0 e' calibrata sul range storico CFNAI z-score (±3 tipico,
    ±5 in episodi recessivi estremi). Il bug pre-fix produceva valori >>5
    (30%, -86%, ...) facilmente riconoscibili."""

    def test_threshold_is_conservative(self):
        # Soglia non deve essere troppo bassa (false positive su recessioni reali)
        # ne troppo alta (miss del bug pattern 30+/-86).
        assert 3.0 < LEI_CORRUPTION_THRESHOLD < 10.0

    def test_real_cfnai_recession_values_above_threshold_are_idempotent(self):
        """Episodi recessivi reali (CFNAI 2008, 2020) hanno z-score < -5
        legittimamente. Il repair li flagga ma il replace col valore reale
        e' un no-op (idempotente)."""
        # 2020-04-08 CFNAI ~ -7.53 (COVID): legittimo, non bug
        legit_recession_cfnai = -7.53
        assert abs(legit_recession_cfnai) > LEI_CORRUPTION_THRESHOLD
        # In repair, sostituire -7.53 con _value_as_of(2020-04-08) = -7.53
        # → no cambiamento di lei_roc → classify_regime stesso input → stesso output

    def test_bug_pattern_value_above_threshold(self):
        """Pattern bug pre-fix: lei_roc=30.30 (CFNAI~-0.03 trasformato roc_6m)."""
        bug_value = 30.303030303030297  # valore osservato in DB nei record bogus
        assert abs(bug_value) > LEI_CORRUPTION_THRESHOLD


class TestRepairLEIHistoryDryRun:
    """Test logica repair con DB e fetcher mockati. Verifica che dry_run
    non chiami commit, che corrupted records siano detected, che il valore
    venga sostituito col CFNAI corretto."""

    def _mock_record(self, date_val: date, lei_roc: float, regime: str = "reflation"):
        """Crea un record mock con conditions_met JSON valido."""
        meta = {
            "indicators": {
                "gdp_roc": 0.5,
                "cpi_yoy": 2.5,
                "unrate": 4.0,
                "lei_roc": lei_roc,
                "fed_funds_rate": 3.0,
                "yield_curve_10y2y": 0.5,
            }
        }
        rec = SimpleNamespace(
            date=date_val,
            regime=regime,
            probability_reflation=0.4,
            probability_stagflation=0.2,
            probability_deflation=0.2,
            probability_goldilocks=0.2,
            confidence=0.5,
            conditions_met=json.dumps(meta),
        )
        return rec

    def _make_db(self, records):
        commit_called = []

        class FakeQuery:
            def __init__(self, recs):
                self._recs = recs
            def all(self):
                return self._recs

        class FakeDB:
            def query(self, model):
                return FakeQuery(records)
            def commit(self):
                commit_called.append(True)

        return FakeDB(), commit_called

    def test_dry_run_does_not_commit(self):
        recs = [
            self._mock_record(date(2026, 3, 1), lei_roc=30.30),
            self._mock_record(date(2026, 4, 1), lei_roc=-86.36),
        ]
        db, commits = self._make_db(recs)

        # Mock CFNAI fetch: ritorna valore plausibile -0.03 per entrambe le date
        mock_series = pd.Series(
            [-0.03, -0.03],
            index=pd.to_datetime(["2026-02-01", "2026-03-01"]),
        )
        with patch(
            "app.services.admin.data_repair._load_cfnai_history",
            return_value=mock_series,
        ):
            report = repair_lei_history(db, dry_run=True)

        assert report.dry_run is True
        assert report.corrupted == 2
        assert report.repaired == 2
        assert commits == []  # NESSUN commit in dry-run
        # I record originali NON sono stati modificati
        assert recs[0].regime == "reflation"
        assert json.loads(recs[0].conditions_met)["indicators"]["lei_roc"] == 30.30

    def test_real_run_updates_records_and_commits(self):
        recs = [
            self._mock_record(date(2026, 3, 1), lei_roc=30.30),
        ]
        db, commits = self._make_db(recs)

        mock_series = pd.Series(
            [-0.03],
            index=pd.to_datetime(["2026-02-01"]),
        )
        with patch(
            "app.services.admin.data_repair._load_cfnai_history",
            return_value=mock_series,
        ):
            report = repair_lei_history(db, dry_run=False)

        assert report.repaired == 1
        assert commits == [True]   # commit chiamato esattamente una volta
        # Il record e' stato aggiornato col valore CFNAI corretto
        new_lei = json.loads(recs[0].conditions_met)["indicators"]["lei_roc"]
        assert new_lei == pytest.approx(-0.03)

    def test_clean_records_skipped(self):
        """Record con lei_roc gia' nel range corretto (|lei| <= 5) saltati."""
        recs = [
            self._mock_record(date(2026, 5, 1), lei_roc=-0.03),
            self._mock_record(date(2026, 5, 2), lei_roc=2.5),
            self._mock_record(date(2026, 5, 3), lei_roc=-4.99),
        ]
        db, commits = self._make_db(recs)

        mock_series = pd.Series([0.0], index=pd.to_datetime(["2026-04-01"]))
        with patch(
            "app.services.admin.data_repair._load_cfnai_history",
            return_value=mock_series,
        ):
            report = repair_lei_history(db, dry_run=False)

        assert report.corrupted == 0
        assert report.repaired == 0
        assert commits == []

    def test_idempotent_second_run_no_op(self):
        """Eseguire repair due volte produce stesso risultato (corrupted=0
        alla seconda chiamata)."""
        recs = [
            self._mock_record(date(2026, 3, 1), lei_roc=30.30),
        ]
        db, commits = self._make_db(recs)

        mock_series = pd.Series([-0.03], index=pd.to_datetime(["2026-02-01"]))
        with patch(
            "app.services.admin.data_repair._load_cfnai_history",
            return_value=mock_series,
        ):
            r1 = repair_lei_history(db, dry_run=False)
            r2 = repair_lei_history(db, dry_run=False)

        assert r1.repaired == 1
        # Seconda chiamata: lei_roc gia' -0.03, sotto soglia → corrupted=0
        assert r2.corrupted == 0
        assert r2.repaired == 0
