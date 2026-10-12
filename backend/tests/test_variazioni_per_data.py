"""Le variazioni si cercano per DATA, non contando le righe (bug trovato il 2026-10-12).

Tre forme dello stesso difetto, tutte in produzione quel giorno:
- ottobre 2025 manca nelle serie BLS (shutdown): 12 righe indietro da agosto 2026 = luglio 2025,
  quindi una variazione su 13 mesi. Sul sito il CPI era 3,71 invece di 3,35;
- FDHBFIN (detentori esteri di Treasury) e' trimestrale ma dichiarata mensile: 12 righe = 3 anni;
- le serie settimanali (sussidi di disoccupazione) avevano roc_3m = 63 righe = 63 settimane,
  mentre la ricostruzione storica usa 13 settimane: il classificatore vedeva dal vivo una cosa
  diversa da quella su cui e' tarato.

Ogni test qui sotto fallisce se si torna al conteggio per posizione.
"""

from datetime import date

import numpy as np
import pandas as pd
import pytest

from app.services.indicators.transforms import (
    calculate_roc,
    calculate_roc_by_date,
    roc_between,
    standard_transforms,
)

CRESCITA = 1.01  # +1% al mese: 12 mesi = +12,68%, 13 mesi = +13,81%
YOY_12 = (CRESCITA**12 - 1) * 100
YOY_13 = (CRESCITA**13 - 1) * 100


def _mensile(fine="2026-10-01", buco=("2025-10-01",)):
    idx = pd.date_range("2023-01-01", fine, freq="MS")
    s = pd.Series(100 * CRESCITA ** np.arange(len(idx)), index=idx)
    return s.drop([pd.Timestamp(b) for b in buco])


class TestRocPerData:
    def test_mese_mancante_non_fa_slittare_il_confronto(self):
        s = _mensile()
        roc = calculate_roc_by_date(s, 12)
        assert roc[pd.Timestamp("2026-08-01")] == pytest.approx(YOY_12)
        # il vecchio calcolo per posizione dava 13 mesi: e' il caso vero del CPI
        assert calculate_roc(s, 12)[pd.Timestamp("2026-08-01")] == pytest.approx(YOY_13)

    def test_mese_di_confronto_mancante_da_vuoto_non_un_altro_mese(self):
        roc = calculate_roc_by_date(_mensile(), 12)
        assert np.isnan(roc[pd.Timestamp("2026-10-01")])  # base ottobre 2025 non esiste
        assert roc[pd.Timestamp("2025-11-01")] == pytest.approx(YOY_12)

    def test_serie_senza_buchi_identica_al_calcolo_di_prima(self):
        s = _mensile(buco=())
        nuovo = calculate_roc_by_date(s, 12)
        vecchio = calculate_roc(s, 12)
        pd.testing.assert_series_equal(nuovo, vecchio, check_names=False)

    def test_trimestrale_dichiarata_mensile_12_mesi_sono_4_trimestri(self):
        idx = pd.date_range("2020-01-01", "2025-10-01", freq="QS")
        s = pd.Series(100 * 1.02 ** np.arange(len(idx)), index=idx)
        roc = calculate_roc_by_date(s, 12)
        assert roc.iloc[-1] == pytest.approx((1.02**4 - 1) * 100)

    def test_settimanale_3_mesi_sono_13_settimane_non_63(self):
        idx = pd.date_range("2024-01-06", "2026-07-04", freq="7D")
        s = pd.Series(np.arange(len(idx), dtype=float) + 100, index=idx)
        roc = calculate_roc_by_date(s, 3)
        ultimo = s.iloc[-1]
        base = s[s.index <= s.index[-1] - pd.DateOffset(months=3)].iloc[-1]
        assert roc.iloc[-1] == pytest.approx((ultimo / base - 1) * 100)
        assert (s.index[-1] - s[s.index <= s.index[-1] - pd.DateOffset(months=3)].index[-1]).days <= 98

    def test_giornaliera_bersaglio_nel_weekend_prende_il_venerdi(self):
        idx = pd.bdate_range("2025-01-01", "2026-10-09")
        s = pd.Series(np.arange(len(idx), dtype=float) + 1, index=idx)
        # 2026-10-05 (lunedi') - 12 mesi = 2025-10-05 (domenica) -> venerdi' 2025-10-03
        v = roc_between(s, pd.Timestamp("2026-10-05"), 12)
        assert v == pytest.approx((s[pd.Timestamp("2026-10-05")] / s[pd.Timestamp("2025-10-03")] - 1) * 100)

    def test_roc_between_none_se_il_passato_manca(self):
        assert roc_between(_mensile(), pd.Timestamp("2026-10-01"), 12) is None
        assert roc_between(_mensile(), pd.Timestamp("2026-08-01"), 12) == pytest.approx(YOY_12)


class TestTrasformazioniStandard:
    def test_settimanale_roc_3m_e_13_settimane(self):
        idx = pd.date_range("2020-01-04", "2026-07-04", freq="7D")
        s = pd.Series(np.arange(len(idx), dtype=float) + 100, index=idx)
        out = standard_transforms(s, "weekly")
        assert out["roc_3m"].iloc[-1] == pytest.approx(calculate_roc_by_date(s, 3).iloc[-1])
        assert out["roc_3m"].iloc[-1] != pytest.approx(calculate_roc(s, 63).iloc[-1])

    def test_mensile_con_buco(self):
        out = standard_transforms(_mensile(fine="2026-08-01"), "monthly")
        assert out["roc_12m"].iloc[-1] == pytest.approx(YOY_12)

    def test_annuale_mantiene_la_variazione_di_un_anno_su_tutti_gli_orizzonti(self):
        idx = pd.date_range("2015-01-01", "2025-01-01", freq="YS")
        s = pd.Series(100 * 1.05 ** np.arange(len(idx)), index=idx)
        out = standard_transforms(s, "annual")
        for k in ("roc_3m", "roc_6m", "roc_12m"):
            assert out[k].iloc[-1] == pytest.approx(5.0)

    def test_fetcher_live_usa_le_trasformazioni_per_data(self, monkeypatch):
        from app.services.indicators.fetcher import FredFetcher

        f = FredFetcher.__new__(FredFetcher)
        f._transform_cache = {}
        monkeypatch.setattr(f, "fetch_series", lambda name, *a, **k: _mensile(fine="2026-08-01"))
        out = f.fetch_and_transform("cpi")
        assert out["roc_12m"].dropna().iloc[-1] == pytest.approx(YOY_12)


class TestPercorsiStorici:
    def test_backfill_regime_cpi_yoy_su_12_mesi(self):
        from app.services.regime.backfill import _build_indicators_as_of

        ind = _build_indicators_as_of({"cpi": _mensile(fine="2026-08-01")}, date(2026, 8, 15))
        assert ind["cpi_yoy"] == pytest.approx(YOY_12)

    def test_minimo_yoy_ultimi_6_mesi_non_usa_13_mesi(self):
        from app.services.regime.backfill import _enrich_with_momentum_derivatives

        ind: dict = {}
        _enrich_with_momentum_derivatives(ind, {"cpi": _mensile(fine="2026-08-01")}, pd.Timestamp("2026-08-15"))
        assert ind["cpi_yoy_min_last_6m"] == pytest.approx(YOY_12)

    def test_m2_yoy_con_buco(self):
        from app.services.regime.backfill import _enrich_with_momentum_derivatives

        ind: dict = {}
        _enrich_with_momentum_derivatives(ind, {"m2": _mensile(fine="2026-08-01")}, pd.Timestamp("2026-08-15"))
        assert ind["m2_yoy"] == pytest.approx(YOY_12)

    def test_dedollar_backfill_cpi_yoy(self):
        from app.services.dedollarization.backfill import _macro_indicators_as_of

        out = _macro_indicators_as_of({"cpi": _mensile(fine="2026-08-01")}, pd.Timestamp("2026-08-15"))
        assert out["cpi_yoy"] == pytest.approx(YOY_12)

    def test_dedollar_backfill_fetcher_usa_le_trasformazioni_per_data(self):
        from app.services.dedollarization.backfill import AsOfFetcher

        f = AsOfFetcher({"cpi": _mensile(fine="2026-08-01")}, date(2026, 8, 15))
        assert f.fetch_and_transform("cpi")["roc_12m"].dropna().iloc[-1] == pytest.approx(YOY_12)

    def test_soglie_adattive_cpi_yoy_dal_db(self, monkeypatch):
        from app.services.indicators import adaptive_thresholds as at

        monkeypatch.setattr(at, "load_series_from_db", lambda db, key: _mensile(fine="2026-08-01"))
        yoy = at.load_cpi_yoy_from_db(None)
        assert yoy.iloc[-1] == pytest.approx(YOY_12)
