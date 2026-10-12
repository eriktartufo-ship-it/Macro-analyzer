"""Area euro ed energia (S50, 2026-10-12): parsing BCE, indicatori derivati, degrado se una fonte cade."""

import numpy as np
import pandas as pd
import pytest

from app.services.indicators.europe_energy import (
    europe_energy_indicators,
    parse_ecb_csv,
    percentile_of_last,
)

CSV_BCE = """KEY,FREQ,REF_AREA,ADJUSTMENT,ICP_ITEM,DATA_PROVIDER,ICP_SUFFIX,TIME_PERIOD,OBS_VALUE
HICP.M.U2.N.000000.4D0.ANR,M,U2,N,000000,4D0,ANR,2026-08,3.2
HICP.M.U2.N.000000.4D0.ANR,M,U2,N,000000,4D0,ANR,2026-09,3.8
"""

CSV_BCE_GIORNALIERA = """KEY,FREQ,TIME_PERIOD,OBS_VALUE
FM.D.U2.EUR.4F.KR.DFR.LEV,D,2026-10-11,2.5
FM.D.U2.EUR.4F.KR.DFR.LEV,D,2026-10-12,2.5
FM.D.U2.EUR.4F.KR.DFR.LEV,D,2026-10-13,
"""


def _mensile(valori, fine="2026-09-01"):
    idx = pd.date_range(end=fine, periods=len(valori), freq="MS")
    return pd.Series(valori, index=idx, dtype=float)


def _giornaliera(inizio, fine, f):
    idx = pd.bdate_range(inizio, fine)
    return pd.Series([f(i) for i in range(len(idx))], index=idx, dtype=float)


def _serie_complete():
    return {
        "ea_hicp": _mensile([2.0] * 6 + [2.5, 2.8, 3.0, 3.1, 3.2, 3.8]),
        "ea_core_hicp": _mensile([2.5] * 12),
        "ea_energy_hicp": _mensile([0.0] * 11 + [18.8]),
        "ea_services_hicp": _mensile([3.2] * 12),
        "ecb_deposit_rate": _giornaliera("2025-10-01", "2026-10-12", lambda i: 2.5),
        "ea_unemployment": _mensile([6.0] * 12 + [6.5], fine="2026-08-01"),
        "ea_aaa_10y": _giornaliera("2025-01-01", "2026-10-09", lambda i: 3.5),
        "ea_aaa_2y": _giornaliera("2025-01-01", "2026-10-09", lambda i: 2.6),
        "ea_all_10y": _giornaliera("2025-01-01", "2026-10-09", lambda i: 4.1),
        "ttf": _giornaliera("2025-01-01", "2026-10-12", lambda i: 30.0 + i * 0.1),
        "brent_future": _giornaliera("2025-01-01", "2026-10-12", lambda i: 100.0),
        "brent_spot": _giornaliera("2024-01-01", "2026-10-06", lambda i: 60.0 if i < 470 else 120.0),
    }


class TestParsingBce:
    def test_csv_mensile_diventa_serie_datata(self):
        s = parse_ecb_csv(CSV_BCE)
        assert s.index[-1] == pd.Timestamp("2026-09-01")
        assert s.iloc[-1] == pytest.approx(3.8)

    def test_valori_vuoti_saltati(self):
        s = parse_ecb_csv(CSV_BCE_GIORNALIERA)
        assert len(s) == 2
        assert s.index[-1] == pd.Timestamp("2026-10-12")


class TestIndicatori:
    def test_inflazione_e_tasso_reale(self):
        ind = europe_energy_indicators(_serie_complete())
        assert ind["ea_hicp_yoy"] == pytest.approx(3.8)
        assert ind["ea_energy_hicp_yoy"] == pytest.approx(18.8)
        assert ind["ecb_deposit_rate"] == pytest.approx(2.5)
        assert ind["ea_real_rate"] == pytest.approx(2.5 - 3.8)
        # accelerazione per DATA: settembre contro marzo
        assert ind["ea_hicp_change_6m"] == pytest.approx(3.8 - 2.0)

    def test_curva_e_spread_sovrano(self):
        ind = europe_energy_indicators(_serie_complete())
        assert ind["ea_aaa_slope_2_10"] == pytest.approx(3.5 - 2.6)
        assert ind["ea_sovereign_spread_10y"] == pytest.approx(4.1 - 3.5)

    def test_disoccupazione_variazione_12_mesi(self):
        ind = europe_energy_indicators(_serie_complete())
        assert ind["ea_unemployment"] == pytest.approx(6.5)
        assert ind["ea_unemployment_change_12m"] == pytest.approx(0.5)

    def test_brent_variazione_e_premio_del_fisico(self):
        ind = europe_energy_indicators(_serie_complete())
        assert ind["brent_spot"] == pytest.approx(120.0)
        assert ind["brent_yoy"] == pytest.approx(100.0)  # 60 -> 120
        assert ind["oil_yoy"] == ind["brent_yoy"]
        # premio = spot meno future ALLA STESSA DATA (lo spot arriva in ritardo)
        assert ind["brent_physical_premium"] == pytest.approx(20.0)

    def test_ttf_variazione_per_data(self):
        s = _serie_complete()
        ind = europe_energy_indicators(s)
        ttf = s["ttf"]
        base = ttf[ttf.index <= ttf.index[-1] - pd.DateOffset(months=12)].iloc[-1]
        assert ind["ttf_yoy"] == pytest.approx((ttf.iloc[-1] / base - 1) * 100)

    def test_differenziale_tassi_reali_usa_area_euro(self):
        ind = europe_energy_indicators(_serie_complete(), us={"fed_funds_rate": 3.88, "cpi_yoy": 3.35})
        assert ind["us_ea_real_rate_diff"] == pytest.approx((3.88 - 3.35) - (2.5 - 3.8))

    def test_una_fonte_giu_toglie_solo_le_sue_chiavi(self):
        s = _serie_complete()
        del s["ttf"]
        s["ea_all_10y"] = pd.Series(dtype=float)
        ind = europe_energy_indicators(s)
        assert "ttf_yoy" not in ind and "ttf_price" not in ind
        assert "ea_sovereign_spread_10y" not in ind
        assert ind["ea_hicp_yoy"] == pytest.approx(3.8)

    def test_niente_dati_niente_chiavi_e_nessuna_eccezione(self):
        assert europe_energy_indicators({}) == {}


class TestPercentile:
    def test_percentile_dell_ultimo_valore(self):
        s = pd.Series(np.arange(1, 101, dtype=float))
        assert percentile_of_last(s) == pytest.approx(100.0)
        # quota della storia <= ultimo valore
        assert percentile_of_last(pd.Series([1.0, 2.0, 3.0, 10.0])) == pytest.approx(100.0)
        assert percentile_of_last(pd.Series([10.0, 2.0, 3.0, 1.0])) == pytest.approx(25.0)

    def test_spread_sovrano_porta_il_suo_percentile_storico(self):
        ind = europe_energy_indicators(_serie_complete())
        assert 0.0 <= ind["ea_sovereign_spread_pctile"] <= 100.0


class TestCablaggio:
    def test_il_refresh_giornaliero_include_gli_indicatori_europei(self, monkeypatch):
        from app.scheduler import jobs
        from app.services.indicators import europe_energy

        monkeypatch.setattr(europe_energy, "live_europe_energy_indicators",
                            lambda us=None: {"ea_hicp_yoy": 3.8, "oil_yoy": 75.0})

        class FetcherMuto:
            def fetch_and_transform(self, *a, **k):
                raise RuntimeError("offline")

            def fetch_series(self, *a, **k):
                raise RuntimeError("offline")

        ind = jobs._prepare_indicators({}, FetcherMuto())
        assert ind["ea_hicp_yoy"] == pytest.approx(3.8)
        assert ind["oil_yoy"] == pytest.approx(75.0)


class TestPayloadPannello:
    def test_gruppi_etichette_e_valori(self):
        from app.services.indicators.europe_energy import europe_energy_payload

        p = europe_energy_payload({"ea_hicp_yoy": 3.8, "ttf_yoy": 158.1, "ea_sovereign_spread_pctile": 56.6,
                                   "cpi_yoy": 3.35}, "2026-10-12")
        assert p["as_of"] == "2026-10-12"
        ids = [g["id"] for g in p["groups"]]
        assert ids == ["inflazione", "tassi", "energia", "lavoro"]
        voci = {i["key"]: i for g in p["groups"] for i in g["items"]}
        assert voci["ea_hicp_yoy"]["value"] == pytest.approx(3.8)
        assert voci["ecb_deposit_rate"]["value"] is None  # mancante = None, non sparisce
        assert "cpi_yoy" not in voci  # solo le chiavi del blocco
        assert p["sovereign_spread_pctile"] == pytest.approx(56.6)
        assert p["available"] is True

    def test_nessun_dato_europeo_available_false(self):
        from app.services.indicators.europe_energy import europe_energy_payload

        assert europe_energy_payload({"cpi_yoy": 3.35}, "2026-10-12")["available"] is False
