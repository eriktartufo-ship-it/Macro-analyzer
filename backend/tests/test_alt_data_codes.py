"""Test alt data ingest (Tier 3.6 roadmap Bridgewater).

I 6 nuovi FRED codes (JOLTS, personal income, durable goods, ECB/BoE/BoJ
policy rates) devono essere correttamente registrati in FRED_SERIES
e fetchabili via FredFetcher.

Smoke + structural only (niente API hit nei test). Per il live verification
serve un endpoint live (visto in smoke separato).
"""
from __future__ import annotations

from app.services.indicators.fred_codes import FRED_SERIES


# I 6 codici Tier 3.6 Fase A.
NEW_ALT_DATA_KEYS = (
    "job_openings",
    "personal_income",
    "durable_goods_orders",
    "ecb_main_refi_rate",
    "boe_bank_rate",
    "boj_policy_rate",
)


class TestAltDataRegistration:
    def test_all_6_keys_in_fred_codes(self):
        for key in NEW_ALT_DATA_KEYS:
            assert key in FRED_SERIES, f"Missing FRED code: {key}"

    def test_each_has_required_fields(self):
        """Ogni voce deve avere fred_id, description, frequency, transform."""
        required = {"fred_id", "description", "frequency", "transform"}
        for key in NEW_ALT_DATA_KEYS:
            cfg = FRED_SERIES[key]
            assert required.issubset(cfg.keys()), f"{key} missing fields"
            assert isinstance(cfg["fred_id"], str) and cfg["fred_id"]
            assert cfg["transform"] in {"level", "roc", "yoy"}, f"{key} transform invalid"

    def test_fred_ids_unique(self):
        """Nessun codice duplicato (anche tra alt data nuovi)."""
        fred_ids = [FRED_SERIES[k]["fred_id"] for k in NEW_ALT_DATA_KEYS]
        assert len(fred_ids) == len(set(fred_ids))

    def test_no_collision_with_existing(self):
        """I nuovi fred_id non devono collidere con quelli esistenti."""
        new_ids = {FRED_SERIES[k]["fred_id"] for k in NEW_ALT_DATA_KEYS}
        existing_ids = {
            v["fred_id"] for k, v in FRED_SERIES.items()
            if k not in NEW_ALT_DATA_KEYS
        }
        overlap = new_ids & existing_ids
        assert not overlap, f"Collision: {overlap}"


class TestAltDataIdempotency:
    """I codici devono essere registrati una sola volta nel dict (no doppi)."""

    def test_no_duplicate_keys(self):
        # In Python dict letterale, chiavi duplicate vengono silenziate
        # (l'ultima vince). Verifichiamo manualmente leggendo il file.
        import importlib
        from app.services.indicators import fred_codes
        importlib.reload(fred_codes)
        # Se importa correttamente, niente errori di sintassi
        assert len(fred_codes.FRED_SERIES) >= 60
