"""Tier 6.6 — Live calibration snapshot persistence.

Tabella che memorizza snapshot mensili della calibration `asset_regime_data`
calcolata via Bayesian shrinkage (`services/scoring/calibration.py`).

Pattern: ogni run scheduler mensile crea un nuovo record con:
- `version`: timestamp ISO del run
- `n_classifications`: records storici usati
- `payload`: JSON completo (asset_regime_data + diagnostics + params)
- `is_active`: bool, solo l'ultimo è True (query last active)

Sostituisce `seed/calibrated_asset_regime.json` (file statico) con
DB versionato, query-able + audit trail.
"""
from datetime import date, datetime

from sqlalchemy import Boolean, Date, DateTime, Float, Index, Integer, String, Text
from sqlalchemy.orm import Mapped, mapped_column

from app.database import Base


class AssetRegimeCalibration(Base):
    """Snapshot calibration `asset_regime_data` versionato."""

    __tablename__ = "asset_regime_calibrations"

    id: Mapped[int] = mapped_column(primary_key=True)
    calibrated_on: Mapped[date] = mapped_column(
        Date, nullable=False, comment="Data calibration (giornata di run)"
    )
    version: Mapped[str] = mapped_column(
        String(50), nullable=False, comment="ISO timestamp del run (audit trail)"
    )
    n_classifications: Mapped[int] = mapped_column(
        Integer, nullable=False, default=0,
        comment="Records storici RegimeClassification usati per il fit"
    )
    n_assets: Mapped[int] = mapped_column(
        Integer, nullable=False, default=0, comment="Asset class coperti"
    )
    avg_brier_delta: Mapped[float | None] = mapped_column(
        Float, nullable=True,
        comment="Delta Brier vs hardcoded prior (negativo = miglioramento)"
    )
    is_active: Mapped[bool] = mapped_column(
        Boolean, nullable=False, default=True,
        comment="True solo per l'ultimo snapshot (lookup veloce)"
    )
    payload: Mapped[str] = mapped_column(
        Text, nullable=False,
        comment="JSON: {asset_regime_data, diagnostics, params}"
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime, default=datetime.utcnow
    )

    __table_args__ = (
        Index("ix_asset_regime_calibrations_active", "is_active"),
        Index("ix_asset_regime_calibrations_date", "calibrated_on"),
    )
