from sqlalchemy import Float, Index, String
from sqlalchemy.orm import Mapped, mapped_column

from app.database import Base


class AssetRegimePerformance(Base):
    """Lookup table performance storiche per asset class x regime.

    Dati hardcoded basati su letteratura (Bridgewater, AQR, ricerca accademica).
    """

    __tablename__ = "asset_regime_performance"

    id: Mapped[int] = mapped_column(primary_key=True)
    asset_class: Mapped[str] = mapped_column(
        String(50), nullable=False,
        comment="Asset class (es. us_equities_growth, gold, bitcoin)"
    )
    regime: Mapped[str] = mapped_column(
        String(30), nullable=False,
        comment="Regime macro"
    )
    hit_rate: Mapped[float] = mapped_column(
        Float, nullable=False,
        comment="Percentuale periodi storici con outperformance (0-1)"
    )
    avg_return_12m: Mapped[float] = mapped_column(
        Float, nullable=False,
        comment="Rendimento medio annualizzato"
    )
    volatility_in_regime: Mapped[float] = mapped_column(
        Float, nullable=False,
        comment="Deviazione standard nel regime"
    )
    sharpe_in_regime: Mapped[float] = mapped_column(
        Float, nullable=False,
        comment="Sharpe ratio nel regime"
    )

    __table_args__ = (
        Index("ix_asset_regime_perf_asset_regime", "asset_class", "regime", unique=True),
    )
