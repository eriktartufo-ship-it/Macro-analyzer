"""Seed script per popolare asset_regime_performance con dati storici hardcoded.

Basato su ricerca accademica: Bridgewater All Weather, AQR, Ilmanen (Expected Returns).
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.services.scoring.engine import ASSET_REGIME_DATA


def get_seed_records() -> list[dict]:
    """Genera i record per il seed della tabella asset_regime_performance."""
    records = []
    for asset_class, regimes in ASSET_REGIME_DATA.items():
        for regime, data in regimes.items():
            records.append({
                "asset_class": asset_class,
                "regime": regime,
                "hit_rate": data["hit_rate"],
                "avg_return_12m": data["avg_return"],
                "volatility_in_regime": data["vol"],
                "sharpe_in_regime": data["sharpe"],
            })
    return records


def seed_database():
    """Inserisce i dati seed nel database PostgreSQL."""
    from sqlalchemy import create_engine
    from sqlalchemy.orm import Session

    from app.config import settings
    from app.database import Base
    from app.models import AssetRegimePerformance

    engine = create_engine(settings.database_url)
    Base.metadata.create_all(engine)

    records = get_seed_records()

    with Session(engine) as session:
        # Cancella dati esistenti
        session.query(AssetRegimePerformance).delete()

        for record in records:
            session.add(AssetRegimePerformance(**record))

        session.commit()
        print(f"Seed completato: {len(records)} record inseriti.")


if __name__ == "__main__":
    seed_database()
