"""FastAPI application entry point."""

import threading
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from loguru import logger

from app.api.routes import router
from app.config import settings
from app.scheduler.jobs import start_scheduler, stop_scheduler


def _maybe_backfill_on_startup() -> None:
    """Se il DB non copre ~1 anno di storia, lancia un backfill completo.

    Gira in background thread per non bloccare lo startup del server.
    """
    try:
        from app.services.backfill import backfill_all, needs_backfill

        if not needs_backfill(min_coverage_days=300):
            logger.info("Startup: DB già popolato, skip backfill automatico")
            return

        logger.info("Startup: DB vuoto o parziale — avvio backfill 365gg in background")
        try:
            backfill_all(days=365)
            logger.info("Startup: backfill completato")
        except Exception as e:
            logger.error(f"Startup backfill fallito: {e}")
    except Exception as e:
        logger.error(f"Impossibile valutare backfill: {e}")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup e shutdown events."""
    logger.info("Avvio Macro Analyzer API...")

    # T11+: assicura che la table runtime_flag_overrides esista, poi carica
    # gli overrides salvati (sopravvivenza docker compose restart).
    try:
        from app.database import Base, engine
        from app.models import ModelSnapshot, PredictionLog, RuntimeFlagOverride  # noqa: F401
        from app.services.config_flags import load_runtime_flags_from_db

        Base.metadata.create_all(bind=engine, tables=[
            RuntimeFlagOverride.__table__,
            ModelSnapshot.__table__,
            PredictionLog.__table__,
        ])

        # Lightweight migration: add column model_version se manca (DB esistenti pre-CRITICAL#2)
        from sqlalchemy import text
        with engine.connect() as conn:
            try:
                conn.execute(text(
                    "ALTER TABLE prediction_log "
                    "ADD COLUMN IF NOT EXISTS model_version VARCHAR(64)"
                ))
                conn.commit()
            except Exception:
                pass  # already exists or not supported (SQLite)

        n = load_runtime_flags_from_db()
        if n > 0:
            logger.info(f"Loaded {n} runtime flag override(s) from DB")
    except Exception as e:
        logger.warning(f"Runtime flag persistence init skipped: {e}")

    # Portfolio Tracking (pt_*): lazy-create tabelle solo se il modulo è attivo.
    if settings.enable_portfolio_tracking:
        try:
            from app.database import Base, engine
            from app.models import PtAdvice, PtPriceCache, PtStrategy, PtTransaction  # noqa: F401

            Base.metadata.create_all(bind=engine, tables=[
                PtTransaction.__table__,
                PtPriceCache.__table__,
                PtAdvice.__table__,
                PtStrategy.__table__,
            ])
            logger.info("Portfolio Tracking attivo: tabelle pt_* pronte")
        except Exception as e:
            logger.warning(f"Portfolio Tracking table init skipped: {e}")

    # CRITICAL #2 council 2026-05-27: lock initial model snapshot se non esiste.
    # Versione sealed iniziale = "v1.0-sealed-YYYY-MM-DD". Successivi lock
    # manuali via POST /api/v1/model/lock.
    try:
        from app.services.validation.model_lock import (
            get_active_version, lock_current_model,
        )
        active = get_active_version()
        if active == "unsealed":
            from datetime import date as _date
            version = f"v1.0-sealed-{_date.today().isoformat()}"
            info = lock_current_model(
                version=version,
                description="Initial sealed snapshot — council CRITICAL #2 trigger.",
            )
            logger.info(f"Locked initial model snapshot: {info['version']}")
        else:
            logger.info(f"Active model version: {active}")
    except Exception as e:
        logger.warning(f"Model snapshot lock init skipped: {e}")

    start_scheduler()
    threading.Thread(target=_maybe_backfill_on_startup, daemon=True).start()
    yield
    stop_scheduler()
    logger.info("Macro Analyzer API fermata")


app = FastAPI(
    title="Macro Analyzer",
    description="Sistema di classificazione macro-regime e scoring asset class",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router)

if settings.enable_portfolio_tracking:
    from app.api.portfolio_tracking import router as pt_router

    app.include_router(pt_router)


@app.get("/")
def root():
    return {"message": "Macro Analyzer API", "docs": "/docs"}
