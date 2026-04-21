"""FastAPI application entry point."""

import threading
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from loguru import logger

from app.api.routes import router
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


@app.get("/")
def root():
    return {"message": "Macro Analyzer API", "docs": "/docs"}
