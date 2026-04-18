"""FastAPI application entry point."""

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from loguru import logger

from app.api.routes import router
from app.scheduler.jobs import start_scheduler, stop_scheduler


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup e shutdown events."""
    logger.info("Avvio Macro Analyzer API...")
    start_scheduler()
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
