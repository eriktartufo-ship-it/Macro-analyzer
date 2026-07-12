from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=PROJECT_ROOT / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    database_url: str = "postgresql+psycopg://postgres:password@localhost:5432/macro_analyzer"
    fred_api_key: str = ""
    groq_api_key: str = ""
    newsapi_key: str = ""
    gemini_api_key: str = ""  # Gemini 2.5 Flash, provider preferito per FOMC + dedollar explainer
    scheduler_hour: int = 6
    scheduler_minute: int = 0

    # Portfolio Tracking (modulo pt_*, default OFF — additivo, zero impatto se spento)
    enable_portfolio_tracking: bool = False
    # Formato: "username:Display Name:password,username2:Display2:password2"
    # (fallback 2 campi "username:password" → display = username capitalizzato)
    pt_users: str = ""
    # Secret HMAC per i token di sessione. MAI cambiarlo una volta in prod
    # (lezione RGV: invalida tutte le sessioni attive).
    pt_secret: str = ""
    # Token admin per push da Claude Code (strategy vetrina + import bulk)
    pt_admin_token: str = ""


settings = Settings()
