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
    gemini_api_key: str = ""
    scheduler_hour: int = 6
    scheduler_minute: int = 0


settings = Settings()
