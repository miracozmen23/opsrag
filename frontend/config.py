"""Environment-backed settings for the Streamlit demo."""

from functools import lru_cache
from urllib.parse import urlsplit

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from app.core.config import PROJECT_ROOT


class FrontendSettings(BaseSettings):
    """Immutable connection settings for the FastAPI client."""

    model_config = SettingsConfigDict(
        env_file=PROJECT_ROOT / ".env",
        env_file_encoding="utf-8",
        env_prefix="OPSRAG_",
        extra="ignore",
        frozen=True,
    )

    api_base_url: str = "http://localhost:8000"
    api_timeout_seconds: float = Field(default=300.0, gt=0)

    @field_validator("api_base_url")
    @classmethod
    def validate_api_base_url(cls, value: str) -> str:
        normalized = value.strip().rstrip("/")
        parsed = urlsplit(normalized)
        if parsed.scheme not in {"http", "https"} or not parsed.netloc:
            raise ValueError("API base URL must be a valid HTTP(S) URL.")
        return normalized


@lru_cache(maxsize=1)
def get_frontend_settings() -> FrontendSettings:
    """Return one cached frontend settings object for the process."""

    return FrontendSettings()
