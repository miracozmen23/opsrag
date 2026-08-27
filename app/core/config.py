"""Environment-backed application settings."""

from functools import lru_cache
from pathlib import Path
from typing import Any

from pydantic import Field, SecretStr, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def resolve_project_path(path: Path) -> Path:
    """Resolve relative runtime paths against the repository, not the shell cwd."""

    return path if path.is_absolute() else (PROJECT_ROOT / path).resolve()


class Settings(BaseSettings):
    """Immutable runtime configuration loaded from environment variables."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        frozen=True,
    )

    app_name: str = "OpsRAG"
    app_env: str = "development"
    log_level: str = "INFO"

    chunk_size_tokens: int = Field(default=600, ge=50)
    chunk_overlap_tokens: int = Field(default=75, ge=0)
    tokenizer_strategy: str = "regex_v1"

    model_cache_dir: Path = Path(".cache/huggingface")
    embedding_model: str = "BAAI/bge-small-en-v1.5"
    embedding_device: str = "cpu"
    embedding_batch_size: int = Field(default=32, ge=1)

    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L6-v2"
    reranker_device: str = "cpu"
    reranker_batch_size: int = Field(default=16, ge=1)

    qdrant_url: str = "http://localhost:6333"
    qdrant_api_key: SecretStr | None = None
    qdrant_collection: str = "opsrag_documents"
    qdrant_timeout_seconds: float = Field(default=10.0, gt=0)
    qdrant_batch_size: int = Field(default=64, ge=1)
    top_k_dense: int = Field(default=10, ge=1)
    top_k_sparse: int = Field(default=10, ge=1)
    top_k_hybrid: int = Field(default=10, ge=1)
    top_k_rerank: int = Field(default=5, ge=1)
    rrf_k: int = Field(default=60, ge=1)

    processed_chunks_path: Path = Path("data/processed/chunks.jsonl")

    llm_provider: str = "openai"
    llm_model: str = ""
    llm_api_key: SecretStr | None = None
    llm_timeout_seconds: float = Field(default=30.0, gt=0)
    llm_max_output_tokens: int = Field(default=800, ge=1)
    ollama_base_url: str = "http://localhost:11434"

    ragas_judge_provider: str = ""
    ragas_judge_model: str = ""
    ragas_cache_dir: Path = Path(".cache/ragas")
    ragas_timeout_seconds: float = Field(default=60.0, gt=0)
    ragas_max_retries: int = Field(default=3, ge=0)
    ragas_max_output_tokens: int = Field(default=512, ge=64)

    @field_validator("qdrant_api_key", "llm_api_key", mode="before")
    @classmethod
    def empty_secret_to_none(cls, value: Any) -> Any:
        """Treat blank optional secrets as unset values."""

        if isinstance(value, str) and not value.strip():
            return None
        return value

    @field_validator("llm_provider", "ragas_judge_provider")
    @classmethod
    def normalize_provider(cls, value: str, info: Any) -> str:
        normalized = value.strip().lower()
        if not normalized and info.field_name == "llm_provider":
            raise ValueError("LLM provider cannot be empty.")
        return normalized

    @field_validator("chunk_overlap_tokens")
    @classmethod
    def validate_overlap(cls, value: int, info: Any) -> int:
        chunk_size = info.data.get("chunk_size_tokens", 600)
        if value >= chunk_size:
            raise ValueError("Chunk overlap must be smaller than chunk size.")
        return value


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return one cached settings object for the process."""

    return Settings()
