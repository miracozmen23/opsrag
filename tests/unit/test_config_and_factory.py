"""Configuration validation and provider selection."""

import pytest
from pydantic import ValidationError

from app.core.config import Settings
from app.llm.factory import LLMConfigurationError, create_llm_service


def test_blank_optional_secrets_become_none() -> None:
    settings = Settings(_env_file=None, qdrant_api_key="", llm_api_key="")
    assert settings.qdrant_api_key is None
    assert settings.llm_api_key is None


def test_overlap_must_be_smaller_than_chunk_size() -> None:
    with pytest.raises(ValidationError, match="overlap"):
        Settings(_env_file=None, chunk_size_tokens=100, chunk_overlap_tokens=100)


def test_sparse_top_k_must_be_positive() -> None:
    with pytest.raises(ValidationError, match="top_k_sparse"):
        Settings(_env_file=None, top_k_sparse=0)


def test_llm_factory_requires_model_and_key() -> None:
    with pytest.raises(LLMConfigurationError, match="LLM_MODEL"):
        create_llm_service(Settings(_env_file=None, llm_model="", llm_api_key=None))
    with pytest.raises(LLMConfigurationError, match="LLM_API_KEY"):
        create_llm_service(
            Settings(_env_file=None, llm_provider="openai", llm_model="test", llm_api_key=None)
        )


def test_llm_factory_rejects_unknown_provider() -> None:
    settings = Settings(
        _env_file=None,
        llm_provider="unknown",
        llm_model="model",
        llm_api_key="secret",
    )
    with pytest.raises(LLMConfigurationError, match="Unsupported"):
        create_llm_service(settings)
