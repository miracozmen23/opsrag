"""Configuration validation and provider selection."""

import pytest
from pydantic import ValidationError

from app.core.config import PROJECT_ROOT, Settings
from app.core.service_factory import create_embedding_service, create_reranker_service
from app.llm.factory import LLMConfigurationError, create_llm_service


def test_blank_optional_secrets_become_none() -> None:
    settings = Settings(_env_file=None, qdrant_api_key="", llm_api_key="")
    assert settings.qdrant_api_key is None
    assert settings.llm_api_key is None


def test_overlap_must_be_smaller_than_chunk_size() -> None:
    with pytest.raises(ValidationError, match="overlap"):
        Settings(_env_file=None, chunk_size_tokens=100, chunk_overlap_tokens=100)


@pytest.mark.parametrize(
    "field_name",
    ["top_k_sparse", "top_k_hybrid", "top_k_rerank", "rrf_k"],
)
def test_retrieval_counts_must_be_positive(field_name: str) -> None:
    with pytest.raises(ValidationError, match=field_name):
        Settings(_env_file=None, **{field_name: 0})


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


def test_model_factories_share_configured_cache_directory(tmp_path) -> None:
    settings = Settings(_env_file=None, model_cache_dir=tmp_path)
    embedding_service = create_embedding_service(settings)
    reranker = create_reranker_service(settings)
    assert embedding_service.cache_folder == tmp_path
    assert reranker.cache_folder == tmp_path


def test_relative_model_cache_is_anchored_to_project_root() -> None:
    settings = Settings(_env_file=None, model_cache_dir="models/cache")
    assert create_embedding_service(settings).cache_folder == PROJECT_ROOT / "models/cache"
    assert create_reranker_service(settings).cache_folder == PROJECT_ROOT / "models/cache"
