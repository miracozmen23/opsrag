"""Optional RAGAS 0.4 adapter for the four milestone evaluation metrics."""

import asyncio
import importlib
import math
import os
import sys
import types
from pathlib import Path
from typing import Any

from app.evaluation.results import MetricName, MetricOutcome


class RagasConfigurationError(RuntimeError):
    """Raised when the optional evaluator cannot be configured safely."""


class RagasScorer:
    """Score one sample with the current RAGAS collections API."""

    def __init__(
        self,
        *,
        faithfulness: Any,
        answer_relevance: Any,
        context_precision: Any,
        context_recall: Any,
        ragas_version: str,
        judge_model: str,
        embedding_model: str,
    ) -> None:
        self._faithfulness = faithfulness
        self._answer_relevance = answer_relevance
        self._context_precision = context_precision
        self._context_recall = context_recall
        self.ragas_version = ragas_version
        self.judge_model = judge_model
        self.embedding_model = embedding_model

    async def score(
        self,
        *,
        question: str,
        response: str,
        reference: str,
        retrieved_contexts: list[str],
    ) -> dict[MetricName, MetricOutcome]:
        """Run all metrics and preserve failed or undefined states explicitly."""

        calls = (
            self._faithfulness.ascore(
                user_input=question,
                response=response,
                retrieved_contexts=retrieved_contexts,
            ),
            self._answer_relevance.ascore(
                user_input=question,
                response=response,
            ),
            self._context_precision.ascore(
                user_input=question,
                reference=reference,
                retrieved_contexts=retrieved_contexts,
            ),
            self._context_recall.ascore(
                user_input=question,
                reference=reference,
                retrieved_contexts=retrieved_contexts,
            ),
        )
        raw_results = await asyncio.gather(*calls, return_exceptions=True)
        names: tuple[MetricName, ...] = (
            "faithfulness",
            "answer_relevance",
            "context_precision",
            "context_recall",
        )
        return {
            name: _normalize_metric_result(raw_result)
            for name, raw_result in zip(names, raw_results, strict=True)
        }


def create_ragas_scorer(
    *,
    provider: str,
    api_key: str,
    judge_model: str,
    ollama_base_url: str,
    embedding_model: str,
    embedding_device: str,
    embedding_batch_size: int,
    model_cache_dir: Path,
    ragas_cache_dir: Path,
    timeout_seconds: float,
    max_retries: int,
    max_output_tokens: int,
) -> RagasScorer:
    """Construct OpenAI-compatible metrics with local Hugging Face embeddings."""

    normalized_provider = provider.strip().lower()
    if normalized_provider not in {"openai", "ollama"}:
        raise RagasConfigurationError(
            f"Unsupported RAGAS judge provider '{provider}'."
        )
    if normalized_provider == "openai" and not api_key.strip():
        raise RagasConfigurationError("An OpenAI API key is required for RAGAS.")
    if not judge_model.strip():
        raise RagasConfigurationError("A RAGAS judge model must be configured.")
    if timeout_seconds <= 0:
        raise RagasConfigurationError("RAGAS timeout must be positive.")
    if max_retries < 0:
        raise RagasConfigurationError("RAGAS max retries cannot be negative.")
    if max_output_tokens < 64:
        raise RagasConfigurationError(
            "RAGAS max output tokens must be at least 64."
        )

    os.environ.setdefault("RAGAS_DO_NOT_TRACK", "true")
    _install_optional_vertexai_compatibility()
    try:
        import ragas
        from openai import AsyncOpenAI
        from ragas.cache import DiskCacheBackend
        from ragas.embeddings import HuggingFaceEmbeddings
        from ragas.llms import llm_factory
        from ragas.metrics.collections import (
            AnswerRelevancy,
            ContextPrecision,
            ContextRecall,
            Faithfulness,
        )
    except ModuleNotFoundError as exc:
        raise RagasConfigurationError(
            "RAGAS evaluation dependencies are missing. "
            'Install them with: pip install -e ".[evaluation]"'
        ) from exc

    model_cache_dir.mkdir(parents=True, exist_ok=True)
    ragas_cache_dir.mkdir(parents=True, exist_ok=True)
    cache = DiskCacheBackend(cache_dir=str(ragas_cache_dir.resolve()))
    client_kwargs: dict[str, Any] = {
        "api_key": api_key if normalized_provider == "openai" else "ollama",
        "timeout": timeout_seconds,
        "max_retries": max_retries,
    }
    if normalized_provider == "ollama":
        if not ollama_base_url.strip():
            raise RagasConfigurationError("An Ollama base URL is required for RAGAS.")
        client_kwargs["base_url"] = f"{ollama_base_url.rstrip('/')}/v1"
    client = AsyncOpenAI(**client_kwargs)
    local_model_args: dict[str, Any] = {}
    if normalized_provider == "ollama":
        local_model_args = {
            "max_tokens": max_output_tokens,
            "reasoning_effort": "none",
        }
    judge_llm = llm_factory(
        judge_model.strip(),
        provider="openai",
        client=client,
        cache=cache,
        **local_model_args,
    )
    embeddings = HuggingFaceEmbeddings(
        model=embedding_model,
        device=embedding_device,
        batch_size=embedding_batch_size,
        cache=cache,
        cache_folder=str(model_cache_dir.resolve()),
    )
    return RagasScorer(
        faithfulness=Faithfulness(llm=judge_llm),
        answer_relevance=AnswerRelevancy(
            llm=judge_llm,
            embeddings=embeddings,
        ),
        context_precision=ContextPrecision(llm=judge_llm),
        context_recall=ContextRecall(llm=judge_llm),
        ragas_version=ragas.__version__,
        judge_model=judge_model.strip(),
        embedding_model=embedding_model,
    )


def _normalize_metric_result(result: Any) -> MetricOutcome:
    if isinstance(result, BaseException):
        message = f"{type(result).__name__}: {result}"[:1000]
        return MetricOutcome(status="failed", error=message)

    raw_value = getattr(result, "value", None)
    reason = getattr(result, "reason", None)
    normalized_reason = str(reason)[:4000] if reason is not None else None
    try:
        value = float(raw_value)
    except (TypeError, ValueError):
        return MetricOutcome(status="undefined", reason=normalized_reason)
    if not math.isfinite(value):
        return MetricOutcome(status="undefined", reason=normalized_reason)
    return MetricOutcome(
        status="scored",
        value=value,
        reason=normalized_reason,
    )


def _install_optional_vertexai_compatibility() -> None:
    """Work around RAGAS 0.4.3 importing a removed optional LangChain path."""

    module_name = "langchain_community.chat_models.vertexai"
    if module_name in sys.modules:
        return
    try:
        importlib.import_module(module_name)
        return
    except ModuleNotFoundError as exc:
        if exc.name != module_name:
            raise

    stub = types.ModuleType(module_name)

    class ChatVertexAI:
        """Unavailable optional provider marker required only during RAGAS import."""

    stub.ChatVertexAI = ChatVertexAI
    sys.modules[module_name] = stub
