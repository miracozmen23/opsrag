"""Retrieval followed by source-grounded answer generation."""

import logging
import math
from time import perf_counter
from typing import Literal, Protocol

from app.llm.base import LanguageModel
from app.rag.models import RAGMetadata, RAGResult, RAGSource
from app.rag.prompts import (
    GROUNDING_INSTRUCTIONS,
    INSUFFICIENT_CONTEXT_ANSWER,
    build_grounded_input,
)
from app.retrieval.models import RetrievedChunk

logger = logging.getLogger(__name__)


class Retriever(Protocol):
    """Retrieval contract used by the basic RAG pipeline."""

    def search(self, query: str, top_k: int) -> list[RetrievedChunk]:
        """Return ranked chunks for a question."""

        ...


class RAGPipelineError(RuntimeError):
    """Raised when a RAG request cannot be completed safely."""


class RAGPipeline:
    """Provider-neutral retrieval and grounded generation pipeline."""

    def __init__(
        self,
        retriever: Retriever,
        llm: LanguageModel,
        *,
        top_k: int = 10,
        retrieval_method: Literal["dense", "hybrid_reranked"] = "dense",
    ) -> None:
        if top_k < 1:
            raise ValueError("RAG top_k must be at least 1.")
        self.retriever = retriever
        self.llm = llm
        self.top_k = top_k
        self.retrieval_method = retrieval_method

    def answer(self, question: str) -> RAGResult:
        normalized_question = question.strip()
        if not normalized_question:
            raise RAGPipelineError("Question cannot be empty.")

        started_at = perf_counter()
        retrieval_started_at = perf_counter()
        try:
            chunks = self.retriever.search(normalized_question, self.top_k)
        except Exception as exc:
            raise RAGPipelineError("Knowledge-base retrieval failed.") from exc
        retrieval_latency_ms = (perf_counter() - retrieval_started_at) * 1000

        if not chunks:
            logger.info(
                "rag_completed retrieved_chunks=0 retrieval_latency_ms=%.2f total_latency_ms=%.2f",
                retrieval_latency_ms,
                (perf_counter() - started_at) * 1000,
            )
            return RAGResult(
                answer=INSUFFICIENT_CONTEXT_ANSWER,
                sources=[],
                retrieval_confidence=0.0,
                metadata=RAGMetadata(
                    retrieved_chunks=0,
                    retrieval_method=self.retrieval_method,
                ),
            )

        sources = _build_sources(chunks)
        prompt_input = build_grounded_input(normalized_question, chunks)
        generation_started_at = perf_counter()
        try:
            answer = self.llm.generate(
                instructions=GROUNDING_INSTRUCTIONS,
                input_text=prompt_input,
            )
        except Exception as exc:
            raise RAGPipelineError("Answer generation failed.") from exc
        generation_latency_ms = (perf_counter() - generation_started_at) * 1000

        logger.info(
            "rag_completed provider=%s model=%s retrieved_chunks=%d "
            "retrieval_latency_ms=%.2f generation_latency_ms=%.2f total_latency_ms=%.2f",
            self.llm.provider_name,
            self.llm.model_name,
            len(chunks),
            retrieval_latency_ms,
            generation_latency_ms,
            (perf_counter() - started_at) * 1000,
        )
        return RAGResult(
            answer=answer,
            sources=sources,
            retrieval_confidence=_retrieval_confidence(chunks),
            metadata=RAGMetadata(
                retrieved_chunks=len(chunks),
                retrieval_method=self.retrieval_method,
            ),
        )


def _build_sources(chunks: list[RetrievedChunk]) -> list[RAGSource]:
    return [
        RAGSource(
            source_id=f"S{index}",
            document=chunk.metadata.source,
            section=chunk.metadata.section,
            score=round(_relevance_score(chunk), 4),
            chunk_id=chunk.metadata.chunk_id,
        )
        for index, chunk in enumerate(chunks, start=1)
    ]


def _relevance_score(chunk: RetrievedChunk) -> float:
    if chunk.rerank_score is None:
        return float(chunk.score)
    return _sigmoid(float(chunk.rerank_score))


def _sigmoid(value: float) -> float:
    """Map a cross-encoder logit to a stable 0-1 relevance heuristic."""

    if value >= 0:
        return 1.0 / (1.0 + math.exp(-value))
    exp_value = math.exp(value)
    return exp_value / (1.0 + exp_value)


def _retrieval_confidence(chunks: list[RetrievedChunk]) -> float:
    """Clamp the best final relevance score; this is not a probability."""

    top_score = max(_relevance_score(chunk) for chunk in chunks)
    return round(max(0.0, min(1.0, top_score)), 4)
