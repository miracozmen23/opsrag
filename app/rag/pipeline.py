"""Retrieval followed by source-grounded answer generation."""

import logging
from time import perf_counter
from typing import Literal, Protocol

from app.llm.base import LanguageModel
from app.observability import NoOpObservability, ObservabilityClient
from app.rag.attribution import (
    SourceAttributionError,
    build_source_contexts,
    select_cited_sources,
)
from app.rag.models import RAGExecution, RAGMetadata, RAGResult, RAGSource
from app.rag.prompts import (
    GROUNDING_INSTRUCTIONS,
    INSUFFICIENT_CONTEXT_ANSWER,
    build_grounded_input,
    is_insufficient_context_answer,
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
        retrieval_method: Literal["dense", "hybrid", "hybrid_reranked"] = "dense",
        observability: ObservabilityClient | None = None,
    ) -> None:
        if top_k < 1:
            raise ValueError("RAG top_k must be at least 1.")
        self.retriever = retriever
        self.llm = llm
        self.top_k = top_k
        self.retrieval_method = retrieval_method
        self.observability = observability or NoOpObservability()

    def answer(self, question: str) -> RAGResult:
        """Return the public grounded answer contract."""

        return self.answer_with_trace(question).result

    def answer_with_trace(self, question: str) -> RAGExecution:
        """Return an answer with the exact ranked chunks used as context."""

        normalized_question = question.strip()
        if not normalized_question:
            raise RAGPipelineError("Question cannot be empty.")

        started_at = perf_counter()
        with self.observability.observe(
            name="rag.pipeline",
            as_type="chain",
            input={"question": normalized_question},
            metadata={
                "retrieval_method": self.retrieval_method,
                "top_k": str(self.top_k),
            },
        ) as pipeline_observation:
            try:
                execution = self._answer_normalized(normalized_question, started_at)
            except RAGPipelineError as exc:
                pipeline_observation.update(
                    level="ERROR",
                    status_message=str(exc),
                    output={"error": str(exc)},
                )
                raise
            except Exception as exc:
                pipeline_observation.update(
                    level="ERROR",
                    status_message="RAG pipeline failed.",
                    output={"error_type": type(exc).__name__},
                )
                raise

            pipeline_observation.update(
                output=execution.result.model_dump(mode="json"),
                metadata={
                    "retrieval_method": self.retrieval_method,
                    "total_latency_ms": (
                        f"{(perf_counter() - started_at) * 1000:.2f}"
                    ),
                },
            )
            return execution

    def _answer_normalized(
        self,
        normalized_question: str,
        started_at: float,
    ) -> RAGExecution:
        retrieval_started_at = perf_counter()
        with self.observability.observe(
            name="rag.retrieve",
            as_type="retriever",
            input={"query": normalized_question, "top_k": self.top_k},
            metadata={"retrieval_method": self.retrieval_method},
        ) as retrieval_observation:
            try:
                chunks = self.retriever.search(normalized_question, self.top_k)
            except Exception as exc:
                retrieval_observation.update(
                    level="ERROR",
                    status_message="Knowledge-base retrieval failed.",
                    output={"error_type": type(exc).__name__},
                )
                raise RAGPipelineError("Knowledge-base retrieval failed.") from exc
            retrieval_latency_ms = (perf_counter() - retrieval_started_at) * 1000
            retrieval_observation.update(
                output={"chunks": _trace_retrieved_chunks(chunks)},
                metadata={
                    "retrieval_method": self.retrieval_method,
                    "result_count": str(len(chunks)),
                    "latency_ms": f"{retrieval_latency_ms:.2f}",
                },
            )

        if not chunks:
            total_latency_ms = (perf_counter() - started_at) * 1000
            logger.info(
                "rag_completed retrieved_chunks=0 retrieval_latency_ms=%.2f total_latency_ms=%.2f",
                retrieval_latency_ms,
                total_latency_ms,
            )
            return RAGExecution(
                result=RAGResult(
                    answer=INSUFFICIENT_CONTEXT_ANSWER,
                    sources=[],
                    retrieval_confidence=0.0,
                    metadata=RAGMetadata(
                        retrieved_chunks=0,
                        retrieval_method=self.retrieval_method,
                    ),
                ),
                retrieved_chunks=(),
            )

        source_contexts = build_source_contexts(chunks)
        prompt_input = build_grounded_input(normalized_question, source_contexts)
        generation_started_at = perf_counter()
        with self.observability.observe(
            name="rag.generate",
            as_type="generation",
            input={
                "instructions": GROUNDING_INSTRUCTIONS,
                "prompt": prompt_input,
            },
            model=self.llm.model_name,
            metadata={"provider": self.llm.provider_name},
        ) as generation_observation:
            try:
                answer = self.llm.generate(
                    instructions=GROUNDING_INSTRUCTIONS,
                    input_text=prompt_input,
                )
            except Exception as exc:
                generation_observation.update(
                    level="ERROR",
                    status_message="Answer generation failed.",
                    output={"error_type": type(exc).__name__},
                )
                raise RAGPipelineError("Answer generation failed.") from exc
            generation_latency_ms = (
                perf_counter() - generation_started_at
            ) * 1000
            generation_observation.update(
                output=answer,
                metadata={
                    "provider": self.llm.provider_name,
                    "latency_ms": f"{generation_latency_ms:.2f}",
                },
            )

        with self.observability.observe(
            name="rag.attribution",
            as_type="span",
            input={
                "answer": answer,
                "allowed_source_ids": [
                    context.source.source_id for context in source_contexts
                ],
            },
        ) as attribution_observation:
            if is_insufficient_context_answer(answer):
                answer = INSUFFICIENT_CONTEXT_ANSWER
                sources = []
            else:
                try:
                    sources = select_cited_sources(answer, source_contexts)
                except SourceAttributionError as exc:
                    attribution_observation.update(
                        level="ERROR",
                        status_message="Answer source attribution failed.",
                        output={"error_type": type(exc).__name__},
                    )
                    raise RAGPipelineError(
                        "Answer source attribution failed."
                    ) from exc
            attribution_observation.update(
                output={
                    "answer": answer,
                    "cited_source_ids": [source.source_id for source in sources],
                }
            )

        total_latency_ms = (perf_counter() - started_at) * 1000
        logger.info(
            "rag_completed provider=%s model=%s retrieved_chunks=%d "
            "cited_sources=%d retrieval_latency_ms=%.2f "
            "generation_latency_ms=%.2f total_latency_ms=%.2f",
            self.llm.provider_name,
            self.llm.model_name,
            len(chunks),
            len(sources),
            retrieval_latency_ms,
            generation_latency_ms,
            total_latency_ms,
        )
        return RAGExecution(
            result=RAGResult(
                answer=answer,
                sources=sources,
                retrieval_confidence=(
                    _retrieval_confidence(sources) if sources else 0.0
                ),
                metadata=RAGMetadata(
                    retrieved_chunks=len(chunks),
                    cited_sources=len(sources),
                    retrieval_method=self.retrieval_method,
                ),
            ),
            retrieved_chunks=tuple(chunks),
        )


def _retrieval_confidence(sources: list[RAGSource]) -> float:
    """Return the best cited-source score; this is not a probability."""

    return round(max(source.score for source in sources), 4)


def _trace_retrieved_chunks(chunks: list[RetrievedChunk]) -> list[dict[str, object]]:
    """Serialize ranked identifiers and scores without duplicating chunk text."""

    return [
        {
            "rank": rank,
            "chunk_id": chunk.metadata.chunk_id,
            "source": chunk.metadata.source,
            "section": chunk.metadata.section,
            "retrieval_method": chunk.retrieval_method,
            "score": chunk.score,
            "rerank_score": chunk.rerank_score,
        }
        for rank, chunk in enumerate(chunks, start=1)
    ]
