"""Minimal LangGraph routing between grounded RAG and direct responses."""

import logging
import re
from time import perf_counter
from typing import Literal, Protocol, TypedDict

from langgraph.graph import END, START, StateGraph

from app.llm.base import LanguageModel
from app.observability import NoOpObservability, ObservabilityClient
from app.rag.models import RAGMetadata, RAGResult
from app.rag.pipeline import RAGPipelineError

logger = logging.getLogger(__name__)

QueryRoute = Literal["knowledge", "general"]

GENERAL_INSTRUCTIONS = """You are OpsRAG, a concise technical assistant.
Respond conversationally to the user's clearly general message.
Do not claim that you searched the knowledge base and do not add source citations.
Answer in the same language as the user when practical."""

_GENERAL_UTTERANCES = frozenset(
    {
        "hello",
        "hello opsrag",
        "hey",
        "hi",
        "good morning",
        "good afternoon",
        "good evening",
        "how are you",
        "who are you",
        "what can you do",
        "thanks",
        "thank you",
        "tell me a joke",
        "merhaba",
        "merhaba opsrag",
        "selam",
        "günaydın",
        "iyi akşamlar",
        "nasılsın",
        "kimsin",
        "ne yapabilirsin",
        "teşekkürler",
        "teşekkür ederim",
        "sağ ol",
        "bana bir şaka yap",
    }
)
_EDGE_PUNCTUATION = re.compile(r"^[\s!?.,;:]+|[\s!?.,;:]+$")
_SOURCE_LIKE_PATTERN = re.compile(r"\[S[^\]\r\n]*\]")
_GENERAL_FACT_PATTERNS = (
    re.compile(r"^(?:what|which) (?:is|are) the capital of\b"),
    re.compile(r"^who (?:is|was|are|were)\b"),
    re.compile(r"^.+ başkenti (?:nedir|neresidir)$"),
    re.compile(r"^.+ kimdir$"),
)


class AnswerService(Protocol):
    """Minimal answer contract shared by the RAG pipeline and graph."""

    def answer(self, question: str) -> RAGResult:
        """Return one answer result."""

        ...


class QueryRouter(Protocol):
    """Classify a question into one of the graph's bounded routes."""

    def route(self, question: str) -> QueryRoute:
        """Return the route for a non-empty question."""

        ...


class RoutingState(TypedDict, total=False):
    """State shared by the classifier and the two terminal answer nodes."""

    question: str
    route: QueryRoute
    result: RAGResult


class RuleBasedQueryRouter:
    """Conservative deterministic router: only obvious general messages bypass RAG."""

    def route(self, question: str) -> QueryRoute:
        normalized = _normalize_for_routing(question)
        if not normalized:
            raise ValueError("Routing question cannot be empty.")
        is_general = normalized in _GENERAL_UTTERANCES or any(
            pattern.search(normalized) for pattern in _GENERAL_FACT_PATTERNS
        )
        return "general" if is_general else "knowledge"


class QueryRoutingGraph:
    """Single-pass LangGraph workflow with knowledge and general branches."""

    def __init__(
        self,
        rag_pipeline: AnswerService,
        llm: LanguageModel,
        *,
        router: QueryRouter | None = None,
        observability: ObservabilityClient | None = None,
    ) -> None:
        self.rag_pipeline = rag_pipeline
        self.llm = llm
        self.router = router or RuleBasedQueryRouter()
        self.observability = observability or NoOpObservability()
        self.workflow = self._build_workflow()

    def answer(self, question: str) -> RAGResult:
        """Route and answer one normalized question through exactly one branch."""

        normalized_question = question.strip()
        if not normalized_question:
            raise RAGPipelineError("Question cannot be empty.")

        started_at = perf_counter()
        with self.observability.observe(
            name="opsrag.query",
            as_type="chain",
            input={"question": normalized_question},
            metadata={"workflow": "langgraph"},
        ) as trace:
            try:
                final_state = self.workflow.invoke({"question": normalized_question})
                result = final_state.get("result")
                route = final_state.get("route")
                if not isinstance(result, RAGResult) or route not in (
                    "knowledge",
                    "general",
                ):
                    raise RAGPipelineError(
                        "Query routing produced an invalid result."
                    )
            except RAGPipelineError as exc:
                trace.update(
                    level="ERROR",
                    status_message=str(exc),
                    output={"error": str(exc)},
                )
                raise
            except Exception as exc:
                trace.update(
                    level="ERROR",
                    status_message="Query routing failed.",
                    output={"error_type": type(exc).__name__},
                )
                raise RAGPipelineError("Query routing failed.") from exc

            latency_ms = (perf_counter() - started_at) * 1000
            trace.update(
                output=result.model_dump(mode="json"),
                metadata={
                    "route": route,
                    "retrieval_method": result.metadata.retrieval_method,
                    "latency_ms": f"{latency_ms:.2f}",
                },
            )
            logger.info(
                "query_routing_completed route=%s latency_ms=%.2f",
                route,
                latency_ms,
            )
            return result

    def _build_workflow(self):
        builder = StateGraph(RoutingState)
        builder.add_node("classify", self._classify_node)
        builder.add_node("rag", self._rag_node)
        builder.add_node("general", self._general_node)
        builder.add_edge(START, "classify")
        builder.add_conditional_edges(
            "classify",
            self._route_from_state,
            {"knowledge": "rag", "general": "general"},
        )
        builder.add_edge("rag", END)
        builder.add_edge("general", END)
        return builder.compile()

    def _classify_node(self, state: RoutingState) -> RoutingState:
        with self.observability.observe(
            name="query.classify",
            as_type="span",
            input={"question": state["question"]},
        ) as observation:
            try:
                route = self.router.route(state["question"])
            except Exception as exc:
                observation.update(
                    level="ERROR",
                    status_message="Query classification failed.",
                    output={"error_type": type(exc).__name__},
                )
                raise
            observation.update(output={"route": route})
        logger.info("query_classified route=%s", route)
        return {"route": route}

    @staticmethod
    def _route_from_state(state: RoutingState) -> QueryRoute:
        return state["route"]

    def _rag_node(self, state: RoutingState) -> RoutingState:
        result = self.rag_pipeline.answer(state["question"])
        if result.metadata.route != "knowledge":
            result = result.model_copy(
                update={
                    "metadata": result.metadata.model_copy(
                        update={"route": "knowledge"}
                    )
                }
            )
        return {"result": result}

    def _general_node(self, state: RoutingState) -> RoutingState:
        with self.observability.observe(
            name="llm.general",
            as_type="generation",
            input={
                "instructions": GENERAL_INSTRUCTIONS,
                "question": state["question"],
            },
            model=self.llm.model_name,
            metadata={"provider": self.llm.provider_name},
        ) as generation:
            try:
                answer = self.llm.generate(
                    instructions=GENERAL_INSTRUCTIONS,
                    input_text=state["question"],
                )
                if _SOURCE_LIKE_PATTERN.search(answer):
                    raise ValueError(
                        "Direct answers cannot contain source citations."
                    )
                result = RAGResult(
                    answer=answer,
                    sources=[],
                    retrieval_confidence=0.0,
                    metadata=RAGMetadata(
                        retrieved_chunks=0,
                        cited_sources=0,
                        retrieval_method="not_used",
                        route="general",
                    ),
                )
            except Exception as exc:
                generation.update(
                    level="ERROR",
                    status_message="Direct answer generation failed.",
                    output={"error_type": type(exc).__name__},
                )
                raise RAGPipelineError("Direct answer generation failed.") from exc
            generation.update(output=answer)
            return {"result": result}


def _normalize_for_routing(question: str) -> str:
    collapsed = " ".join(question.casefold().split())
    return _EDGE_PUNCTUATION.sub("", collapsed)
