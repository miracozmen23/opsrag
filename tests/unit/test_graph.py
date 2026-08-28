"""Minimal LangGraph query routing behavior."""

from typing import Literal

import pytest

from app.rag.graph import QueryRoutingGraph, RuleBasedQueryRouter
from app.rag.models import RAGMetadata, RAGResult, RAGSource
from app.rag.pipeline import RAGPipelineError
from tests.helpers import RecordingObservability


def make_rag_result() -> RAGResult:
    return RAGResult(
        answer="Use the documented service name [S1].",
        sources=[
            RAGSource(
                source_id="S1",
                document="docker.md",
                title="Docker Guide",
                section="Networking",
                score=0.91,
                chunk_id="chunk_1",
                chunk_ids=("chunk_1",),
            )
        ],
        retrieval_confidence=0.91,
        metadata=RAGMetadata(
            retrieved_chunks=3,
            cited_sources=1,
            retrieval_method="hybrid_reranked",
        ),
    )


class FakeRAGPipeline:
    def __init__(self) -> None:
        self.calls: list[str] = []

    def answer(self, question: str) -> RAGResult:
        self.calls.append(question)
        return make_rag_result()


class FakeLanguageModel:
    provider_name = "fake"
    model_name = "fake-model"

    def __init__(self, answer: str = "Hello! How can I help?") -> None:
        self.answer_text = answer
        self.calls: list[dict[str, str]] = []

    def generate(self, *, instructions: str, input_text: str) -> str:
        self.calls.append({"instructions": instructions, "input_text": input_text})
        return self.answer_text


class FixedRouter:
    def __init__(self, route: Literal["knowledge", "general"]) -> None:
        self.selected_route = route
        self.calls: list[str] = []

    def route(self, question: str) -> Literal["knowledge", "general"]:
        self.calls.append(question)
        return self.selected_route


@pytest.mark.parametrize(
    "question",
    [
        "Hello!",
        "WHAT CAN YOU DO?",
        "  Thank you.  ",
        "Merhaba OpsRAG!",
        "Nasılsın?",
        "Bana bir şaka yap.",
        "What is the capital of France?",
        "Fransa'nın başkenti nedir?",
        "Ada Lovelace kimdir?",
    ],
)
def test_rule_router_bypasses_rag_for_clearly_general_messages(question: str) -> None:
    assert RuleBasedQueryRouter().route(question) == "general"


@pytest.mark.parametrize(
    "question",
    [
        "How do I resolve Qdrant HTTP 503?",
        "Why is PostgreSQL unavailable in Docker Compose?",
        "What is Qdrant?",
    ],
)
def test_rule_router_defaults_nontrivial_questions_to_knowledge(question: str) -> None:
    assert RuleBasedQueryRouter().route(question) == "knowledge"


def test_knowledge_route_runs_rag_node_only() -> None:
    rag_pipeline = FakeRAGPipeline()
    llm = FakeLanguageModel()
    router = FixedRouter("knowledge")
    graph = QueryRoutingGraph(rag_pipeline, llm, router=router)

    result = graph.answer(" How do I fix Qdrant? ")

    assert router.calls == ["How do I fix Qdrant?"]
    assert rag_pipeline.calls == ["How do I fix Qdrant?"]
    assert llm.calls == []
    assert result.sources[0].document == "docker.md"
    assert result.metadata.route == "knowledge"
    assert result.metadata.retrieval_method == "hybrid_reranked"


def test_general_route_runs_direct_llm_node_without_retrieval() -> None:
    rag_pipeline = FakeRAGPipeline()
    llm = FakeLanguageModel("Merhaba! Size nasıl yardımcı olabilirim?")
    router = FixedRouter("general")
    graph = QueryRoutingGraph(rag_pipeline, llm, router=router)

    result = graph.answer(" Merhaba! ")

    assert rag_pipeline.calls == []
    assert len(llm.calls) == 1
    assert llm.calls[0]["input_text"] == "Merhaba!"
    assert "do not add source citations" in llm.calls[0]["instructions"]
    assert result.answer.startswith("Merhaba")
    assert result.sources == []
    assert result.retrieval_confidence == 0.0
    assert result.metadata == RAGMetadata(
        retrieved_chunks=0,
        cited_sources=0,
        retrieval_method="not_used",
        route="general",
    )


def test_general_route_records_nested_query_and_generation_observations() -> None:
    observability = RecordingObservability()
    graph = QueryRoutingGraph(
        FakeRAGPipeline(),
        FakeLanguageModel("Hello from the model."),
        router=FixedRouter("general"),
        observability=observability,
    )

    result = graph.answer("Hello")

    assert result.answer == "Hello from the model."
    assert [record["name"] for record in observability.records] == [
        "opsrag.query",
        "query.classify",
        "llm.general",
    ]
    generation = observability.by_name("llm.general")
    assert generation["parent"] == "opsrag.query"
    assert generation["model"] == "fake-model"
    assert generation["input"]["question"] == "Hello"
    assert generation["updates"][-1] == {"output": "Hello from the model."}
    root_update = observability.by_name("opsrag.query")["updates"][-1]
    assert root_update["output"]["metadata"]["route"] == "general"
    assert root_update["metadata"]["retrieval_method"] == "not_used"


def test_workflow_contains_only_bounded_single_pass_nodes() -> None:
    graph = QueryRoutingGraph(FakeRAGPipeline(), FakeLanguageModel())
    drawable = graph.workflow.get_graph()
    assert set(drawable.nodes) == {"__start__", "classify", "rag", "general", "__end__"}


@pytest.mark.parametrize("answer", ["", "Hello [S1]."])
def test_general_route_rejects_invalid_direct_answers(answer: str) -> None:
    graph = QueryRoutingGraph(
        FakeRAGPipeline(),
        FakeLanguageModel(answer),
        router=FixedRouter("general"),
    )
    with pytest.raises(RAGPipelineError, match="Direct answer generation failed"):
        graph.answer("hello")


def test_general_route_records_error_state_before_raising() -> None:
    observability = RecordingObservability()
    graph = QueryRoutingGraph(
        FakeRAGPipeline(),
        FakeLanguageModel("Invalid [S1]."),
        router=FixedRouter("general"),
        observability=observability,
    )

    with pytest.raises(RAGPipelineError, match="Direct answer generation failed"):
        graph.answer("hello")

    generation_updates = observability.by_name("llm.general")["updates"]
    root_updates = observability.by_name("opsrag.query")["updates"]
    assert generation_updates[-1]["level"] == "ERROR"
    assert root_updates[-1]["level"] == "ERROR"


def test_graph_rejects_blank_question_before_routing() -> None:
    with pytest.raises(RAGPipelineError, match="cannot be empty"):
        QueryRoutingGraph(FakeRAGPipeline(), FakeLanguageModel()).answer("   ")
