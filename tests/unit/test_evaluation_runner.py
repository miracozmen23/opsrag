"""Offline benchmark orchestration, summaries, and artifact contracts."""

import asyncio
import json
from datetime import datetime, timezone
from pathlib import Path

import pytest
from pydantic import ValidationError

from app.evaluation.models import EvaluationCase, EvaluationDataset
from app.evaluation.results import (
    METRIC_NAMES,
    BenchmarkResults,
    MetricOutcome,
)
from app.evaluation.runner import (
    APPLICATION_FAILURE_RESPONSE,
    create_evaluation_pipelines,
    has_metric_failures,
    run_benchmark,
    sha256_file,
    write_benchmark_results,
)
from app.rag.pipeline import RAGPipelineError
from app.rag.models import RAGExecution, RAGMetadata, RAGResult, RAGSource
from app.rag.prompts import INSUFFICIENT_CONTEXT_ANSWER
from tests.helpers import make_retrieved_chunk


class FakePipeline:
    def __init__(self, executions: dict[str, RAGExecution]) -> None:
        self.executions = executions
        self.calls: list[str] = []

    def answer_with_trace(self, question: str) -> RAGExecution:
        self.calls.append(question)
        return self.executions[question]


class FakeScorer:
    ragas_version = "test-ragas"
    judge_model = "judge-model"
    embedding_model = "embedding-model"

    async def score(self, **kwargs) -> dict[str, MetricOutcome]:
        value = 0.8 if kwargs["question"] == "Answerable?" else 0.6
        return {
            name: MetricOutcome(status="scored", value=value)
            for name in METRIC_NAMES
        }


class FakeRetrieverSuite:
    def __init__(self) -> None:
        self.requested: list[str] = []

    def get(self, configuration: str):
        self.requested.append(configuration)
        return object()


def test_metric_outcome_does_not_allow_fake_zero_for_failures() -> None:
    with pytest.raises(ValidationError, match="require an error"):
        MetricOutcome(status="failed", value=0.0)
    with pytest.raises(ValidationError, match="finite value"):
        MetricOutcome(status="scored", value=float("nan"))


def test_runner_evaluates_every_case_and_summarizes_real_scores(
    tmp_path: Path,
) -> None:
    dataset_path = tmp_path / "questions.jsonl"
    dataset_path.write_text("reviewed benchmark\n", encoding="utf-8")
    answerable = EvaluationCase(
        case_id="eval_001",
        question="Answerable?",
        expected_answer="Use the guide.",
        expected_source="guide.md",
        expected_section="Section",
        category="semantic",
        should_answer=True,
    )
    insufficient = EvaluationCase(
        case_id="eval_002",
        question="Unavailable?",
        expected_answer=INSUFFICIENT_CONTEXT_ANSWER,
        expected_source=None,
        expected_section=None,
        category="insufficient_context",
        should_answer=False,
    )
    chunk = make_retrieved_chunk(
        text="Grounded evidence.",
        source="guide.md",
        section="Section",
    )
    answer_result = RAGResult(
        answer="Use the guide [S1].",
        sources=[
            RAGSource(
                source_id="S1",
                document="guide.md",
                title="Guide",
                section="Section",
                score=0.9,
                chunk_id=chunk.metadata.chunk_id,
                chunk_ids=(chunk.metadata.chunk_id,),
            )
        ],
        retrieval_confidence=0.9,
        metadata=RAGMetadata(retrieved_chunks=1, cited_sources=1),
    )
    unavailable_result = RAGResult(
        answer=INSUFFICIENT_CONTEXT_ANSWER,
        sources=[],
        retrieval_confidence=0.0,
        metadata=RAGMetadata(retrieved_chunks=0),
    )
    pipeline = FakePipeline(
        {
            answerable.question: RAGExecution(
                result=answer_result,
                retrieved_chunks=(chunk,),
            ),
            insufficient.question: RAGExecution(
                result=unavailable_result,
                retrieved_chunks=(),
            ),
        }
    )

    results = asyncio.run(
        run_benchmark(
            dataset=EvaluationDataset(cases=(answerable, insufficient)),
            pipelines={"dense": pipeline},
            scorer=FakeScorer(),
            dataset_path=dataset_path,
            answer_provider="ollama",
            answer_model="answer-model",
            judge_provider="ollama",
            generated_at=datetime(2026, 8, 27, tzinfo=timezone.utc),
        )
    )

    assert pipeline.calls == ["Answerable?", "Unavailable?"]
    assert results.dataset_sha256 == sha256_file(dataset_path)
    assert results.generated_at.isoformat() == "2026-08-27T00:00:00+00:00"
    assert results.answer_provider == "ollama"
    assert results.judge_provider == "ollama"
    configuration = results.configurations[0]
    assert configuration.expected_source_hit_rate == 1.0
    assert configuration.answerability_accuracy == 1.0
    assert configuration.metrics["faithfulness"].mean == 0.7
    assert configuration.metrics["faithfulness"].scored == 2
    assert configuration.cases[0].retrieved_contexts[0].text == "Grounded evidence."
    assert configuration.cases[1].expected_source_retrieved is None
    assert not has_metric_failures(results)

    output_path = tmp_path / "results.json"
    write_benchmark_results(results, output_path)
    stored = BenchmarkResults.model_validate_json(
        output_path.read_text(encoding="utf-8")
    )
    assert stored == results
    assert json.loads(output_path.read_text(encoding="utf-8"))["schema_version"] == "1.2"


def test_runner_records_pipeline_failure_and_continues(tmp_path: Path) -> None:
    dataset_path = tmp_path / "questions.jsonl"
    dataset_path.write_text("reviewed benchmark\n", encoding="utf-8")
    failed_case = EvaluationCase(
        case_id="eval_001",
        question="Broken answer?",
        expected_answer="Expected.",
        expected_source="guide.md",
        expected_section="Section",
        category="semantic",
        should_answer=True,
    )

    class FailingPipeline:
        def answer_with_trace(self, question: str) -> RAGExecution:
            raise RAGPipelineError("Answer source attribution failed.")

    results = asyncio.run(
        run_benchmark(
            dataset=EvaluationDataset(cases=(failed_case,)),
            pipelines={"dense": FailingPipeline()},
            scorer=FakeScorer(),
            dataset_path=dataset_path,
            answer_provider="ollama",
            answer_model="answer-model",
            judge_provider="ollama",
            generated_at=datetime(2026, 8, 27, tzinfo=timezone.utc),
        )
    )

    configuration = results.configurations[0]
    case = configuration.cases[0]
    assert configuration.application_failures == 1
    assert case.application_status == "failed"
    assert case.response == APPLICATION_FAILURE_RESPONSE
    assert case.expected_source_retrieved is False
    assert case.answerability_correct is False
    assert case.metrics["faithfulness"].status == "failed"
    assert "source attribution" in case.application_error.lower()
    assert has_metric_failures(results)


def test_pipeline_factory_uses_same_top_k_for_selected_configurations() -> None:
    retrievers = FakeRetrieverSuite()
    llm = object()

    pipelines = create_evaluation_pipelines(
        retrievers=retrievers,
        llm=llm,
        configurations=("dense", "hybrid_reranked"),
        top_k=5,
    )

    assert retrievers.requested == ["dense", "hybrid_reranked"]
    assert pipelines["dense"].top_k == 5
    assert pipelines["hybrid_reranked"].top_k == 5
    assert pipelines["dense"].llm is llm
    assert pipelines["hybrid_reranked"].retrieval_method == "hybrid_reranked"


def test_pipeline_factory_rejects_duplicate_configurations() -> None:
    with pytest.raises(ValueError, match="must be unique"):
        create_evaluation_pipelines(
            retrievers=FakeRetrieverSuite(),
            llm=object(),
            configurations=("dense", "dense"),
            top_k=5,
        )
