"""Run the fixed benchmark against traceable production RAG pipelines."""

import hashlib
import logging
from collections.abc import Mapping
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter
from typing import Protocol

from app.evaluation.models import EvaluationCase, EvaluationDataset
from app.evaluation.results import (
    METRIC_NAMES,
    BenchmarkResults,
    ConfigurationResult,
    EvaluationCaseResult,
    EvaluationConfiguration,
    MetricName,
    MetricOutcome,
    MetricSummary,
    RetrievedEvidence,
)
from app.llm.base import LanguageModel
from app.rag.models import RAGExecution
from app.rag.pipeline import RAGPipeline, RAGPipelineError
from app.rag.prompts import INSUFFICIENT_CONTEXT_ANSWER
from app.retrieval.factory import RetrieverSuite

logger = logging.getLogger(__name__)
APPLICATION_FAILURE_RESPONSE = "Application failed before producing a valid answer."


class EvaluationScorer(Protocol):
    """Async metric boundary implemented by the optional RAGAS adapter."""

    ragas_version: str
    judge_model: str
    embedding_model: str

    async def score(
        self,
        *,
        question: str,
        response: str,
        reference: str,
        retrieved_contexts: list[str],
    ) -> dict[MetricName, MetricOutcome]:
        """Return all required metric outcomes for one application response."""

        ...


class TraceableRAGPipeline(Protocol):
    """Minimal application boundary needed by offline evaluation."""

    def answer_with_trace(self, question: str) -> RAGExecution:
        """Return the generated answer and exact ranked retrieval contexts."""

        ...


async def run_benchmark(
    *,
    dataset: EvaluationDataset,
    pipelines: Mapping[EvaluationConfiguration, TraceableRAGPipeline],
    scorer: EvaluationScorer,
    dataset_path: Path,
    answer_provider: str,
    answer_model: str,
    judge_provider: str,
    generated_at: datetime | None = None,
) -> BenchmarkResults:
    """Evaluate every case for every configured retrieval path."""

    if not pipelines:
        raise ValueError("At least one evaluation pipeline is required.")
    configuration_results: list[ConfigurationResult] = []
    for configuration, pipeline in pipelines.items():
        logger.info(
            "evaluation_configuration_started configuration=%s cases=%d",
            configuration,
            len(dataset.cases),
        )
        configuration_results.append(
            await _evaluate_configuration(
                configuration=configuration,
                cases=dataset.cases,
                pipeline=pipeline,
                scorer=scorer,
            )
        )

    timestamp = generated_at or datetime.now(timezone.utc)
    if timestamp.tzinfo is None:
        raise ValueError("Benchmark generated_at must include a timezone.")
    return BenchmarkResults(
        generated_at=timestamp,
        dataset_path=dataset_path.as_posix(),
        dataset_sha256=sha256_file(dataset_path),
        dataset_case_count=len(dataset.cases),
        ragas_version=scorer.ragas_version,
        answer_provider=answer_provider,
        answer_model=answer_model,
        judge_provider=judge_provider,
        judge_model=scorer.judge_model,
        embedding_model=scorer.embedding_model,
        configurations=tuple(configuration_results),
    )


def create_evaluation_pipelines(
    *,
    retrievers: RetrieverSuite,
    llm: LanguageModel,
    configurations: tuple[EvaluationConfiguration, ...],
    top_k: int,
) -> dict[EvaluationConfiguration, RAGPipeline]:
    """Wrap selected shared retrievers with identical generation settings."""

    if top_k < 1:
        raise ValueError("Evaluation context count must be at least 1.")
    if not configurations:
        raise ValueError("At least one retrieval configuration is required.")
    if len(set(configurations)) != len(configurations):
        raise ValueError("Evaluation retrieval configurations must be unique.")
    return {
        configuration: RAGPipeline(
            retrievers.get(configuration),
            llm,
            top_k=top_k,
            retrieval_method=configuration,
        )
        for configuration in configurations
    }


def write_benchmark_results(results: BenchmarkResults, output_path: Path) -> None:
    """Atomically replace a results artifact after full model validation."""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_suffix(output_path.suffix + ".tmp")
    temporary_path.write_text(
        results.model_dump_json(indent=2) + "\n",
        encoding="utf-8",
    )
    temporary_path.replace(output_path)


def sha256_file(path: Path) -> str:
    """Return the content hash that binds results to the reviewed dataset."""

    if not path.is_file():
        raise ValueError(f"Cannot hash missing evaluation dataset: {path}")
    return hashlib.sha256(path.read_bytes()).hexdigest()


def has_metric_failures(results: BenchmarkResults) -> bool:
    """Report whether any real metric invocation failed."""

    return any(
        outcome.status == "failed"
        for configuration in results.configurations
        for case in configuration.cases
        for outcome in case.metrics.values()
    )


async def _evaluate_configuration(
    *,
    configuration: EvaluationConfiguration,
    cases: tuple[EvaluationCase, ...],
    pipeline: TraceableRAGPipeline,
    scorer: EvaluationScorer,
) -> ConfigurationResult:
    results: list[EvaluationCaseResult] = []
    for index, case in enumerate(cases, start=1):
        started_at = perf_counter()
        try:
            execution = pipeline.answer_with_trace(case.question)
        except RAGPipelineError as exc:
            application_latency_ms = (perf_counter() - started_at) * 1000
            error = _format_application_error(exc)
            failed_metrics = {
                name: MetricOutcome(
                    status="failed",
                    error=f"Application failure prevented scoring: {error}",
                )
                for name in METRIC_NAMES
            }
            results.append(
                EvaluationCaseResult(
                    case_id=case.case_id,
                    category=case.category,
                    question=case.question,
                    should_answer=case.should_answer,
                    expected_answer=case.expected_answer,
                    expected_source=case.expected_source,
                    expected_section=case.expected_section,
                    application_status="failed",
                    application_error=error,
                    response=APPLICATION_FAILURE_RESPONSE,
                    retrieved_contexts=(),
                    cited_sources=(),
                    expected_source_retrieved=False if case.should_answer else None,
                    answerability_correct=False,
                    retrieval_confidence=0.0,
                    latency_ms=round(application_latency_ms, 3),
                    metrics=failed_metrics,
                )
            )
            logger.warning(
                "evaluation_case_failed configuration=%s case_id=%s progress=%d/%d "
                "latency_ms=%.2f error=%s",
                configuration,
                case.case_id,
                index,
                len(cases),
                application_latency_ms,
                error,
            )
            continue
        application_latency_ms = (perf_counter() - started_at) * 1000
        contexts = [chunk.text for chunk in execution.retrieved_chunks]
        metrics = await scorer.score(
            question=case.question,
            response=execution.result.answer,
            reference=case.expected_answer,
            retrieved_contexts=contexts,
        )
        evidence = tuple(
            RetrievedEvidence(
                rank=rank,
                chunk_id=chunk.metadata.chunk_id,
                source=chunk.metadata.source,
                section=chunk.metadata.section,
                text=chunk.text,
            )
            for rank, chunk in enumerate(execution.retrieved_chunks, start=1)
        )
        result = execution.result
        results.append(
            EvaluationCaseResult(
                case_id=case.case_id,
                category=case.category,
                question=case.question,
                should_answer=case.should_answer,
                expected_answer=case.expected_answer,
                expected_source=case.expected_source,
                expected_section=case.expected_section,
                response=result.answer,
                retrieved_contexts=evidence,
                cited_sources=tuple(source.document for source in result.sources),
                expected_source_retrieved=_expected_source_retrieved(case, evidence),
                answerability_correct=_answerability_correct(case, result.answer),
                retrieval_confidence=result.retrieval_confidence,
                latency_ms=round(application_latency_ms, 3),
                metrics=metrics,
            )
        )
        logger.info(
            "evaluation_case_completed configuration=%s case_id=%s progress=%d/%d "
            "latency_ms=%.2f",
            configuration,
            case.case_id,
            index,
            len(cases),
            application_latency_ms,
        )

    answerable_results = [result for result in results if result.should_answer]
    source_hits = sum(
        result.expected_source_retrieved is True for result in answerable_results
    )
    return ConfigurationResult(
        configuration=configuration,
        case_count=len(results),
        application_failures=sum(
            result.application_status == "failed" for result in results
        ),
        expected_source_hit_rate=round(source_hits / len(answerable_results), 4),
        answerability_accuracy=round(
            sum(result.answerability_correct for result in results) / len(results),
            4,
        ),
        mean_latency_ms=round(
            sum(result.latency_ms for result in results) / len(results),
            3,
        ),
        metrics=_summarize_metrics(results),
        cases=tuple(results),
    )


def _format_application_error(error: BaseException) -> str:
    parts: list[str] = []
    current: BaseException | None = error
    while current is not None and len(parts) < 3:
        parts.append(f"{type(current).__name__}: {current}")
        current = current.__cause__
    return " <- ".join(parts)[:1000]


def _expected_source_retrieved(
    case: EvaluationCase,
    evidence: tuple[RetrievedEvidence, ...],
) -> bool | None:
    if not case.should_answer:
        return None
    return any(
        item.source == case.expected_source and item.section == case.expected_section
        for item in evidence
    )


def _answerability_correct(case: EvaluationCase, response: str) -> bool:
    refused = response.strip() == INSUFFICIENT_CONTEXT_ANSWER
    return refused != case.should_answer


def _summarize_metrics(
    cases: list[EvaluationCaseResult],
) -> dict[MetricName, MetricSummary]:
    summaries: dict[MetricName, MetricSummary] = {}
    for metric_name in METRIC_NAMES:
        outcomes = [case.metrics[metric_name] for case in cases]
        values = [
            outcome.value
            for outcome in outcomes
            if outcome.status == "scored" and outcome.value is not None
        ]
        summaries[metric_name] = MetricSummary(
            mean=round(sum(values) / len(values), 4) if values else None,
            scored=sum(outcome.status == "scored" for outcome in outcomes),
            undefined=sum(outcome.status == "undefined" for outcome in outcomes),
            failed=sum(outcome.status == "failed" for outcome in outcomes),
        )
    return summaries
