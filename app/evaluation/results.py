"""Serializable contracts for real benchmark executions and metric summaries."""

import math
from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from app.evaluation.models import EvaluationCategory

EvaluationConfiguration = Literal["dense", "hybrid", "hybrid_reranked"]
MetricName = Literal[
    "faithfulness",
    "answer_relevance",
    "context_precision",
    "context_recall",
]
METRIC_NAMES: tuple[MetricName, ...] = (
    "faithfulness",
    "answer_relevance",
    "context_precision",
    "context_recall",
)


class MetricOutcome(BaseModel):
    """One metric value without disguising undefined or failed scoring as zero."""

    model_config = ConfigDict(frozen=True)

    status: Literal["scored", "undefined", "failed"]
    value: float | None = None
    reason: str | None = None
    error: str | None = None

    @model_validator(mode="after")
    def validate_status_contract(self) -> "MetricOutcome":
        if self.status == "scored":
            if self.value is None or not math.isfinite(self.value):
                raise ValueError("Scored metrics require one finite value.")
            if self.error is not None:
                raise ValueError("Scored metrics cannot include an error.")
        elif self.status == "undefined":
            if self.value is not None or self.error is not None:
                raise ValueError("Undefined metrics cannot include a value or error.")
        else:
            if self.value is not None or not self.error:
                raise ValueError("Failed metrics require an error and no value.")
        return self


class RetrievedEvidence(BaseModel):
    """One ranked chunk supplied to generation and metric evaluation."""

    model_config = ConfigDict(frozen=True)

    rank: int = Field(ge=1)
    chunk_id: str = Field(min_length=1)
    source: str = Field(min_length=1)
    section: str = Field(min_length=1)
    text: str = Field(min_length=1)


class EvaluationCaseResult(BaseModel):
    """Real application output and metric outcomes for one benchmark case."""

    model_config = ConfigDict(frozen=True)

    case_id: str = Field(pattern=r"^eval_[0-9]{3}$")
    category: EvaluationCategory
    question: str = Field(min_length=1)
    should_answer: bool
    expected_answer: str = Field(min_length=1)
    expected_source: str | None
    expected_section: str | None
    response: str = Field(min_length=1)
    retrieved_contexts: tuple[RetrievedEvidence, ...]
    cited_sources: tuple[str, ...]
    expected_source_retrieved: bool | None
    answerability_correct: bool
    retrieval_confidence: float = Field(ge=0.0, le=1.0)
    latency_ms: float = Field(ge=0.0)
    metrics: dict[MetricName, MetricOutcome]

    @model_validator(mode="after")
    def validate_metric_coverage(self) -> "EvaluationCaseResult":
        if set(self.metrics) != set(METRIC_NAMES):
            raise ValueError("Every case must contain all four evaluation metrics.")
        if self.should_answer and self.expected_source_retrieved is None:
            raise ValueError("Answerable cases require a source-hit result.")
        if not self.should_answer and self.expected_source_retrieved is not None:
            raise ValueError(
                "Unanswerable cases cannot claim an expected-source hit result."
            )
        return self


class MetricSummary(BaseModel):
    """Aggregate of only valid finite scores for one metric."""

    model_config = ConfigDict(frozen=True)

    mean: float | None
    scored: int = Field(ge=0)
    undefined: int = Field(ge=0)
    failed: int = Field(ge=0)


class ConfigurationResult(BaseModel):
    """All real outputs and summaries for one retrieval configuration."""

    model_config = ConfigDict(frozen=True)

    configuration: EvaluationConfiguration
    case_count: int = Field(ge=1)
    expected_source_hit_rate: float = Field(ge=0.0, le=1.0)
    answerability_accuracy: float = Field(ge=0.0, le=1.0)
    mean_latency_ms: float = Field(ge=0.0)
    metrics: dict[MetricName, MetricSummary]
    cases: tuple[EvaluationCaseResult, ...]

    @model_validator(mode="after")
    def validate_case_count(self) -> "ConfigurationResult":
        if self.case_count != len(self.cases):
            raise ValueError("Configuration case_count must match stored cases.")
        if set(self.metrics) != set(METRIC_NAMES):
            raise ValueError("Configuration summary must contain all four metrics.")
        return self


class BenchmarkResults(BaseModel):
    """Versioned top-level output written only from a real benchmark execution."""

    model_config = ConfigDict(frozen=True)

    schema_version: Literal["1.0"] = "1.0"
    generated_at: datetime
    dataset_path: str = Field(min_length=1)
    dataset_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    dataset_case_count: int = Field(ge=1)
    ragas_version: str = Field(min_length=1)
    answer_model: str = Field(min_length=1)
    judge_model: str = Field(min_length=1)
    embedding_model: str = Field(min_length=1)
    configurations: tuple[ConfigurationResult, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_configurations(self) -> "BenchmarkResults":
        names = [result.configuration for result in self.configurations]
        if len(names) != len(set(names)):
            raise ValueError("Benchmark configurations must be unique.")
        if any(
            result.case_count != self.dataset_case_count
            for result in self.configurations
        ):
            raise ValueError("Every configuration must evaluate the complete dataset.")
        return self
