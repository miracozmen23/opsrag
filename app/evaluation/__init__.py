"""Evaluation dataset contracts and validation helpers."""

from app.evaluation.dataset import read_evaluation_jsonl, validate_evaluation_dataset
from app.evaluation.models import EvaluationCase, EvaluationDataset, EvaluationReport
from app.evaluation.results import BenchmarkResults, MetricOutcome

__all__ = [
    "EvaluationCase",
    "EvaluationDataset",
    "EvaluationReport",
    "BenchmarkResults",
    "MetricOutcome",
    "read_evaluation_jsonl",
    "validate_evaluation_dataset",
]
