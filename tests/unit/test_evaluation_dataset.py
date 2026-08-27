"""Evaluation dataset schema and repository benchmark validation."""

import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from app.evaluation.dataset import (
    REQUIRED_CATEGORIES,
    read_evaluation_jsonl,
    validate_evaluation_dataset,
)
from app.evaluation.models import EvaluationCase, EvaluationDataset
from app.rag.prompts import INSUFFICIENT_CONTEXT_ANSWER

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def test_repository_benchmark_is_balanced_and_source_valid() -> None:
    report = validate_evaluation_dataset(
        PROJECT_ROOT / "evaluation" / "questions.jsonl",
        PROJECT_ROOT / "data" / "raw",
    )
    assert report.cases == 36
    assert report.answerable == 30
    assert report.insufficient_context == 6
    assert report.category_counts == {
        category: 6 for category in REQUIRED_CATEGORIES
    }
    assert set(report.source_counts) == {
        "api_error_handling.md",
        "docker_troubleshooting.md",
        "fastapi_deployment.md",
        "logging_and_environment.md",
        "postgresql_troubleshooting.md",
    }


def test_repository_case_ids_are_sequential_for_manual_review() -> None:
    dataset = read_evaluation_jsonl(
        PROJECT_ROOT / "evaluation" / "questions.jsonl"
    )
    assert [case.case_id for case in dataset.cases] == [
        f"eval_{index:03d}" for index in range(1, 37)
    ]


def test_unanswerable_case_requires_canonical_empty_evidence() -> None:
    with pytest.raises(ValidationError, match="canonical"):
        EvaluationCase(
            case_id="eval_999",
            question="Unknown topic?",
            expected_answer="I do not know.",
            expected_source=None,
            expected_section=None,
            category="insufficient_context",
            should_answer=False,
        )

    valid = EvaluationCase(
        case_id="eval_999",
        question="Unknown topic?",
        expected_answer=INSUFFICIENT_CONTEXT_ANSWER,
        expected_source=None,
        expected_section=None,
        category="insufficient_context",
        should_answer=False,
    )
    assert valid.supporting_sources == ()


def test_answerable_case_requires_source_and_section() -> None:
    with pytest.raises(ValidationError, match="require expected_source"):
        EvaluationCase(
            case_id="eval_999",
            question="How should this work?",
            expected_answer="Use the documented setting.",
            expected_source=None,
            expected_section=None,
            category="semantic",
            should_answer=True,
        )


def test_dataset_rejects_duplicate_questions() -> None:
    case = EvaluationCase(
        case_id="eval_001",
        question="How should this work?",
        expected_answer="Use the documented setting.",
        expected_source="guide.md",
        expected_section="Setup",
        category="semantic",
        should_answer=True,
    )
    with pytest.raises(ValidationError, match="questions must be unique"):
        EvaluationDataset(
            cases=(case, case.model_copy(update={"case_id": "eval_002"}))
        )


def test_reader_reports_invalid_jsonl_line(tmp_path: Path) -> None:
    path = tmp_path / "invalid.jsonl"
    path.write_text(
        json.dumps(
            {
                "case_id": "eval_001",
                "question": "question",
                "expected_answer": "answer",
                "expected_source": "guide.md",
                "expected_section": "Setup",
                "category": "semantic",
                "should_answer": True,
            }
        )
        + "\n{not json}\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="line 2"):
        read_evaluation_jsonl(path)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("expected_source", "invented.md", "unknown source"),
        ("expected_section", "Invented section", "unknown section"),
    ],
)
def test_validator_rejects_invented_evidence_metadata(
    tmp_path: Path,
    field: str,
    value: str,
    message: str,
) -> None:
    source_path = PROJECT_ROOT / "evaluation" / "questions.jsonl"
    records = [
        json.loads(line)
        for line in source_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    records[0][field] = value
    modified_path = tmp_path / "questions.jsonl"
    modified_path.write_text(
        "\n".join(json.dumps(record) for record in records) + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match=message):
        validate_evaluation_dataset(modified_path, PROJECT_ROOT / "data" / "raw")


def test_dataset_detects_duplicate_questions_with_whitespace_variation() -> None:
    base = EvaluationCase(
        case_id="eval_001",
        question="How should this work?",
        expected_answer="Use the documented setting.",
        expected_source="guide.md",
        expected_section="Setup",
        category="semantic",
        should_answer=True,
    )
    duplicate = base.model_copy(
        update={"case_id": "eval_002", "question": "How  should this   work?"}
    )
    with pytest.raises(ValidationError, match="questions must be unique"):
        EvaluationDataset(cases=(base, duplicate))
