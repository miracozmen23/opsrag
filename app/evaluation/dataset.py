"""Load and validate the version-controlled evaluation JSONL dataset."""

from collections import Counter
from pathlib import Path

from app.evaluation.models import (
    EvaluationCase,
    EvaluationCategory,
    EvaluationDataset,
    EvaluationReport,
)
from app.ingestion.loader import load_documents

MIN_BENCHMARK_CASES = 30
MAX_BENCHMARK_CASES = 50
REQUIRED_CATEGORIES: tuple[EvaluationCategory, ...] = (
    "semantic",
    "exact_keyword",
    "error_code",
    "multi_sentence",
    "ambiguous",
    "insufficient_context",
)


def read_evaluation_jsonl(input_path: Path) -> EvaluationDataset:
    """Read ordered UTF-8 JSONL cases with line-specific validation errors."""

    if not input_path.is_file():
        raise ValueError(f"Evaluation dataset does not exist: {input_path}")

    cases: list[EvaluationCase] = []
    for line_number, line in enumerate(
        input_path.read_text(encoding="utf-8").splitlines(),
        start=1,
    ):
        if not line.strip():
            continue
        try:
            cases.append(EvaluationCase.model_validate_json(line))
        except Exception as exc:
            raise ValueError(
                f"Invalid evaluation JSONL at line {line_number}: {exc}"
            ) from exc
    return EvaluationDataset(cases=tuple(cases))


def validate_evaluation_dataset(
    input_path: Path,
    source_dir: Path,
) -> EvaluationReport:
    """Validate benchmark size, category coverage, and real source metadata."""

    dataset = read_evaluation_jsonl(input_path)
    case_count = len(dataset.cases)
    if not MIN_BENCHMARK_CASES <= case_count <= MAX_BENCHMARK_CASES:
        raise ValueError(
            "Evaluation dataset must contain "
            f"{MIN_BENCHMARK_CASES}-{MAX_BENCHMARK_CASES} cases; got {case_count}."
        )

    load_result = load_documents(source_dir)
    if load_result.failures:
        failed_sources = ", ".join(failure.source for failure in load_result.failures)
        raise ValueError(f"Knowledge-base sources could not be loaded: {failed_sources}")
    sections_by_source = {
        document.source: {section.title for section in document.sections}
        for document in load_result.documents
    }

    category_counts = Counter(case.category for case in dataset.cases)
    missing_categories = [
        category for category in REQUIRED_CATEGORIES if category_counts[category] == 0
    ]
    if missing_categories:
        raise ValueError(
            "Evaluation dataset is missing required categories: "
            + ", ".join(missing_categories)
        )

    source_counts: Counter[str] = Counter()
    for case in dataset.cases:
        if not case.should_answer:
            continue
        source = case.expected_source
        section = case.expected_section
        if source is None or section is None:
            raise ValueError(f"{case.case_id} is missing expected evidence metadata.")
        _validate_source(case.case_id, source, sections_by_source)
        if section not in sections_by_source[source]:
            raise ValueError(
                f"{case.case_id} references unknown section "
                f"'{section}' in '{source}'."
            )
        source_counts[source] += 1
        for supporting_source in case.supporting_sources:
            _validate_source(case.case_id, supporting_source, sections_by_source)
            source_counts[supporting_source] += 1

    uncovered_sources = sorted(set(sections_by_source) - set(source_counts))
    if uncovered_sources:
        raise ValueError(
            "Evaluation dataset does not cover knowledge-base sources: "
            + ", ".join(uncovered_sources)
        )

    answerable = sum(case.should_answer for case in dataset.cases)
    return EvaluationReport(
        cases=case_count,
        answerable=answerable,
        insufficient_context=case_count - answerable,
        category_counts={
            category: category_counts[category] for category in REQUIRED_CATEGORIES
        },
        source_counts=dict(sorted(source_counts.items())),
    )


def _validate_source(
    case_id: str,
    source: str,
    sections_by_source: dict[str, set[str]],
) -> None:
    if source not in sections_by_source:
        raise ValueError(f"{case_id} references unknown source '{source}'.")
