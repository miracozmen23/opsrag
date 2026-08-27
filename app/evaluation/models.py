"""Validated models for the manually reviewed RAG benchmark."""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from app.rag.prompts import INSUFFICIENT_CONTEXT_ANSWER

EvaluationCategory = Literal[
    "semantic",
    "exact_keyword",
    "error_code",
    "multi_sentence",
    "ambiguous",
    "insufficient_context",
]


class EvaluationCase(BaseModel):
    """One question with its human-reviewed reference answer and evidence."""

    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    case_id: str = Field(pattern=r"^eval_[0-9]{3}$")
    question: str = Field(min_length=1, max_length=2000)
    expected_answer: str = Field(min_length=1)
    expected_source: str | None
    expected_section: str | None
    supporting_sources: tuple[str, ...] = ()
    category: EvaluationCategory
    should_answer: bool

    @model_validator(mode="after")
    def validate_evidence_contract(self) -> "EvaluationCase":
        if self.should_answer:
            if self.category == "insufficient_context":
                raise ValueError("Answerable cases cannot use insufficient_context.")
            if not self.expected_source or not self.expected_section:
                raise ValueError(
                    "Answerable cases require expected_source and expected_section."
                )
        else:
            if self.category != "insufficient_context":
                raise ValueError(
                    "Unanswerable cases must use the insufficient_context category."
                )
            if self.expected_source is not None or self.expected_section is not None:
                raise ValueError("Unanswerable cases cannot declare expected evidence.")
            if self.supporting_sources:
                raise ValueError("Unanswerable cases cannot declare supporting sources.")
            if self.expected_answer != INSUFFICIENT_CONTEXT_ANSWER:
                raise ValueError(
                    "Unanswerable cases must use the canonical insufficient-context answer."
                )

        if len(set(self.supporting_sources)) != len(self.supporting_sources):
            raise ValueError("Supporting sources must be unique.")
        if self.expected_source in self.supporting_sources:
            raise ValueError("Primary source cannot be repeated as a supporting source.")
        return self


class EvaluationDataset(BaseModel):
    """One ordered, duplicate-free benchmark dataset."""

    model_config = ConfigDict(frozen=True)

    cases: tuple[EvaluationCase, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_uniqueness(self) -> "EvaluationDataset":
        case_ids = [case.case_id for case in self.cases]
        if len(set(case_ids)) != len(case_ids):
            raise ValueError("Evaluation case IDs must be unique.")

        normalized_questions = [
            " ".join(case.question.casefold().split()) for case in self.cases
        ]
        if len(set(normalized_questions)) != len(normalized_questions):
            raise ValueError("Evaluation questions must be unique.")
        return self


class EvaluationReport(BaseModel):
    """Deterministic summary returned by benchmark validation."""

    model_config = ConfigDict(frozen=True)

    cases: int = Field(ge=0)
    answerable: int = Field(ge=0)
    insufficient_context: int = Field(ge=0)
    category_counts: dict[str, int]
    source_counts: dict[str, int]
