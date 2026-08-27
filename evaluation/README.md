# Evaluation Dataset

`questions.jsonl` is the manually reviewed benchmark for OpsRAG. It is intentionally data-only: validating it does not run retrieval, contact Qdrant, call an LLM, or report model-quality scores.

## Current coverage

- 36 total cases
- 30 answerable cases grounded in the knowledge base
- 6 insufficient-context cases
- 6 cases in each required category
- all 5 current knowledge-base documents covered

The required categories are:

| Category | Purpose |
| --- | --- |
| `semantic` | Tests paraphrases that do not depend on exact wording. |
| `exact_keyword` | Tests commands, configuration names, and technical terms. |
| `error_code` | Tests exact HTTP, database, Docker, or application error signals. |
| `multi_sentence` | Tests questions that combine related symptoms or constraints. |
| `ambiguous` | Tests underspecified technical questions that still have a safe, grounded answer. |
| `insufficient_context` | Tests questions whose answer is absent from the knowledge base. |

## Case schema

Each non-empty line in `questions.jsonl` is one JSON object:

- `case_id`: stable sequential identifier in the form `eval_NNN`
- `question`: unique benchmark question
- `expected_answer`: manually written reference answer
- `expected_source`: primary knowledge-base filename, or `null` when unanswerable
- `expected_section`: exact primary section title, or `null` when unanswerable
- `supporting_sources`: optional additional filenames needed by the reference answer
- `category`: one of the six categories above
- `should_answer`: whether the knowledge base contains enough evidence

Answerable cases require a real primary source and exact section title. Unanswerable cases must use `category="insufficient_context"`, declare no sources, and use this canonical answer:

```text
The knowledge base does not contain enough context to answer this question.
```

## Validate and review

From the repository root, run:

```bash
python scripts/validate_evaluation.py
```

Validation rejects malformed JSON, duplicate IDs or normalized questions, datasets outside the 30–50 case range, missing categories, unknown source files, unknown section titles, uncovered knowledge-base documents, and invalid answerability contracts. The reported source counts include primary and supporting-source references.

Manual review should also confirm that each question is realistic, the expected answer says no more than its cited evidence supports, supporting sources are genuinely necessary, and insufficient-context questions are truly outside the current corpus.

## Add or revise a case

1. Read the relevant file under `data/raw` and copy its section title exactly.
2. Add or edit one single-line JSON object while keeping `case_id` values unique and ordered.
3. Keep the reference answer concise and fully supported by the declared evidence.
4. Run the dataset validator and the complete test suite.

Performance metrics are intentionally absent here. Milestone 11 will consume this frozen benchmark to evaluate system answers without changing its reference data during measurement.
