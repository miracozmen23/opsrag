# Evaluation Dataset

`questions.jsonl` is the manually reviewed benchmark dataset for OpsRAG. Dataset validation is intentionally offline: it does not run retrieval, contact Qdrant, call an LLM, or report model-quality scores. The separate benchmark runner writes real measurements to `results.json`.

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

## Run the benchmark

Start Qdrant, build the index, and configure the answer and judge providers in `.env`. For a no-API-charge baseline, use a local Ollama model for both:

```env
LLM_PROVIDER=ollama
LLM_MODEL=qwen3.5:2b
OLLAMA_BASE_URL=http://localhost:11434
RAGAS_JUDGE_PROVIDER=ollama
RAGAS_JUDGE_MODEL=qwen3.5:2b
```

Then run all three retrieval configurations:

```bash
python scripts/evaluate.py
```

Use `--configurations dense`, `hybrid`, or `hybrid_reranked` for a targeted run. If either configured provider is OpenAI, the command requires `--confirm-paid-run` before making any requests.

## Results contract

`results.json` includes a dataset SHA-256 hash, generation timestamp, RAGAS version, embedding model, answer and judge provider/model names, and one complete section per retrieval configuration. Each case preserves:

- the real generated answer and retrieved evidence;
- cited sources, latency, retrieval confidence, and expected-source hit;
- answerability correctness and any application error;
- faithfulness, answer relevance, context precision, and context recall outcomes.

Each metric outcome is `scored`, `undefined`, or `failed`. Aggregate means use only finite scored values and publish their scored/undefined/failed counts. Application failures also remain explicit and do not prevent subsequent cases from running. This makes partial local-model limitations auditable instead of silently treating them as zero-quality answers.

## Checked-in free baseline

`results.json` was generated on 2026-08-27 from all 108 case/configuration executions. Both answer generation and judging used Ollama `qwen3.5:2b`; RAGAS was 0.4.3 and embeddings were local `BAAI/bge-small-en-v1.5`. No paid OpenAI request was used.

Deterministic and runtime measurements:

| Retrieval | App failures | Expected-source hit | Answerability accuracy | Mean app latency |
| --- | ---: | ---: | ---: | ---: |
| Dense | 3/36 | 0.9333 | 0.8889 | 6,559.567 ms |
| Hybrid | 3/36 | 0.9000 | 0.8889 | 6,323.569 ms |
| Hybrid + reranking | 3/36 | 0.8667 | 0.9167 | 10,583.188 ms |

RAGAS means show their valid score coverage in parentheses:

| Retrieval | Faithfulness | Answer relevance | Context precision | Context recall |
| --- | ---: | ---: | ---: | ---: |
| Dense | 0.5455 (11/36) | 0.7424 (33/36) | 0.8215 (33/36) | 0.8177 (32/36) |
| Hybrid | 0.6026 (13/36) | 0.7285 (33/36) | 0.7841 (33/36) | 0.7903 (31/36) |
| Hybrid + reranking | 0.5000 (11/36) | 0.7452 (33/36) | 0.7747 (33/36) | 0.7833 (30/36) |

The local run used a speed-oriented 256-token judge limit. Most missing faithfulness scores were incomplete structured outputs, while application failures came from strict citation validation. Because faithfulness coverage is only 11–13 cases per configuration, its means must not be used to rank the retrieval strategies. The JSON artifact retains every exact case outcome and error for later audit.

This is a captured free baseline for its recorded provider/model combination. Small local judge models trade evaluation reliability for zero API cost; compare runs only after checking the provenance, runtime profile, and metric outcome counts.
