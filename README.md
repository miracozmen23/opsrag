# OpsRAG

OpsRAG is a compact technical knowledge assistant built incrementally as a production-oriented RAG portfolio project. The current implementation routes requests through a bounded LangGraph workflow, runs technical questions through hybrid retrieval and cross-encoder reranking, answers clearly general messages directly without pretending retrieval occurred, evaluates real outputs, and can trace requests end to end with optional Langfuse observability.

## Current scope

Completed through **Milestone 14 — Dockerization**:

- FastAPI application with `GET /health`
- Markdown, TXT, and text-based PDF ingestion
- conservative cleaning and title/section preservation
- deterministic 600-token chunks with 75-token overlap
- configurable Sentence Transformers embedding service
- Qdrant collection creation, safe explicit recreation, and dense search
- deterministic in-memory BM25 indexing and lexical search for exact technical terms
- Reciprocal Rank Fusion over configurable dense and sparse candidate lists
- deterministic chunk-ID deduplication and normalized hybrid retrieval results
- configurable cross-encoder reranking of the hybrid candidate pool
- separate raw RRF and cross-encoder scores on internal retrieval results
- final RAG context selection from reranked chunks rather than raw retrieval order
- project-local model caching so model downloads stay with the repository drive
- grounded prompt with grouped source identifiers and chunk excerpts
- deterministic source deduplication by original document, section, and page
- human-readable source title, filename, section, page, and contributing chunk IDs
- strict rejection of missing, malformed, or invented LLM source identifiers
- API output limited to sources actually cited in the generated answer
- typed LangGraph state with classifier, RAG, and direct-answer nodes
- conditional `knowledge` and `general` branches with no loops or agent hierarchy
- conservative deterministic routing for English and Turkish general messages
- lazy knowledge-pipeline construction so general requests do not read chunks or contact Qdrant
- explicit route and retrieval-usage metadata on every API answer
- version-controlled benchmark with 36 manually reviewable questions and reference answers
- balanced coverage of semantic, exact-keyword, error-code, multi-sentence, ambiguous, and insufficient-context cases
- validation of every answerable case against real knowledge-base source and section metadata
- reproducible comparison of dense, hybrid, and hybrid-plus-reranking retrieval
- RAGAS faithfulness, answer relevance, context precision, and context recall scoring
- versioned per-case results with retrieved evidence, citations, latency, and model metadata
- explicit `scored`, `undefined`, and `failed` metric states instead of silently replacing failures with zero
- per-case application failure isolation so one invalid answer cannot discard a complete benchmark run
- local Ollama support for both answer generation and RAGAS judging without paid API calls
- an explicit command-line confirmation guard whenever OpenAI would be used by either role
- optional Langfuse SDK v4 tracing with no-op behavior when disabled or incompletely configured
- nested query, classification, retrieval, generation, and source-attribution observations
- trace payloads containing prompts, answers, chunk identifiers, retrieval/reranking scores, latency, and error states
- fail-open exporter isolation so tracing failures cannot break an answer request
- provider-neutral LLM protocol and OpenAI Responses API adapter
- `POST /api/v1/ask` returning answer, sources, retrieval metadata, and a retrieval-derived confidence heuristic
- responsive product-style Streamlit experience backed exclusively by the public FastAPI contract
- polished hero, example-question shortcuts, answer metrics, relevance bars, and evidence panels
- answer, cited-source, retrieval-confidence, route, and execution-metadata presentation
- configurable API URL and timeout with clear unavailable, timeout, and invalid-response errors
- one CPU-only application image shared by the FastAPI and Streamlit services
- health-gated Docker Compose topology for `qdrant`, `api`, and `frontend`
- repository-local model cache, document data, and Qdrant storage bind-mounted from the host drive
- non-root application containers, localhost-only published ports, and disabled Qdrant telemetry
- mock-based tests that do not require an API key or paid model call

## Docker quick start

Docker is the shortest clone-to-question path. Copy the environment template and
configure either the free local Ollama profile or an OpenAI profile first:

```powershell
copy .env.example .env
```

For the free local profile, keep Ollama running on the host and pull the selected
model before starting the containers:

```powershell
ollama pull qwen3.5:2b
docker compose up --build -d
```

Create the deterministic chunks and index them into the containerized Qdrant:

```powershell
docker compose run --rm api python scripts/ingest.py
docker compose run --rm api python scripts/index.py
```

Then open:

- Streamlit: `http://localhost:8501`
- FastAPI docs: `http://localhost:8000/docs`
- FastAPI health: `http://localhost:8000/health`

Inspect service health or follow logs:

```powershell
docker compose ps
docker compose logs -f api frontend
```

Stop the stack without deleting project data:

```powershell
docker compose down
```

Stop any manually started Uvicorn, Streamlit, or Qdrant process before Compose so
ports `8000`, `8501`, `6333`, and `6334` are available. Compose reaches a host
Ollama instance through `host.docker.internal`; an OpenAI configuration does not
require Ollama. On native Linux, Ollama may additionally need
`OLLAMA_HOST=0.0.0.0:11434` so containers can reach it.

The application image deliberately installs CPU-only PyTorch. Model downloads are
stored in the repository-local `.cache` directory, documents and chunks in
`data`, and Qdrant files in `qdrant_storage`. These bind mounts keep growing RAG
artifacts on the drive containing the repository instead of inside Docker's
virtual disk. All three paths are excluded from Git as appropriate.

## Local setup

Python 3.11 or newer is required.

```bash
python -m venv .venv
.venv\Scripts\activate
python -m pip install -e ".[dev]"
copy .env.example .env
```

Start Qdrant:

```bash
docker compose up -d qdrant
```

Build the local knowledge index:

```bash
python scripts/ingest.py
python scripts/index.py
```

BM25 search uses the same deterministic chunk artifact and does not require Qdrant:

```bash
python scripts/sparse_search.py "pg_hba.conf authentication"
```

Hybrid search requires the processed chunks and an indexed Qdrant collection:

```bash
python scripts/hybrid_search.py "HTTP 503 Qdrant"
```

Run the same hybrid flow with final cross-encoder reranking:

```bash
python scripts/rerank_search.py "When should an API return 4xx instead of 5xx?"
```

The first model-backed command downloads model files. `MODEL_CACHE_DIR` defaults to `.cache/huggingface`, and relative runtime paths are resolved against the repository root rather than the shell's current directory. On the current `D:\RAG` checkout, the embedding and reranker caches therefore remain on D:. The cache is excluded from Git.

Candidate and context sizes are independently configurable:

```env
TOP_K_DENSE=10
TOP_K_SPARSE=10
TOP_K_HYBRID=10
TOP_K_RERANK=5
RRF_K=60
```

`TOP_K_HYBRID` controls the cross-encoder candidate pool. `TOP_K_RERANK` controls how many of those chunks become final LLM context.

Collection replacement is never implicit. Use `python scripts/index.py --recreate` only when you intentionally want to delete and rebuild the configured collection.

For a fully local run with no per-request API charge, install Ollama, pull a model,
and configure both generation and evaluation to use it:

```bash
ollama pull qwen3.5:2b
```

```env
LLM_PROVIDER=ollama
LLM_MODEL=qwen3.5:2b
OLLAMA_BASE_URL=http://localhost:11434
RAGAS_JUDGE_PROVIDER=ollama
RAGAS_JUDGE_MODEL=qwen3.5:2b
```

`LLM_API_KEY` is not required in this profile. A stronger remote model can instead
be configured with OpenAI:

```env
LLM_PROVIDER=openai
LLM_MODEL=<an OpenAI model available to your project>
LLM_API_KEY=<your API key>
```

The model name is intentionally not hard-coded. Do not commit `.env`. Local model
quality and speed depend on the selected model and hardware; the small model above
is a cost-free baseline rather than a gold-standard evaluator.

Start the API:

```bash
uvicorn app.main:app --reload
```

## Streamlit demo

Keep the API running, then open a second terminal in the repository and start the
demo:

```bash
python -m streamlit run frontend/streamlit_app.py
```

Streamlit prints the local browser address, normally `http://localhost:8501`.
The responsive product demo includes selectable example prompts and sends each
submitted question to `POST /api/v1/ask`. It displays the validated answer,
retrieval confidence, route, context count, retrieval method, and cited source
metadata with source-level relevance indicators. It does not import or execute
the retrieval pipeline directly.

The default backend address and the longer local-model-friendly timeout can be
overridden in `.env`:

```env
OPSRAG_API_BASE_URL=http://localhost:8000
OPSRAG_API_TIMEOUT_SECONDS=300
```

If FastAPI is stopped, times out, or returns a response outside the published
schema, the page shows a safe error instead of an application traceback.

## API

Health:

```http
GET /health
```

Ask a knowledge-base question:

```http
POST /api/v1/ask
Content-Type: application/json

{"question":"Why does PostgreSQL return connection refused in Docker Compose?"}
```

Example response shape:

```json
{
  "answer": "Inside the application container, connect to the PostgreSQL service name and container port [S1].",
  "sources": [
    {
      "source_id": "S1",
      "document": "postgresql_troubleshooting.md",
      "title": "PostgreSQL Troubleshooting",
      "section": "Connection refused",
      "page_number": null,
      "score": 0.9491,
      "chunk_id": "chunk_...",
      "chunk_ids": ["chunk_..."]
    }
  ],
  "retrieval_confidence": 0.9491,
  "metadata": {
    "retrieved_chunks": 5,
    "cited_sources": 1,
    "retrieval_method": "hybrid_reranked",
    "route": "knowledge"
  }
}
```

Retrieved chunks from the same original document, title, section, and page share one source identifier. Multiple contributing chunks appear once through `chunk_ids`. After generation, OpsRAG validates every source-like token, rejects unknown or malformed identifiers, removes repeated citations, and returns only sources referenced by the answer. The API never accepts document names supplied by the LLM; all source metadata comes from retrieved chunks.

The cross-encoder emits raw logits. Internal retrieval results retain those logits as `rerank_score` and preserve the RRF score separately as `score`. Public source scores apply a sigmoid to the reranker logit. `retrieval_confidence` is the best score among the sources actually cited by the answer, producing a stable `[0, 1]` relevance heuristic. This value is not a calibrated probability and is not generated by the LLM.

### Query routing

The request workflow is deliberately small:

```text
question -> classify
              |-- knowledge -> existing grounded RAG pipeline -> answer
              `-- general   -> direct LLM, no retrieval      -> answer
```

The deterministic router sends greetings, thanks, identity/capability questions, jokes, and a small set of clear general-fact forms directly to the LLM. Everything ambiguous or technical defaults to `knowledge`, which is the safer route for a private technical assistant. There is no planner, supervisor, autonomous loop, or multi-agent system.

A general response makes retrieval absence explicit:

```json
{
  "answer": "Hello! How can I help?",
  "sources": [],
  "retrieval_confidence": 0.0,
  "metadata": {
    "retrieved_chunks": 0,
    "cited_sources": 0,
    "retrieval_method": "not_used",
    "route": "general"
  }
}
```

The knowledge pipeline is constructed lazily only after LangGraph selects that branch. A clearly general request therefore does not load the JSONL chunk corpus, build BM25, access Qdrant, or load embedding/reranker models. Both routes still require the configured LLM service.

## Optional Langfuse observability

Tracing is disabled by default. Enable it only after creating a Langfuse project and deciding that the configured host may receive the request data:

```env
LANGFUSE_ENABLED=true
LANGFUSE_PUBLIC_KEY=<your public key>
LANGFUSE_SECRET_KEY=<your secret key>
LANGFUSE_BASE_URL=https://cloud.langfuse.com
LANGFUSE_SAMPLE_RATE=1.0
```

One knowledge request produces this nested observation shape:

```text
opsrag.query (chain)
|-- query.classify (span)
`-- rag.pipeline (chain)
    |-- rag.retrieve (retriever)
    |-- rag.generate (generation)
    `-- rag.attribution (span)
```

General requests contain `query.classify` and `llm.general` beneath the root query. The trace stores the user question, full generation prompt, model response, final answer metadata, retrieved chunk identifiers, source names, retrieval and reranking scores, stage latency, and explicit error states. The current provider-neutral LLM contract returns text only, so token usage and cost are not fabricated when a provider does not expose them.

**Privacy:** the full RAG prompt contains retrieved knowledge-base excerpts. When Langfuse is enabled, those excerpts, the user question, and the answer are sent to `LANGFUSE_BASE_URL`. Keep tracing disabled for data that the configured Langfuse deployment is not permitted to receive. Missing credentials, a missing SDK, initialization failures, and exporter failures all fall back safely without making the API request fail.

## Evaluation dataset

The benchmark is stored as JSONL in `evaluation/questions.jsonl`. It contains 36 cases: 30 answerable questions grounded in the five current knowledge-base documents and 6 questions that must return the canonical insufficient-context response. Each of the six required categories contains exactly 6 cases.

Validate the dataset structure, duplicate rules, category coverage, and source/section references:

```bash
python scripts/validate_evaluation.py
```

The command prints deterministic case, category, and source-reference counts. Source counts include both primary and supporting-source references. It does not call an LLM or calculate answer-quality metrics.

Run the complete benchmark with the providers configured in `.env`:

```bash
python scripts/evaluate.py
```

The default run evaluates all 36 cases with dense, hybrid, and hybrid-plus-reranking retrieval and writes `evaluation/results.json`. If either the answer provider or judge provider is OpenAI, the runner refuses to begin until `--confirm-paid-run` is supplied. That flag acknowledges possible charges; it does not bypass provider billing or create credit.

The results artifact records the dataset hash, answer and judge providers/models, retrieved contexts, answers, citations, latency, source-hit and answerability checks, and all four RAGAS outcomes. Failed or undefined metric calls remain visible and are excluded from metric means rather than being reported as false zeroes.

### Current free local baseline

The checked-in `evaluation/results.json` was generated on 2026-08-27 across 108 executions (36 cases × 3 configurations), using Ollama `qwen3.5:2b` for both answers and judging, RAGAS 0.4.3, and local `BAAI/bge-small-en-v1.5` embeddings. No paid OpenAI request was used.

| Retrieval | App failures | Expected-source hit | Answerability accuracy | Mean app latency |
| --- | ---: | ---: | ---: | ---: |
| Dense | 3/36 | 93.33% | 88.89% | 6.56 s |
| Hybrid | 3/36 | 90.00% | 88.89% | 6.32 s |
| Hybrid + reranking | 3/36 | 86.67% | 91.67% | 10.58 s |

These are baseline measurements, not a claim that reranking wins every metric. The 256-token local judge profile produced many incomplete faithfulness outputs, so those failures remain explicit in the artifact and faithfulness means have low coverage. See the evaluation guide for scored counts and all RAGAS means.

See [evaluation/README.md](evaluation/README.md) for the dataset and results schemas, review checklist, and instructions for adding cases.

## Tests

```bash
python -m pytest
```

The OpenAI adapter follows the official Responses API shape (`instructions`, `input`, `max_output_tokens`, and the SDK `output_text` convenience property). Unit, API, HTTP-boundary, and Streamlit interaction tests use injected fake clients and services, so the suite does not require a paid model call.

See [docs/architecture.md](docs/architecture.md) for module boundaries and deferred milestones.
