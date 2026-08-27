# OpsRAG

OpsRAG is a compact technical knowledge assistant built incrementally as a production-oriented RAG portfolio project. The current implementation routes requests through a bounded LangGraph workflow, runs technical questions through hybrid retrieval and cross-encoder reranking, answers clearly general messages directly without pretending retrieval occurred, and includes a source-validated evaluation benchmark for the next measurement stage.

## Current scope

Completed through **Milestone 10 — Evaluation Dataset**:

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
- provider-neutral LLM protocol and OpenAI Responses API adapter
- `POST /api/v1/ask` returning answer, sources, retrieval metadata, and a retrieval-derived confidence heuristic
- mock-based tests that do not require an API key or paid model call

RAGAS, Langfuse, Streamlit, and full application containers are later milestones.

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

Configure an LLM model and secret in `.env`:

```env
LLM_PROVIDER=openai
LLM_MODEL=<an OpenAI model available to your project>
LLM_API_KEY=<your API key>
```

The model name is intentionally not hard-coded. Do not commit `.env`.

Start the API:

```bash
uvicorn app.main:app --reload
```

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

## Evaluation dataset

The benchmark is stored as JSONL in `evaluation/questions.jsonl`. It contains 36 cases: 30 answerable questions grounded in the five current knowledge-base documents and 6 questions that must return the canonical insufficient-context response. Each of the six required categories contains exactly 6 cases.

Validate the dataset structure, duplicate rules, category coverage, and source/section references:

```bash
python scripts/validate_evaluation.py
```

The command prints deterministic case, category, and source-reference counts. Source counts include both primary and supporting-source references. It does not call an LLM, calculate answer-quality metrics, or claim RAG performance; those capabilities belong to the next evaluation milestone.

See [evaluation/README.md](evaluation/README.md) for the schema, review checklist, and instructions for adding cases.

## Tests

```bash
python -m pytest
```

The OpenAI adapter follows the official Responses API shape (`instructions`, `input`, `max_output_tokens`, and the SDK `output_text` convenience property). Unit and API tests use injected fake clients and services.

See [docs/architecture.md](docs/architecture.md) for module boundaries and deferred milestones.
