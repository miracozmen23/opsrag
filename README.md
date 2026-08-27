# OpsRAG

OpsRAG is a compact technical knowledge assistant built incrementally as a production-oriented RAG portfolio project. The current implementation loads local documents, creates deterministic chunks, indexes normalized BGE embeddings in Qdrant, supports independent dense and BM25 retrieval, and generates a source-grounded answer through a provider-neutral LLM interface.

## Current scope

Completed through **Milestone 5 — BM25 Sparse Retrieval**:

- FastAPI application with `GET /health`
- Markdown, TXT, and text-based PDF ingestion
- conservative cleaning and title/section preservation
- deterministic 600-token chunks with 75-token overlap
- configurable Sentence Transformers embedding service
- Qdrant collection creation, safe explicit recreation, and dense search
- deterministic in-memory BM25 indexing and lexical search for exact technical terms
- grounded prompt with source identifiers
- provider-neutral LLM protocol and OpenAI Responses API adapter
- `POST /api/v1/ask` returning answer, sources, retrieval metadata, and a retrieval-derived confidence heuristic
- mock-based tests that do not require an API key or paid model call

Hybrid retrieval, reranking, LangGraph, RAGAS, Langfuse, Streamlit, and full application containers are later milestones.

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
      "section": "Connection refused",
      "score": 0.82,
      "chunk_id": "chunk_..."
    }
  ],
  "retrieval_confidence": 0.82,
  "metadata": {
    "retrieved_chunks": 10,
    "retrieval_method": "dense"
  }
}
```

`retrieval_confidence` is the top dense cosine score clamped to `[0, 1]`. It is not a calibrated probability.

## Tests

```bash
python -m pytest
```

The OpenAI adapter follows the official Responses API shape (`instructions`, `input`, `max_output_tokens`, and the SDK `output_text` convenience property). Unit and API tests use injected fake clients and services.

See [docs/architecture.md](docs/architecture.md) for module boundaries and deferred milestones.
