# OpsRAG — Project Specification & Step-by-Step Development Plan

> **Purpose:** Build a compact, production-oriented Retrieval-Augmented Generation (RAG) project that demonstrates skills commonly expected from modern AI / LLM / GenAI Engineer roles.
>
> **Primary goal:** Finish the project in a short time while still demonstrating real-world RAG engineering practices such as hybrid retrieval, reranking, evaluation, observability, API development, testing, and containerization.
>
> **Development style:** Implement the project incrementally. Do not skip ahead unless the current milestone is complete and verified.

---

## 1. Project Summary

**Project Name:** OpsRAG

**Description:**  
OpsRAG is a technical knowledge assistant that answers questions using a private knowledge base containing technical documentation, troubleshooting guides, API notes, and operational documents.

The system must retrieve relevant context from the knowledge base before generating an answer.

The final answer must include:

- A grounded answer based on retrieved documents
- Source citations
- Retrieval metadata
- A confidence-related signal or retrieval score
- Observable execution traces
- Evaluation results

This is **not** intended to be a large enterprise system. Keep the scope small, clear, testable, and portfolio-friendly.

---

# 2. Main CV / Portfolio Goals

The final repository should clearly demonstrate practical experience with:

- Python
- FastAPI
- Retrieval-Augmented Generation (RAG)
- LangChain
- LangGraph
- Qdrant
- Dense vector search
- BM25 sparse search
- Hybrid retrieval
- Embeddings
- Cross-encoder reranking
- Source-grounded generation
- RAG evaluation
- RAGAS
- Langfuse
- Docker
- Docker Compose
- Pytest
- GitHub Actions

The repository should look like an engineering project, not just a notebook demo.

---

# 3. Scope

## In Scope

Implement:

1. Document ingestion
2. Text cleaning
3. Chunking
4. Embedding generation
5. Qdrant vector storage
6. Dense retrieval
7. BM25 retrieval
8. Hybrid search
9. Reranking
10. RAG answer generation
11. Source citations
12. FastAPI endpoints
13. Basic LangGraph routing
14. RAGAS evaluation
15. Langfuse observability
16. Streamlit demo UI
17. Docker / Docker Compose
18. Unit and integration tests
19. GitHub Actions
20. High-quality README documentation

---

## Out of Scope

Do **not** add these unless the main project is fully complete:

- Kubernetes
- Kafka
- Celery
- Redis
- Multi-agent architecture
- Complex agent systems
- Fine-tuning
- Authentication
- Authorization
- Complex user management
- AWS / Azure / GCP infrastructure
- MCP server
- Large-scale distributed architecture
- Mobile application
- Complex React frontend

These features can unnecessarily increase development time.

---

# 4. Target Architecture

```text
                         ┌─────────────────┐
                         │      User       │
                         └────────┬────────┘
                                  │
                                  ▼
                         ┌─────────────────┐
                         │    FastAPI      │
                         └────────┬────────┘
                                  │
                                  ▼
                         ┌─────────────────┐
                         │   LangGraph     │
                         │  Query Router   │
                         └────────┬────────┘
                                  │
                    ┌─────────────┴─────────────┐
                    │                           │
                    ▼                           ▼
           Knowledge Question            General Question
                    │                           │
                    ▼                           ▼
          ┌──────────────────┐           Direct LLM Answer
          │ Hybrid Retrieval │
          └────────┬─────────┘
                   │
          ┌────────┴─────────┐
          │                  │
          ▼                  ▼
    Dense Search           BM25
      Qdrant
          │                  │
          └────────┬─────────┘
                   │
                   ▼
          Candidate Documents
                   │
                   ▼
             Reranker
                   │
                   ▼
             Top Context
                   │
                   ▼
                  LLM
                   │
                   ▼
        Grounded Final Answer
                   │
        ┌──────────┴──────────┐
        │                     │
        ▼                     ▼
     Sources             Trace / Metrics
                              │
                              ▼
                           Langfuse
```

---

# 5. Technology Stack

## Backend

- Python 3.11+
- FastAPI
- Uvicorn
- Pydantic

## LLM / RAG

- LangChain
- LangGraph
- OpenAI / Gemini / Ollama-compatible LLM abstraction

The application should avoid hard-coding a single provider wherever practical.

## Embeddings

Preferred options:

- sentence-transformers
- BGE family embedding model

Keep the embedding layer configurable.

## Vector Database

- Qdrant

Use Qdrant through Docker Compose for local development.

## Sparse Retrieval

- BM25

Possible implementation:

- `rank-bm25`

## Reranking

Use a cross-encoder reranker from `sentence-transformers`.

Example model category:

```text
cross-encoder/ms-marco-*
```

Do not tightly couple the project to one model name unless necessary.

## Evaluation

- RAGAS

Metrics should include at least:

- Faithfulness
- Answer Relevance
- Context Precision
- Context Recall

## Observability

- Langfuse

Track when possible:

- Prompt
- Query
- Retrieved documents
- Retrieval scores
- LLM response
- Token usage
- Latency
- Errors

## Frontend

- Streamlit

Frontend is secondary. Keep it minimal.

## Infrastructure

- Docker
- Docker Compose

## Testing

- Pytest

## CI

- GitHub Actions

---

# 6. Proposed Repository Structure

```text
opsrag/
│
├── app/
│   ├── __init__.py
│   │
│   ├── api/
│   │   ├── __init__.py
│   │   ├── routes.py
│   │   └── schemas.py
│   │
│   ├── core/
│   │   ├── __init__.py
│   │   ├── config.py
│   │   └── logging.py
│   │
│   ├── ingestion/
│   │   ├── __init__.py
│   │   ├── loader.py
│   │   ├── cleaner.py
│   │   ├── chunker.py
│   │   └── pipeline.py
│   │
│   ├── embeddings/
│   │   ├── __init__.py
│   │   └── embedding_service.py
│   │
│   ├── retrieval/
│   │   ├── __init__.py
│   │   ├── vector_search.py
│   │   ├── bm25_search.py
│   │   ├── hybrid_search.py
│   │   └── reranker.py
│   │
│   ├── rag/
│   │   ├── __init__.py
│   │   ├── prompts.py
│   │   ├── pipeline.py
│   │   └── graph.py
│   │
│   ├── evaluation/
│   │   ├── __init__.py
│   │   └── ragas_eval.py
│   │
│   ├── observability/
│   │   ├── __init__.py
│   │   └── langfuse_client.py
│   │
│   └── main.py
│
├── frontend/
│   └── streamlit_app.py
│
├── data/
│   ├── raw/
│   └── processed/
│
├── evaluation/
│   ├── dataset.json
│   └── results.json
│
├── scripts/
│   ├── ingest.py
│   └── evaluate.py
│
├── tests/
│   ├── unit/
│   └── integration/
│
├── docs/
│   └── architecture.md
│
├── .github/
│   └── workflows/
│       └── test.yml
│
├── .env.example
├── .gitignore
├── docker-compose.yml
├── Dockerfile
├── pyproject.toml
├── README.md
└── PROJECT_PLAN.md
```

The structure can be adjusted when necessary, but avoid collapsing everything into a single file.

---

# 7. Core Data Model

Every chunk should contain metadata similar to:

```json
{
  "chunk_id": "doc_001_chunk_004",
  "document_id": "doc_001",
  "source": "postgres_troubleshooting.md",
  "title": "PostgreSQL Troubleshooting",
  "section": "Connection Refused",
  "text": "Example chunk content...",
  "chunk_index": 4
}
```

Recommended optional metadata:

```json
{
  "created_at": "...",
  "document_type": "markdown",
  "token_count": 530
}
```

The metadata must make source attribution possible.

---

# 8. API Contract

## Health Endpoint

```http
GET /health
```

Example:

```json
{
  "status": "ok"
}
```

---

## Ask Endpoint

```http
POST /api/v1/ask
```

Request:

```json
{
  "question": "Why does my application get PostgreSQL connection refused errors?"
}
```

Response:

```json
{
  "answer": "The application may be starting before PostgreSQL is ready...",
  "sources": [
    {
      "document": "postgres_troubleshooting.md",
      "section": "Connection Refused",
      "score": 0.91
    }
  ],
  "confidence": 0.87,
  "metadata": {
    "retrieved_chunks": 20,
    "reranked_chunks": 5
  }
}
```

Exact values may evolve during implementation.

---

# 9. Development Rules for Codex

When working on this project, follow these rules:

1. Work on **one milestone at a time**.
2. Do not implement future milestones unless explicitly requested.
3. Before modifying code:
   - inspect the existing repository;
   - understand current files;
   - reuse existing abstractions where reasonable.
4. Avoid unnecessary frameworks.
5. Avoid overengineering.
6. Keep modules small and readable.
7. Use type hints.
8. Use Pydantic models where appropriate.
9. Use environment variables for secrets and configuration.
10. Never commit real API keys.
11. Add/update tests for important behavior.
12. Prefer deterministic functions where possible.
13. Do not duplicate code.
14. Add logging around important pipeline stages.
15. Use clear exception messages.
16. Keep external services behind small wrapper classes.
17. Do not create fake benchmark results.
18. Do not create fake evaluation metrics.
19. Do not claim functionality that has not been implemented.
20. Update documentation after meaningful architectural changes.

---

# 10. Environment Variables

Initial `.env.example` should eventually support:

```env
APP_ENV=development

LLM_PROVIDER=
LLM_MODEL=
LLM_API_KEY=

EMBEDDING_MODEL=

QDRANT_URL=http://qdrant:6333
QDRANT_COLLECTION=opsrag_documents

LANGFUSE_PUBLIC_KEY=
LANGFUSE_SECRET_KEY=
LANGFUSE_HOST=

TOP_K_DENSE=10
TOP_K_SPARSE=10
TOP_K_RERANK=5
```

Only add environment variables when they are actually used.

---

# 11. Development Milestones

---

# Milestone 0 — Repository Bootstrap

## Goal

Create a clean project foundation.

## Tasks

- [ ] Create project folders
- [ ] Create Python virtual environment instructions
- [ ] Add `pyproject.toml`
- [ ] Add dependency management
- [ ] Add `.gitignore`
- [ ] Add `.env.example`
- [ ] Add FastAPI application
- [ ] Add `/health`
- [ ] Add basic configuration module
- [ ] Add basic logging
- [ ] Add initial tests
- [ ] Add initial README

## Definition of Done

Run:

```bash
uvicorn app.main:app --reload
```

Then:

```http
GET /health
```

must return:

```json
{
  "status": "ok"
}
```

Tests must pass.

Do not implement RAG yet.

---

# Milestone 1 — Dataset and Document Ingestion

## Goal

Create a reproducible ingestion pipeline.

## Initial Supported Formats

Start with:

- Markdown
- TXT
- PDF

HTML support is optional and can be added later.

## Tasks

- [ ] Create a small technical knowledge base
- [ ] Implement document loader
- [ ] Normalize encoding
- [ ] Remove unnecessary whitespace
- [ ] Preserve source metadata
- [ ] Preserve titles / section information when possible
- [ ] Add ingestion CLI script
- [ ] Add tests

## Dataset Theme

Prefer technical operations / developer support documents such as:

- Docker troubleshooting
- PostgreSQL troubleshooting
- FastAPI deployment
- API error handling
- Logging
- Environment variables
- Service startup
- Network issues
- REST API troubleshooting

Do not use an unnecessarily large dataset.

A small but high-quality dataset is preferable.

## Definition of Done

Running:

```bash
python scripts/ingest.py
```

should successfully load the local documents and report:

- Number of documents
- Number of parsed sections
- Number of failures

No embeddings yet.

---

# Milestone 2 — Chunking

## Goal

Transform parsed documents into high-quality retrieval chunks.

## Initial Strategy

Use recursive text splitting.

Target range:

```text
500–700 tokens per chunk
```

Overlap:

```text
50–100 tokens
```

These numbers are starting points, not fixed rules.

## Requirements

Each chunk must preserve:

- `chunk_id`
- `document_id`
- `source`
- `title`
- `section`
- `chunk_index`
- `text`

## Tasks

- [ ] Implement chunking service
- [ ] Preserve metadata
- [ ] Add token counting
- [ ] Add tests
- [ ] Log chunk statistics

## Useful Statistics

At ingestion time, print or log:

```text
documents: X
chunks: X
average tokens/chunk: X
min tokens: X
max tokens: X
```

## Definition of Done

The same document should produce deterministic chunks when configuration does not change.

---

# Milestone 3 — Embeddings + Qdrant

## Goal

Store chunk embeddings in Qdrant and support semantic retrieval.

## Tasks

- [ ] Add Qdrant to Docker Compose
- [ ] Implement embedding service
- [ ] Create Qdrant collection
- [ ] Upsert chunks
- [ ] Store chunk metadata as payload
- [ ] Implement dense search
- [ ] Add search tests
- [ ] Handle collection recreation safely

## Dense Retrieval Interface

Conceptually:

```python
search(query: str, top_k: int) -> list[RetrievedChunk]
```

Result object should include:

```text
text
metadata
score
retrieval_method
```

## Definition of Done

Given a technical question, dense retrieval should return semantically relevant chunks from Qdrant.

No LLM answer generation required yet.

---

# Milestone 4 — Basic RAG

## Goal

Create the first end-to-end RAG answer.

Flow:

```text
question
→ dense retrieval
→ context formatting
→ LLM prompt
→ answer
```

## Prompt Requirements

The LLM must:

- Answer using the provided context
- Avoid inventing unsupported claims
- Say when the context is insufficient
- Refer to source identifiers

## Tasks

- [ ] Implement prompt template
- [ ] Implement RAG pipeline
- [ ] Create response model
- [ ] Connect pipeline to `/api/v1/ask`
- [ ] Return sources
- [ ] Add tests using mocked LLM calls where possible

## Definition of Done

API can answer a question using Qdrant context and return source metadata.

---

# Milestone 5 — BM25 Sparse Retrieval

## Goal

Add lexical retrieval.

## Tasks

- [ ] Build BM25 index from chunks
- [ ] Implement BM25 search interface
- [ ] Return normalized result format
- [ ] Add tests

## Why

Dense search works well for semantic similarity.

BM25 helps with:

- Error codes
- Exact product names
- Function names
- Environment variables
- Command names
- Technical terminology

## Definition of Done

BM25 retrieval works independently from Qdrant retrieval.

---

# Milestone 6 — Hybrid Retrieval

## Goal

Combine dense and sparse retrieval.

Flow:

```text
                 Query
                   │
        ┌──────────┴──────────┐
        │                     │
        ▼                     ▼
 Dense Retrieval          BM25 Search
        │                     │
        └──────────┬──────────┘
                   │
                   ▼
             Fusion Layer
                   │
                   ▼
          Candidate Chunks
```

## Initial Fusion Strategy

Prefer a simple approach such as:

- Reciprocal Rank Fusion (RRF)

Avoid complex learned fusion.

## Tasks

- [ ] Implement hybrid retriever
- [ ] Deduplicate chunks
- [ ] Add RRF or equivalent simple fusion
- [ ] Include retrieval method metadata
- [ ] Add tests

## Definition of Done

Hybrid retrieval produces one ranked candidate list from dense + sparse results.

---

# Milestone 7 — Reranking

## Goal

Improve final context quality with a cross-encoder.

Flow:

```text
Hybrid Search
     ↓
Top ~20 candidates
     ↓
Cross-Encoder
     ↓
Top ~5 chunks
```

## Tasks

- [ ] Add reranker service
- [ ] Make candidate count configurable
- [ ] Make final context count configurable
- [ ] Record reranking scores
- [ ] Add latency logging
- [ ] Add tests

## Definition of Done

The final RAG context uses reranked chunks rather than raw retrieval ordering.

---

# Milestone 8 — Source Attribution

## Goal

Make responses clearly grounded.

Every answer should expose sources.

## Example

```json
{
  "answer": "...",
  "sources": [
    {
      "document": "docker_troubleshooting.md",
      "section": "Container Networking",
      "score": 0.92
    }
  ]
}
```

## Requirements

- No invented source names
- Sources must come from retrieved chunks
- Duplicate sources should be handled cleanly
- Source metadata should be human-readable

## Definition of Done

The user can see which documents were used for every RAG response.

---

# Milestone 9 — LangGraph Query Routing

## Goal

Demonstrate basic LangGraph usage without creating a complex agent system.

Flow:

```text
User Query
    ↓
Classifier
    ↓
┌───────────────┬───────────────┐
│ Knowledge     │ General       │
│ Question      │ Question      │
└───────┬───────┴───────┬───────┘
        │               │
        ▼               ▼
       RAG          Direct LLM
        │               │
        └───────┬───────┘
                ▼
              Answer
```

## Important

Keep routing simple.

Do not create:

- planner agents
- supervisor agents
- multi-agent systems
- autonomous loops

## Tasks

- [ ] Define graph state
- [ ] Add routing node
- [ ] Add RAG node
- [ ] Add general answer node
- [ ] Add graph tests

## Definition of Done

Knowledge questions use RAG.

Clearly general questions can bypass retrieval.

---

# Milestone 10 — Evaluation Dataset

## Goal

Create a small benchmark for the project.

Target:

```text
30–50 questions
```

Each item should contain at least:

```json
{
  "question": "...",
  "expected_answer": "...",
  "expected_source": "..."
}
```

## Dataset Guidelines

Questions should include:

- Semantic questions
- Exact keyword questions
- Error code questions
- Multi-sentence questions
- Ambiguous questions
- Questions with insufficient context

Do not generate meaningless synthetic questions just to increase the count.

## Definition of Done

Evaluation dataset is version-controlled and manually reviewable.

---

# Milestone 11 — RAGAS Evaluation

## Goal

Measure RAG quality.

## Initial Metrics

- Faithfulness
- Answer Relevance
- Context Precision
- Context Recall

## Tasks

- [ ] Add evaluation runner
- [ ] Run benchmark dataset
- [ ] Store results
- [ ] Produce summary metrics
- [ ] Compare retrieval configurations

## Optional Comparison

Compare:

```text
Dense only
vs
Hybrid
vs
Hybrid + Reranker
```

This is strongly recommended because it creates useful portfolio evidence.

## Example Result Table

```text
Configuration          Faithfulness   Context Precision
Dense Only             ...
Hybrid                  ...
Hybrid + Reranker       ...
```

Never invent values.

Only add metrics after real evaluation runs.

## Definition of Done

`evaluation/results.json` contains real evaluation outputs.

README contains real measurements.

---

# Milestone 12 — Langfuse Observability

## Goal

Trace RAG execution.

## Track

At minimum:

- User query
- Retrieval stage
- Retrieved chunk identifiers
- Retrieval / reranking scores
- Prompt
- Model response
- Latency
- Error state

If available:

- Token usage
- Cost

## Tasks

- [ ] Add Langfuse configuration
- [ ] Add tracing wrapper
- [ ] Trace retrieval
- [ ] Trace generation
- [ ] Ensure project works if Langfuse is disabled

## Important

Observability must be optional.

The application should not crash when Langfuse credentials are missing.

## Definition of Done

A RAG request can be inspected end-to-end in Langfuse when credentials are configured.

---

# Milestone 13 — Streamlit Demo

## Goal

Create a minimal UI for GitHub/demo purposes.

## UI Requirements

Show:

- Question input
- Generated answer
- Source documents
- Confidence / score
- Optional execution metadata

Example:

```text
┌──────────────────────────────────────┐
│            OpsRAG                    │
├──────────────────────────────────────┤
│ Ask a technical question             │
│ [.................................]  │
│                        [Ask]         │
├──────────────────────────────────────┤
│ Answer                               │
│ ...                                  │
│                                      │
│ Sources                              │
│ - postgres_troubleshooting.md        │
│ - docker_networking.md               │
│                                      │
│ Confidence: 0.87                     │
└──────────────────────────────────────┘
```

Do not spend significant time on design.

## Definition of Done

Streamlit communicates with the FastAPI backend successfully.

---

# Milestone 14 — Dockerization

## Goal

Run the project locally with minimal setup.

## Docker Compose Services

Recommended:

```text
api
qdrant
frontend
```

Optional additional services only if truly required.

## Expected Command

```bash
docker compose up --build
```

## Definition of Done

A new developer can clone the repository, configure `.env`, run Docker Compose, ingest documents, and ask questions.

---

# Milestone 15 — Tests

## Goal

Add enough tests to show engineering quality.

## Unit Tests

Prioritize:

- Cleaning
- Chunking
- BM25
- RRF
- Deduplication
- Source formatting
- Router logic

## Integration Tests

Prioritize:

- Qdrant retrieval
- `/health`
- `/api/v1/ask`
- End-to-end retrieval pipeline

Mock paid/external LLM calls where practical.

## Definition of Done

```bash
pytest
```

passes locally.

---

# Milestone 16 — GitHub Actions

## Goal

Run tests automatically.

Workflow should:

1. Checkout
2. Setup Python
3. Install dependencies
4. Run linting if configured
5. Run tests

Do not create a complicated CI/CD system.

## Definition of Done

Pull requests / pushes run automated tests.

---

# Milestone 17 — Final README

## README Must Include

1. Project summary
2. Why the project exists
3. Key features
4. Architecture diagram
5. Technology stack
6. Local setup
7. Environment variables
8. Docker instructions
9. Ingestion instructions
10. API examples
11. Evaluation methodology
12. Real evaluation results
13. Screenshots
14. Langfuse screenshot if appropriate
15. Repository structure
16. Engineering decisions
17. Limitations
18. Future improvements

---

# 12. Evaluation Experiments

Before calling the project complete, run at least these experiments.

## Experiment A — Dense Retrieval

```text
Qdrant vector search
→ LLM
```

Record retrieval/evaluation results.

---

## Experiment B — Hybrid Retrieval

```text
Dense
+
BM25
→ Fusion
→ LLM
```

Compare with Experiment A.

---

## Experiment C — Hybrid + Reranker

```text
Dense
+
BM25
→ Fusion
→ Cross-Encoder
→ LLM
```

Compare with A and B.

The final README should explain whether the extra complexity actually improved results.

Do not assume it did.

---

# 13. Confidence Handling

Do not pretend that an arbitrary number generated by the LLM is a reliable confidence score.

Preferred implementation:

Derive a simple `retrieval_confidence` or `context_relevance` indicator from retrieval/reranking signals.

Example naming:

```json
{
  "retrieval_confidence": 0.87
}
```

Document clearly that this is a retrieval-derived heuristic, not calibrated model probability.

---

# 14. Logging

Important pipeline stages should log structured information.

Example:

```text
request_id
query
dense_candidates
sparse_candidates
hybrid_candidates
reranked_candidates
retrieval_latency_ms
generation_latency_ms
total_latency_ms
```

Do not log API keys or secrets.

---

# 15. Error Handling

Handle:

- Empty query
- Unsupported document type
- Failed parsing
- Qdrant unavailable
- Embedding failure
- LLM unavailable
- No relevant context
- Invalid environment configuration

Return useful errors instead of raw stack traces.

---

# 16. Security Basics

Even though this is a portfolio project:

- Never commit secrets
- Validate uploaded file types if upload support is added
- Limit file size if upload support is added
- Do not execute document contents
- Avoid unsafe deserialization
- Sanitize filenames if files can be uploaded

Do not implement enterprise security features unless required.

---

# 17. Coding Quality Requirements

Use:

- Python type hints
- Docstrings where useful
- Clear naming
- Small functions
- Dependency injection where it reduces coupling
- Configuration objects
- Reusable schemas
- Consistent logging

Avoid:

- Huge classes
- 500-line modules
- Global mutable state
- Hard-coded API keys
- Hard-coded absolute file paths
- Silent exception handling
- Copy-pasted retrieval logic
- Notebook-only implementation

---

# 18. Git Strategy

Use small meaningful commits.

Suggested pattern:

```text
feat: bootstrap FastAPI application
feat: add document ingestion pipeline
feat: add recursive chunking
feat: integrate qdrant vector store
feat: implement dense retrieval
feat: add bm25 sparse retrieval
feat: implement hybrid retrieval with rrf
feat: add cross encoder reranker
feat: add grounded answer generation
feat: add langgraph query routing
feat: add ragas evaluation pipeline
feat: add langfuse tracing
feat: add streamlit demo
test: add retrieval unit tests
ci: add github actions
docs: finalize project documentation
```

Do not create artificial commit history after the project is finished.

---

# 19. Recommended Development Order

Follow this order:

```text
0. Bootstrap
1. Documents
2. Chunking
3. Embeddings
4. Qdrant
5. Dense Retrieval
6. Basic RAG
7. BM25
8. Hybrid Retrieval
9. Reranker
10. Citations
11. LangGraph
12. Evaluation Dataset
13. RAGAS
14. Langfuse
15. Streamlit
16. Docker polish
17. Tests
18. GitHub Actions
19. README
```

Do not start LangGraph, RAGAS, Langfuse, or UI before basic retrieval works correctly.

---

# 20. MVP Completion Criteria

The MVP is complete when all of the following are true:

- [ ] Technical documents can be ingested
- [ ] Documents are chunked with metadata
- [ ] Chunks are embedded
- [ ] Chunks are stored in Qdrant
- [ ] Dense retrieval works
- [ ] BM25 retrieval works
- [ ] Hybrid retrieval works
- [ ] Reranking works
- [ ] RAG produces grounded answers
- [ ] Responses include sources
- [ ] FastAPI endpoint works
- [ ] LangGraph routing works
- [ ] Evaluation dataset exists
- [ ] RAGAS evaluation runs
- [ ] Real evaluation results exist
- [ ] Langfuse traces requests
- [ ] Streamlit demo works
- [ ] Docker Compose starts the required services
- [ ] Tests pass
- [ ] GitHub Actions passes
- [ ] README documents the complete system

---

# 21. Stretch Goals

Only consider after MVP completion.

Possible additions:

- Query rewriting
- Multi-query retrieval
- Metadata filtering
- Parent-child retrieval
- Semantic chunking comparison
- pgvector comparison
- Local LLM mode with Ollama
- Response streaming
- Document upload endpoint
- Simple caching
- Retrieval threshold tuning
- Prompt injection defense experiments
- RAG evaluation dashboard

Do not implement all stretch goals.

Choose at most one or two if they add clear portfolio value.

---

# 22. Questions Codex Should Ask Before Major Changes

Before making a major architectural decision, verify:

1. Does this help the project demonstrate RAG engineering?
2. Does this materially improve retrieval quality, evaluation, or maintainability?
3. Can it be completed without expanding the project scope too much?
4. Is there a simpler implementation that demonstrates the same skill?
5. Can the result be measured or tested?

If the answer is no, prefer the simpler design.

---

# 23. Final Portfolio Message

The repository should communicate this message:

> This project is not merely a chatbot that sends retrieved text to an LLM.  
> It is a compact, measurable RAG system with document ingestion, hybrid retrieval, reranking, grounded generation, evaluation, observability, API design, testing, and containerized deployment.

---

# 24. Expected CV Description After Completion

Do not add this text to the CV until the relevant features are actually implemented.

Suggested wording:

> **OpsRAG — Production-Oriented RAG Knowledge Assistant**  
> Developed a production-oriented RAG system using Python, FastAPI, LangChain, LangGraph, and Qdrant. Implemented hybrid retrieval combining dense embeddings and BM25, cross-encoder reranking, source-grounded generation, and automated RAG evaluation using RAGAS. Added Langfuse observability, automated testing, and Docker-based local deployment.

Suggested technologies:

```text
Python · FastAPI · RAG · LangChain · LangGraph · Qdrant ·
Embeddings · BM25 · Hybrid Search · Reranking · RAGAS ·
Langfuse · Docker · Pytest · GitHub Actions
```

Only list technologies that are genuinely implemented.

---

# 25. First Implementation Task

Start with **Milestone 0 — Repository Bootstrap**.

Codex instruction:

```text
Read PROJECT_PLAN.md completely before making changes.

Implement only Milestone 0.

Requirements:
- Create the proposed initial project structure only as needed for Milestone 0.
- Create a minimal FastAPI application.
- Implement GET /health returning {"status": "ok"}.
- Add configuration management.
- Add basic logging.
- Add pyproject.toml.
- Add .gitignore.
- Add .env.example.
- Add pytest tests for the health endpoint.
- Add a concise README with local startup instructions.

Do not implement document ingestion, embeddings, Qdrant, LangChain,
LangGraph, RAG, BM25, reranking, RAGAS, Langfuse, Streamlit, or Docker
services yet unless absolutely required for Milestone 0.

After implementation:
1. Run tests.
2. Report changed files.
3. Report test results.
4. Report any assumptions.
5. Stop and wait for the next milestone.
```

---

# 26. Project Principle

**Build the smallest version that proves the engineering concept, measure it, then improve it.**

Do not optimize for number of technologies.

Optimize for:

- working code;
- clear architecture;
- measurable retrieval quality;
- reproducibility;
- explainability in interviews;
- portfolio value.
