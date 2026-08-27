# OpsRAG Architecture

## Implemented through Milestone 5

```text
raw Markdown/TXT/PDF
        |
        v
load + clean + preserve sections
        |
        v
deterministic token chunks (JSONL)
        |-------------------------------|
        v                               v
sentence-transformer embeddings     BM25 lexical index
        |                               |
        v                               v
Qdrant cosine collection           sparse retrieval
        |
        v
question -> dense retrieval -> source-labelled JSON context
        |
        v
provider-neutral LanguageModel interface
        |
        v
OpenAI Responses API adapter -> grounded answer + source metadata
```

BM25 is implemented as an independent lexical retriever over the same validated JSONL chunks. It preserves the normalized retrieval result contract and targets exact error codes, environment variables, commands, and product terminology. Hybrid fusion is intentionally deferred to Milestone 6.

The API process remains healthy without Qdrant or LLM credentials. External clients and the embedding model are created lazily when `/api/v1/ask` is used. HTTP tests replace the RAG dependency, so they never make paid model calls.

## Boundaries

- `app/ingestion` owns source parsing, cleaning, chunk IDs, and JSONL artifacts.
- `app/embeddings` owns query/document vector generation.
- `app/retrieval` owns Qdrant payload conventions and normalized results.
- `app/retrieval/bm25_search.py` owns deterministic lexical tokenization and BM25 ranking.
- `app/llm` owns the provider-neutral generation contract. Only its OpenAI adapter knows the Responses API.
- `app/rag` owns grounding instructions, context formatting, orchestration, and the retrieval-derived confidence heuristic.
- `app/api` validates public payloads and converts service failures to stable HTTP errors.

Hybrid fusion, reranking, LangGraph, evaluation, observability, and the UI are intentionally deferred to their milestones.

## Confidence semantics

`retrieval_confidence` is the highest dense cosine score clamped to `[0, 1]`. It is a simple context-relevance heuristic, not a calibrated probability and not an LLM self-assessment.
