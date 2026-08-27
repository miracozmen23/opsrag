# OpsRAG Architecture

## Implemented through Milestone 6

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
        |                               |
        v                               v
  dense ranked list  ------------> RRF fusion
                                      |
                                      v
                            hybrid candidate list

question -> dense retrieval -> source-labelled JSON context
        |
        v
provider-neutral LanguageModel interface
        |
        v
OpenAI Responses API adapter -> grounded answer + source metadata
```

BM25 is implemented as an independent lexical retriever over the same validated JSONL chunks. It preserves the normalized retrieval result contract and targets exact error codes, environment variables, commands, and product terminology.

`HybridRetriever` requests configurable candidate counts from dense and sparse retrieval, deduplicates by stable chunk ID, and combines ranks with Reciprocal Rank Fusion. Raw cosine and BM25 scores are deliberately not compared because their scales have different meanings. The basic RAG API remains dense-only until the next reranking milestone connects the hybrid candidate list to final context selection.

The API process remains healthy without Qdrant or LLM credentials. External clients and the embedding model are created lazily when `/api/v1/ask` is used. HTTP tests replace the RAG dependency, so they never make paid model calls.

## Boundaries

- `app/ingestion` owns source parsing, cleaning, chunk IDs, and JSONL artifacts.
- `app/embeddings` owns query/document vector generation.
- `app/retrieval` owns Qdrant payload conventions and normalized results.
- `app/retrieval/bm25_search.py` owns deterministic lexical tokenization and BM25 ranking.
- `app/retrieval/hybrid_search.py` owns chunk deduplication and RRF fusion.
- `app/llm` owns the provider-neutral generation contract. Only its OpenAI adapter knows the Responses API.
- `app/rag` owns grounding instructions, context formatting, orchestration, and the retrieval-derived confidence heuristic.
- `app/api` validates public payloads and converts service failures to stable HTTP errors.

Reranking, LangGraph, evaluation, observability, and the UI are intentionally deferred to their milestones.

## Confidence semantics

`retrieval_confidence` is the highest dense cosine score clamped to `[0, 1]`. It is a simple context-relevance heuristic, not a calibrated probability and not an LLM self-assessment.
