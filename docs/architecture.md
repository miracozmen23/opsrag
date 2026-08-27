# OpsRAG Architecture

## Implemented through Milestone 7

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
                                      |
                                      v
                              cross-encoder
                                      |
                                      v
                         reranked final RAG context
                                      |
                                      v
                       source-labelled JSON context
                                      |
                                      v
                    provider-neutral LanguageModel
                                      |
                                      v
                    OpenAI Responses API adapter
                                      |
                                      v
                     grounded answer + source metadata
```

BM25 is implemented as an independent lexical retriever over the same validated JSONL chunks. It preserves the normalized retrieval result contract and targets exact error codes, environment variables, commands, and product terminology.

`HybridRetriever` requests configurable candidate counts from dense and sparse retrieval, deduplicates by stable chunk ID, and combines ranks with Reciprocal Rank Fusion. Raw cosine and BM25 scores are deliberately not compared because their scales have different meanings.

`RerankingRetriever` asks the hybrid retriever for a configurable candidate pool and passes each `(query, chunk text)` pair to a lazy `CrossEncoderReranker`. The wrapper selects the configurable final context count. Stable input order breaks equal-score ties. Each result retains its RRF score and records the cross-encoder logit separately as `rerank_score`.

The production API dependency graph now constructs dense retrieval, BM25, RRF fusion, and cross-encoder reranking in that order. `RAGPipeline` receives only the final reranked chunks, so prompts no longer use raw dense or hybrid ordering.

Both sentence-transformer services use the same `MODEL_CACHE_DIR`. Relative cache and processed-data paths resolve against the repository root. This keeps model artifacts on the repository drive and prevents the launch directory from silently changing storage placement.

The API process remains healthy without Qdrant or LLM credentials. External clients and the embedding model are created lazily when `/api/v1/ask` is used. HTTP tests replace the RAG dependency, so they never make paid model calls.

## Boundaries

- `app/ingestion` owns source parsing, cleaning, chunk IDs, and JSONL artifacts.
- `app/embeddings` owns query/document vector generation.
- `app/retrieval` owns Qdrant payload conventions and normalized results.
- `app/retrieval/bm25_search.py` owns deterministic lexical tokenization and BM25 ranking.
- `app/retrieval/hybrid_search.py` owns chunk deduplication and RRF fusion.
- `app/retrieval/reranker.py` owns cross-encoder scoring and final candidate selection.
- `app/llm` owns the provider-neutral generation contract. Only its OpenAI adapter knows the Responses API.
- `app/rag` owns grounding instructions, context formatting, orchestration, and the retrieval-derived confidence heuristic.
- `app/api` validates public payloads and converts service failures to stable HTTP errors.

LangGraph, evaluation, observability, and the UI are intentionally deferred to their milestones.

## Confidence semantics

The configured cross-encoder emits raw relevance logits. OpsRAG preserves these internally, applies a numerically stable sigmoid to the best final logit, and exposes the result as `retrieval_confidence`. Public source scores use the same transformation. It is a context-relevance heuristic, not a calibrated probability and not an LLM self-assessment.

Directly constructed dense-only `RAGPipeline` instances remain supported for isolated tests and reuse; without a reranker score, their confidence falls back to the previous clamped retrieval score behavior.
