# OpsRAG Architecture

## Implemented through Milestone 11

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
```

Request-time routing is a separate bounded workflow:

```text
                              question
                                 |
                                 v
                    LangGraph classify node
                         /              \
                        /                \
               knowledge                 general
                   |                        |
                   v                        v
       lazy hybrid + reranked RAG       direct LLM
                   |                        |
                   v                        v
       group duplicate sources       no retrieval metadata
                   |
                   v
       source-labelled JSON context
                   |
                   v
       provider-neutral LanguageModel
                   |
                   v
         validate answer citations
                   |
                   v
       grounded answer + cited sources
```

The evaluation path is deliberately separate from request-time execution:

```text
raw knowledge-base documents
            |
            v
manually reviewed questions + reference answers (JSONL)
            |
            v
schema, category, duplicate, source, and section validation
            |
            v
version-controlled benchmark
            |
            v
dense / hybrid / hybrid + reranking pipelines
            |
            v
real generated answers + retrieved evidence
            |
            v
RAGAS metrics + deterministic source/answerability checks
            |
            v
versioned JSON result artifact
```

BM25 is implemented as an independent lexical retriever over the same validated JSONL chunks. It preserves the normalized retrieval result contract and targets exact error codes, environment variables, commands, and product terminology.

`HybridRetriever` requests configurable candidate counts from dense and sparse retrieval, deduplicates by stable chunk ID, and combines ranks with Reciprocal Rank Fusion. Raw cosine and BM25 scores are deliberately not compared because their scales have different meanings.

`RerankingRetriever` asks the hybrid retriever for a configurable candidate pool and passes each `(query, chunk text)` pair to a lazy `CrossEncoderReranker`. The wrapper selects the configurable final context count. Stable input order breaks equal-score ties. Each result retains its RRF score and records the cross-encoder logit separately as `rerank_score`.

When the knowledge route is selected, the production API constructs dense retrieval, BM25, RRF fusion, and cross-encoder reranking in that order. `RAGPipeline` receives only the final reranked chunks, so prompts never use raw dense or hybrid ordering.

Before prompt construction, source attribution groups chunks by original source identity, title, section, and page while preserving reranked order. Each source receives one deterministic request-local identifier such as `S1`. The prompt contains human-readable metadata plus all contributing chunk excerpts under that identifier.

After generation, the pipeline parses source-like tokens in the answer. It rejects missing citations, malformed forms such as `[S1, S2]`, and identifiers that were not supplied in context. Repeated valid citations are collapsed, and the API returns only cited sources in first-reference order. Document names, titles, sections, pages, and chunk identifiers always originate from retrieved metadata rather than LLM text.

`QueryRoutingGraph` uses typed state and exactly three nodes: `classify`, `rag`, and `general`. A conditional edge selects one terminal answer node, and that node connects directly to `END`; there are no cycles. `RuleBasedQueryRouter` only bypasses retrieval for explicit general-message and general-fact patterns. All other questions default to the knowledge path.

The knowledge pipeline is behind a lazy answer-service boundary. Building the top-level graph requires the configured LLM but does not read processed chunks, build BM25, connect to Qdrant, or load transformer models. Those operations begin only when the `rag` node is selected. The general node returns no sources, zero retrieval confidence, `retrieval_method="not_used"`, and `route="general"`.

Both sentence-transformer services use the same `MODEL_CACHE_DIR`. Relative cache and processed-data paths resolve against the repository root. This keeps model artifacts on the repository drive and prevents the launch directory from silently changing storage placement.

The API process remains healthy without Qdrant or LLM credentials. External clients and the embedding model are created lazily when `/api/v1/ask` is used. HTTP tests replace the RAG dependency, so they never make paid model calls.

The benchmark dataset is a static, human-reviewable JSONL artifact. `app/evaluation` enforces typed case contracts, unique IDs and normalized questions, 30–50 case bounds, all six required categories, and complete knowledge-source coverage. Answerable cases must reference a source file and an exact section title loaded through the production ingestion path. Insufficient-context cases cannot claim evidence and must use the same canonical fallback text as the RAG prompt.

Dataset validation remains deterministic and offline. The separate benchmark runner executes all selected retrieval configurations against the frozen dataset, captures application outputs, then calculates RAGAS faithfulness, answer relevance, context precision, and context recall. It also calculates deterministic expected-source hit rate and answerability accuracy.

Answer generation and judging are independently configurable as OpenAI or Ollama providers. The Ollama adapter uses its OpenAI-compatible local endpoint, while RAGAS uses local Hugging Face embeddings. Runs involving OpenAI require an explicit paid-run confirmation. Each application or metric failure is preserved in the results contract; unavailable measurements are never coerced to zero, and one failed case does not abort later cases.

## Boundaries

- `app/ingestion` owns source parsing, cleaning, chunk IDs, and JSONL artifacts.
- `app/embeddings` owns query/document vector generation.
- `app/retrieval` owns Qdrant payload conventions and normalized results.
- `app/retrieval/bm25_search.py` owns deterministic lexical tokenization and BM25 ranking.
- `app/retrieval/hybrid_search.py` owns chunk deduplication and RRF fusion.
- `app/retrieval/reranker.py` owns cross-encoder scoring and final candidate selection.
- `app/llm` owns the provider-neutral generation contract. Provider adapters isolate OpenAI Responses API and Ollama's OpenAI-compatible chat endpoint details.
- `app/rag/attribution.py` owns source grouping, public relevance scores, and citation validation.
- `app/rag/graph.py` owns graph state, deterministic classification, conditional routing, and the direct-answer node.
- `app/rag` owns grounding instructions, context formatting, orchestration, and the retrieval-derived confidence heuristic.
- `app/evaluation` owns benchmark case models, JSONL loading, dataset/source validation, RAGAS scoring, execution, and versioned result contracts.
- `app/api` validates public payloads and converts service failures to stable HTTP errors.
- `evaluation/questions.jsonl` is the version-controlled, manually reviewed benchmark dataset.
- `evaluation/results.json` is produced only by a real run and records model/provider provenance plus per-case and aggregate outcomes.

Observability and the UI are intentionally deferred to their milestones.

## Confidence semantics

The configured cross-encoder emits raw relevance logits. OpsRAG preserves these internally and applies a numerically stable sigmoid for public source scores. `retrieval_confidence` is the highest score among sources cited by the answer, not merely the first retrieved chunk. It is a context-relevance heuristic, not a calibrated probability and not an LLM self-assessment.

Directly constructed dense-only `RAGPipeline` instances remain supported for isolated tests and reuse; without a reranker score, their confidence falls back to the previous clamped retrieval score behavior.

General-route answers do not use retrieval, so they expose `retrieval_confidence=0.0` rather than manufacturing a confidence signal.
