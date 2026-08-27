"""Run the complete OpsRAG benchmark and store real RAGAS measurements."""

import argparse
import asyncio
import json
import logging
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.core.config import Settings, resolve_project_path  # noqa: E402
from app.core.logging import configure_logging  # noqa: E402
from app.evaluation.dataset import (  # noqa: E402
    read_evaluation_jsonl,
    validate_evaluation_dataset,
)
from app.evaluation.ragas_eval import (  # noqa: E402
    RagasConfigurationError,
    create_ragas_scorer,
)
from app.evaluation.results import (  # noqa: E402
    BenchmarkResults,
    EvaluationConfiguration,
)
from app.evaluation.runner import (  # noqa: E402
    create_evaluation_pipelines,
    has_metric_failures,
    run_benchmark,
    write_benchmark_results,
)
from app.llm.factory import LLMConfigurationError, create_llm_service  # noqa: E402
from app.rag.pipeline import RAGPipelineError  # noqa: E402
from app.retrieval.factory import create_retriever_suite  # noqa: E402

ALL_CONFIGURATIONS: tuple[EvaluationConfiguration, ...] = (
    "dense",
    "hybrid",
    "hybrid_reranked",
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=PROJECT_ROOT / "evaluation" / "questions.jsonl",
        help="Validated benchmark JSONL path.",
    )
    parser.add_argument(
        "--sources",
        type=Path,
        default=PROJECT_ROOT / "data" / "raw",
        help="Knowledge-base source directory used for validation.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "evaluation" / "results.json",
        help="Versioned results artifact written after the complete run.",
    )
    parser.add_argument(
        "--configurations",
        nargs="+",
        choices=ALL_CONFIGURATIONS,
        default=list(ALL_CONFIGURATIONS),
        help="Retrieval configurations to compare; defaults to all three.",
    )
    parser.add_argument(
        "--judge-model",
        default=None,
        help="Judge model; defaults to RAGAS_JUDGE_MODEL then LLM_MODEL.",
    )
    parser.add_argument(
        "--confirm-paid-run",
        action="store_true",
        help="Acknowledge paid calls when either answer or judge provider is OpenAI.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    settings = Settings(_env_file=PROJECT_ROOT / ".env")
    judge_provider = settings.ragas_judge_provider or settings.llm_provider
    uses_paid_openai = "openai" in {settings.llm_provider, judge_provider}
    if uses_paid_openai and not args.confirm_paid_run:
        print(
            "Refusing to start paid model calls without --confirm-paid-run.",
            file=sys.stderr,
        )
        return 2

    configure_logging(settings.log_level)
    input_path = args.input.resolve()
    source_dir = args.sources.resolve()
    output_path = args.output.resolve()
    if output_path == input_path:
        print("Evaluation output cannot overwrite the benchmark dataset.", file=sys.stderr)
        return 2

    try:
        validate_evaluation_dataset(input_path, source_dir)
        dataset = read_evaluation_jsonl(input_path)
        llm = create_llm_service(settings)
        retrievers = create_retriever_suite(settings)
        configurations = tuple(args.configurations)
        pipelines = create_evaluation_pipelines(
            retrievers=retrievers,
            llm=llm,
            configurations=configurations,
            top_k=settings.top_k_rerank,
        )
        judge_model = (
            args.judge_model or settings.ragas_judge_model or settings.llm_model
        )
        api_key = (
            settings.llm_api_key.get_secret_value()
            if settings.llm_api_key is not None
            else ""
        )
        scorer = create_ragas_scorer(
            provider=judge_provider,
            api_key=api_key,
            judge_model=judge_model,
            ollama_base_url=settings.ollama_base_url,
            embedding_model=settings.embedding_model,
            embedding_device=settings.embedding_device,
            embedding_batch_size=settings.embedding_batch_size,
            model_cache_dir=resolve_project_path(settings.model_cache_dir),
            ragas_cache_dir=resolve_project_path(settings.ragas_cache_dir),
            timeout_seconds=settings.ragas_timeout_seconds,
            max_retries=settings.ragas_max_retries,
            max_output_tokens=settings.ragas_max_output_tokens,
        )
        results = asyncio.run(
            run_benchmark(
                dataset=dataset,
                pipelines=pipelines,
                scorer=scorer,
                dataset_path=input_path,
                answer_provider=settings.llm_provider,
                answer_model=settings.llm_model,
                judge_provider=judge_provider,
            )
        )
        write_benchmark_results(results, output_path)
    except (
        LLMConfigurationError,
        RAGPipelineError,
        RagasConfigurationError,
        OSError,
        ValueError,
    ) as exc:
        logging.getLogger(__name__).error("Evaluation failed: %s", exc)
        return 1

    print(json.dumps(_summary(results, output_path), indent=2))
    return 1 if has_metric_failures(results) else 0


def _summary(results: BenchmarkResults, output_path: Path) -> dict[str, Any]:
    return {
        "results_file": str(output_path),
        "dataset_cases": results.dataset_case_count,
        "answer_provider": results.answer_provider,
        "answer_model": results.answer_model,
        "judge_provider": results.judge_provider,
        "judge_model": results.judge_model,
        "ragas_version": results.ragas_version,
        "configurations": {
            result.configuration: {
                "application_failures": result.application_failures,
                "expected_source_hit_rate": result.expected_source_hit_rate,
                "answerability_accuracy": result.answerability_accuracy,
                "mean_latency_ms": result.mean_latency_ms,
                "metrics": {
                    name: summary.model_dump()
                    for name, summary in result.metrics.items()
                },
            }
            for result in results.configurations
        },
    }


if __name__ == "__main__":
    raise SystemExit(main())
