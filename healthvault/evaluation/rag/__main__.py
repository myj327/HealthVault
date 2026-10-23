"""Command-line entry point for reproducible offline RAG evaluation."""

from __future__ import annotations

import argparse
import os
from datetime import datetime
from pathlib import Path

from healthvault.evaluation.rag.dataset import (
    DEFAULT_DATASET_PATH,
    load_query_relevance_dataset,
)
from healthvault.evaluation.rag.retrieval import (
    canonical_json_bytes,
    build_retrieval_run_manifest,
    compare_retrieval_strategies,
    render_markdown_summary,
    TASK_BASELINE_HEAD,
)
from healthvault.evaluation.rag.generation_run import (
    DEFAULT_GENERATION_OUTPUT_DIR,
    DEFAULT_RETRIEVAL_RESULT_PATH,
    execute_generation_run,
    prepare_generation_preflight,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="HealthVault fixed RAG evaluation")
    subparsers = parser.add_subparsers(dest="command", required=True)
    retrieval = subparsers.add_parser(
        "retrieval", help="compare natural_section and hybrid_v1 offline"
    )
    retrieval.add_argument("--dataset", type=Path, default=DEFAULT_DATASET_PATH)
    retrieval.add_argument(
        "--output", type=Path, default=Path(__file__).parent / "results"
    )
    retrieval.add_argument("--code-head", default=TASK_BASELINE_HEAD)
    generation = subparsers.add_parser(
        "generation", help="preflight generation/Judge; execute only when explicit"
    )
    generation.add_argument("--dataset", type=Path, default=DEFAULT_DATASET_PATH)
    generation.add_argument(
        "--retrieval-result", type=Path, default=DEFAULT_RETRIEVAL_RESULT_PATH
    )
    generation.add_argument("--output", type=Path, default=DEFAULT_GENERATION_OUTPUT_DIR)
    generation.add_argument(
        "--execute",
        action="store_true",
        help="perform requests; requires DEEPSEEK_API_KEY and prior user approval",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "retrieval":
        dataset = load_query_relevance_dataset(args.dataset)
        result = compare_retrieval_strategies(dataset)
        args.output.mkdir(parents=True, exist_ok=True)
        (args.output / "retrieval-results.json").write_bytes(canonical_json_bytes(result))
        (args.output / "retrieval-summary.md").write_text(
            render_markdown_summary(result), encoding="utf-8", newline="\n"
        )
        manifest = build_retrieval_run_manifest(
            result,
            executed_at=datetime.now().astimezone().isoformat(timespec="seconds"),
            code_head=args.code_head,
        )
        (args.output / "retrieval-run-manifest.json").write_bytes(
            canonical_json_bytes(manifest)
        )
        print(f"queries={len(dataset.queries)} api_calls=0")
        print(f"result={args.output / 'retrieval-results.json'}")
        return 0
    if args.command == "generation":
        key_configured = "DEEPSEEK_API_KEY" in os.environ
        if not args.execute:
            plan = prepare_generation_preflight(
                dataset_path=args.dataset,
                retrieval_result_path=args.retrieval_result,
                output_dir=args.output,
                key_configured=key_configured,
            )
            print(
                "mode=preflight api_calls=0 "
                f"key_status={plan['api_key_status']} "
                f"max_attempts={plan['maximum_attempts_with_retries']}"
            )
            print(f"plan={args.output / 'generation-preflight.json'}")
            return 0
        api_key = os.environ.get("DEEPSEEK_API_KEY", "")
        if not api_key:
            raise SystemExit("--execute requires DEEPSEEK_API_KEY")
        summary = execute_generation_run(
            dataset_path=args.dataset,
            retrieval_result_path=args.retrieval_result,
            output_dir=args.output,
            api_key=api_key,
        )
        print(f"attempts={summary['ledger']['attempts']}")
        print(f"summary={args.output / 'generation-run-summary.json'}")
        return 0
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
