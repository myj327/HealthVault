"""Bind the current implementation and replay signed RAG Gold without paid APIs."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path

from healthvault.evaluation.rag.metrics import aggregate_retrieval_metrics
from healthvault.infrastructure.sqlite_fts_health_knowledge import DEFAULT_ALIASES_PATH, DEFAULT_CHUNK_ARTIFACT_PATH, SQLiteFTSHealthKnowledgeRepository
from healthvault.rag.corpus import load_chunk_artifact

ROOT = Path(__file__).resolve().parents[2]
GOLD = ROOT / "healthvault/evaluation/rag/task-026/task-026-human-reviewed-gold.json"
PROTOCOL = "healthvault-current-2026-10-02-v2"


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        stream.write(json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n")


def implementation_binding() -> dict:
    files = [path for directory in ("healthvault", "agents", "tests/healthvault", "web-ui/src")
             for path in (ROOT / directory).rglob("*") if path.suffix in {".py", ".tsx", ".ts", ".css"}]
    files.extend(ROOT / name for name in ("AGENTS.md", "web-ui/package.json", "requirements-web.txt"))
    entries = {path.relative_to(ROOT).as_posix(): digest(path) for path in sorted(files)}
    return {"files": entries, "sha256": hashlib.sha256(json.dumps(entries, sort_keys=True).encode()).hexdigest()}


def prepare(output: Path, *, gold_path: Path = GOLD) -> tuple[dict, dict]:
    gold = json.loads(gold_path.read_text(encoding="utf-8"))
    artifact = load_chunk_artifact(DEFAULT_CHUNK_ARTIFACT_PATH)
    if gold.get("gold_status") not in {"signed", "agent_reviewed"} or digest(DEFAULT_CHUNK_ARTIFACT_PATH) != gold["corpus"]["artifact_sha256"] or digest(DEFAULT_ALIASES_PATH) != gold["corpus"]["aliases_sha256"]:
        raise ValueError("The signed Gold does not match the current immutable corpus/aliases.")
    confirmation = gold.get("human_confirmation") if gold["gold_status"] == "signed" else gold.get("review_confirmation")
    if not confirmation or (gold["gold_status"] == "agent_reviewed" and (confirmation.get("reviewer_kind") != "assistant" or not confirmation.get("user_authorization"))):
        raise ValueError("Gold review identity or explicit authorization is missing.")
    ids = {chunk.chunk_id for chunk in artifact.chunks}
    if any(set(item["qrels"]) - ids for item in gold["queries"]):
        raise ValueError("Gold contains citations absent from the current corpus.")
    manifest = {
        "protocol": PROTOCOL, "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "prepared_pending_independent_review_and_paid_approval",
        "implementation": implementation_binding(), "gold_path": str(gold_path.resolve()),
        "gold_sha256": digest(gold_path), "gold_confirmation": confirmation, "gold_review_kind": gold["gold_status"],
        "corpus": gold["corpus"], "chunk_count": len(artifact.chunks), "query_count": len(gold["queries"]),
        "scope": "regression_on_" + gold["gold_status"] + "_gold", "untouched_held_out": False,
        "retrieval": {"baseline": "lexical", "variant": "hybrid_bge_m3_faiss_rrf", "query_transform": "none", "top_k": 5},
        "paid_evaluation_plan": {
            "status": "not_run", "generator": "glm-5.3-flash", "judge": "glm-5.3",
            "separate_clients_and_contexts": True, "judge_dimensions": ["evidence_faithfulness", "key_point_completeness"],
            "rag_generation_calls": 48, "rag_judge_calls": 48, "dev_calibration_calls": 8,
            "web_answer_verifier_calls": 48,
            "maximum_logical_calls": 152, "max_output_tokens_per_call": 2048,
            "execution_readiness": "planning_only; an approved current-Web runner must enforce these limits",
            "transport_retry_limit": 0, "human_review_min_queries": 10,
            "approval": "required_for_this_new_binding; no_previous_approval_reused",
        },
        "skill_task_benchmark": {"status": "not_run", "reason": "Needs a new current-code protocol review; workflow plan probes are reported separately."},
        "api_calls": 0,
    }
    save(output / "implementation-manifest.json", manifest)
    return gold, manifest


def run_retrieval(output: Path, *, model_path: Path | None, gold_path: Path = GOLD) -> dict:
    gold, manifest = prepare(output, gold_path=gold_path)
    lexical = SQLiteFTSHealthKnowledgeRepository()
    repositories = {"lexical": lexical}
    index_manifest = None
    if model_path is not None:
        for key in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "HF_DATASETS_OFFLINE"):
            os.environ[key] = "1"
        from healthvault.infrastructure.bge_faiss import BGEM3Embedder, FaissVectorIndex
        from healthvault.infrastructure.hybrid_health_knowledge import HybridHealthKnowledgeRepository
        from healthvault.rag.build_hybrid_index import build_index
        print("Building an isolated real BGE-M3/FAISS index; paid API calls=0", flush=True)
        embedder = BGEM3Embedder(model_path)
        index_root = output / "vector-index"
        index_manifest = build_index(artifact_path=DEFAULT_CHUNK_ARTIFACT_PATH, output_directory=index_root, model_path=model_path, embedder=embedder)
        index = FaissVectorIndex.load(index_root / "index.faiss", index_root / "chunk_ids.json", index_manifest.dimension)
        repositories["hybrid"] = HybridHealthKnowledgeRepository(lexical, embedder=embedder, vector_index=index)
    queries = []
    contexts = []
    try:
        for item in gold["queries"]:
            row = {"query_id": item["query_id"], "answerable": item["answerable"], "qrels": item["qrels"]}
            context = {"query_id": item["query_id"], "query": item["query"]}
            for arm, repository in repositories.items():
                started = time.perf_counter()
                hits = repository.search(item["query"], 5)
                row[arm] = {"citation_ids": [hit.document.citation_id for hit in hits], "elapsed_ms": round((time.perf_counter()-started)*1000, 3)}
                context[arm] = [hit.to_dict() for hit in hits]
            queries.append(row); contexts.append(context)
            print(f"Retrieved {len(queries)}/{len(gold['queries'])} queries", flush=True)
    finally:
        lexical.close()
    metrics = {}
    for arm in repositories:
        values = aggregate_retrieval_metrics((set(row["qrels"]), row[arm]["citation_ids"]) for row in queries)
        metrics[arm] = {key: values[key] for key in ("query_count", "answerable_count", "unanswerable_count", "recall_at_5", "mrr_at_5", "unanswerable_false_positive_rate")}
    result = {
        "protocol": PROTOCOL, "implementation_sha256": manifest["implementation"]["sha256"], "gold_sha256": manifest["gold_sha256"],
        "status": "completed_offline_regression", "untouched_held_out": False, "api_calls": 0,
        "index": index_manifest.to_dict() if index_manifest else None, "metrics": metrics, "queries": queries,
        "limits": ["Gold review kind: " + manifest["gold_review_kind"], "Retrieval metrics do not measure generated medical answer correctness", "False positive metric describes retrieval candidates, not final refusal accuracy"],
    }
    save(output / "retrieval-results.json", result)
    save(output / "generation-contexts.json", {"protocol": PROTOCOL, "implementation_sha256": result["implementation_sha256"], "queries": contexts})
    print(json.dumps(metrics, sort_keys=True), flush=True)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, help="New output directory; existing artifacts are never overwritten")
    parser.add_argument("--retrieval", action="store_true")
    parser.add_argument("--model-path", type=Path, help="Use only a pre-downloaded local BGE-M3; no remote download")
    parser.add_argument("--gold", type=Path, default=GOLD)
    args = parser.parse_args()
    output = args.output or ROOT / ".task_outputs" / ("task-066-" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ"))
    if args.retrieval:
        run_retrieval(output, model_path=args.model_path, gold_path=args.gold)
    else:
        _, manifest = prepare(output, gold_path=args.gold)
        print(json.dumps({key: manifest[key] for key in ("protocol", "chunk_count", "query_count", "api_calls")}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
