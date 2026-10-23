"""Current Web answer regression on signed Gold and frozen retrieval rankings."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import secrets
import time
import urllib.request
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from urllib.parse import urlparse
from unittest.mock import patch

from healthvault.application.search_health_knowledge import search_health_knowledge
from healthvault.evaluation.current_delivery import ROOT, GOLD, digest, implementation_binding, prepare, save
from healthvault.evaluation.rag.metrics import aggregate_retrieval_metrics
from healthvault.evaluation.rag.task_026 import (
    JUDGE_DIMENSIONS, _balanced_pairwise_schedule, _validate_judge_output,
    build_judge_system_prompt, run_fake_judge_calibration,
)
from healthvault.infrastructure.sqlite_fts_health_knowledge import DEFAULT_ALIASES_PATH, DEFAULT_CHUNK_ARTIFACT_PATH, SQLiteFTSHealthKnowledgeRepository
from healthvault.web.answers import answer_with_model, knowledge_evidence, retrieve_question_knowledge
from healthvault.web.answer_validation import valid_verifier_output
from healthvault.web.model_runtime import configured_web_model

PROTOCOL = "healthvault-web-rag-2026-10-04-v2"
PHASE_CAPS = {"generation": 48, "verification": 48, "judge": 48, "calibration": 8}
MAX_OUTPUT_TOKENS = 2048
MAX_INPUT_BYTES = 80_000
APPROVAL_ROOT = ROOT / ".task_outputs/task-067-approvals"


def value_hash(value: object) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


class FrozenController:
    """Replay immutable hits through the production request/evidence gates."""
    def __init__(self, database: Path, corpus: SQLiteFTSHealthKnowledgeRepository):
        self.record_repository = SimpleNamespace(database_path=database)
        self.corpus = corpus
        self.hits = ()
        self.last_result = {"results": []}

    def select(self, hits: list[dict]) -> None:
        self.hits = tuple(self.corpus.hit_for_chunk_id(hit["citation_id"], rank=index + 1,
                          score=hit["score"], matched_terms=tuple(hit["matched_terms"]))
                          for index, hit in enumerate(hits))
        self.last_result = {"results": []}

    def search(self, _query: str, limit: int):
        return self.hits[:limit]

    def search_knowledge(self, query: str, *, limit: int = 5) -> dict:
        self.last_result = search_health_knowledge({"query": query, "limit": limit}, self).to_dict()
        return self.last_result


def prepare_run(output: Path, retrieval_root: Path, *, api_base: str, gold_path: Path = GOLD) -> dict:
    parsed = urlparse(api_base)
    if parsed.scheme != "https" or not parsed.hostname or parsed.username or parsed.query or parsed.fragment:
        raise ValueError("An HTTPS model endpoint without embedded credentials is required.")
    output.mkdir(parents=True, exist_ok=False)
    gold, binding = prepare(output / "binding", gold_path=gold_path)
    confirmation = binding["gold_confirmation"]
    workbook = Path(confirmation["source_review_file"]) if "source_review_file" in confirmation else ROOT / confirmation["source_workbook"]
    if digest(workbook) != confirmation.get("source_review_file_sha256", confirmation.get("source_workbook_sha256")):
        raise ValueError("Signed Gold workbook hash changed.")
    contexts = json.loads((retrieval_root / "generation-contexts.json").read_text(encoding="utf-8"))
    rankings = json.loads((retrieval_root / "retrieval-results.json").read_text(encoding="utf-8"))
    if rankings["gold_sha256"] != binding["gold_sha256"] or contexts["implementation_sha256"] != rankings["implementation_sha256"]:
        raise ValueError("Retrieval inputs are bound to different Gold or implementations.")
    rows = contexts["queries"]
    if len(rows) != len(gold["queries"]) or len({row["query_id"] for row in rows}) != len(rows):
        raise ValueError("Retrieval query coverage is incomplete or duplicated.")
    by_id = {row["query_id"]: row for row in rows}
    ranking_by_id = {row["query_id"]: row for row in rankings["queries"]}
    gated = []
    corpus = SQLiteFTSHealthKnowledgeRepository()
    controller = FrozenController(output / "unused.sqlite3", corpus)
    try:
        for item in gold["queries"]:
            row = by_id[item["query_id"]]
            if row["query"] != item["query"]:
                raise ValueError("Original question changed.")
            prepared = {"query_id": item["query_id"]}
            for arm in ("lexical", "hybrid"):
                hits = row[arm]
                if [hit["citation_id"] for hit in hits] != ranking_by_id[item["query_id"]][arm]["citation_ids"]:
                    raise ValueError("Saved contexts do not match the frozen rankings.")
                if len(hits) > 5 or len({hit["citation_id"] for hit in hits}) != len(hits):
                    raise ValueError("Invalid top-five rankings.")
                for index, hit in enumerate(hits):
                    canonical = corpus.hit_for_chunk_id(hit["citation_id"], rank=index + 1, score=hit["score"], matched_terms=())
                    if canonical is None or any(hit[key] != canonical.to_dict()[key] for key in ("excerpt", "title", "body_sha256", "corpus_version", "source")):
                        raise ValueError("Retrieved evidence differs from the immutable corpus.")
                controller.select(hits)
                result = retrieve_question_knowledge(controller.search_knowledge, item["query"])
                prepared[arm] = [hit["citation_id"] for hit in result.get("results", [])]
            gated.append(prepared)
    finally:
        corpus.close()
    save(output / "inputs/gold.json", gold)
    save(output / "inputs/contexts.json", contexts)
    save(output / "gated-rankings.json", {"queries": gated, "metrics": {
        arm: aggregate_retrieval_metrics((set(item["qrels"]), row[arm]) for item, row in zip(gold["queries"], gated))
        for arm in ("lexical", "hybrid")}})
    preflight = {
        "protocol": PROTOCOL, "created_at": datetime.now(timezone.utc).isoformat(),
        "implementation_sha256": binding["implementation"]["sha256"],
        "bindings": {**{str((output / name).resolve()): digest(output / name) for name in (
            "inputs/gold.json", "inputs/contexts.json", "gated-rankings.json", "binding/implementation-manifest.json")},
            **{str(name.resolve()): digest(name) for name in (DEFAULT_CHUNK_ARTIFACT_PATH, DEFAULT_ALIASES_PATH, workbook)}},
        "inputs": {"gold": str((output / "inputs/gold.json").resolve()), "contexts": str((output / "inputs/contexts.json").resolve())},
        "endpoint": api_base.rstrip("/") + "/chat/completions",
        "models": {"generation": "glm-5.3-flash", "verification": "glm-5.3-flash", "judge": "glm-5.3", "calibration": "glm-5.3"},
        "parameters": {"temperature": 0, "do_sample": False, "thinking": {"type": "enabled"},
                       "reasoning_effort": "low", "max_tokens": MAX_OUTPUT_TOKENS, "response_format": {"type": "json_object"}, "stream": False},
        "limits": {"phases": PHASE_CAPS, "maximum_calls": sum(PHASE_CAPS.values()), "maximum_attempts": sum(PHASE_CAPS.values()),
                   "transport_retries": 0, "input_bytes_per_call": MAX_INPUT_BYTES, "timeout_seconds": 60},
        "scope": "current_web_api_regression_with_frozen_rankings_and_current_gates", "untouched_held_out": False,
        "approval_required": True, "api_calls": 0, "human_review_min_queries": 10,
        "gold_review_kind": binding["gold_review_kind"],
        "calibration": run_fake_judge_calibration()["pairs"],
        "provider_documentation": "https://docs.z.ai/api-reference/llm/chat-completion",
    }
    preflight["sha256"] = value_hash(preflight)
    save(output / "preflight.json", preflight)
    return preflight


def validate_preflight(path: Path) -> dict:
    plan = json.loads(path.read_text(encoding="utf-8"))
    actual = {key: value for key, value in plan.items() if key != "sha256"}
    if plan["sha256"] != value_hash(actual) or plan["protocol"] != PROTOCOL:
        raise ValueError("Preflight integrity failed.")
    if implementation_binding()["sha256"] != plan["implementation_sha256"]:
        raise ValueError("Implementation changed after preflight.")
    if any(digest(Path(name)) != expected for name, expected in plan["bindings"].items()):
        raise ValueError("Bound input changed after preflight.")
    if any(name not in plan["bindings"] for name in plan["inputs"].values()):
        raise ValueError("Execution inputs are not bound to this preflight.")
    if plan["limits"]["phases"] != PHASE_CAPS or plan["limits"]["transport_retries"] != 0 or plan["parameters"]["max_tokens"] != MAX_OUTPUT_TOKENS:
        raise ValueError("Frozen call limits changed.")
    if plan["calibration"] != run_fake_judge_calibration()["pairs"]:
        raise ValueError("Frozen calibration changed.")
    return plan


class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, *_args, **_kwargs):
        return None


class ModelHTTPClient:
    is_real = True

    def __init__(self, endpoint: str, api_key: str):
        self.endpoint = endpoint
        self.api_key = api_key
        self.opener = urllib.request.build_opener(NoRedirect())

    def complete(self, payload: dict) -> dict:
        request = urllib.request.Request(self.endpoint, data=json.dumps(payload, ensure_ascii=False).encode(),
                  headers={"Authorization": "Bearer " + self.api_key, "Content-Type": "application/json"}, method="POST")
        with self.opener.open(request, timeout=60) as response:
            return json.loads(response.read().decode("utf-8"))


class BoundedCalls:
    def __init__(self, plan: dict, output: Path, clients: dict):
        self.plan, self.output, self.clients = plan, output, clients
        self.calls = []
        self.counts = Counter()
        self.failed = False
        self.current_id = ""

    def append(self, row: dict) -> None:
        with (self.output / "calls.jsonl").open("a", encoding="utf-8", newline="\n") as stream:
            stream.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")

    def call(self, phase: str, system: str, data: dict) -> str:
        if self.failed or self.counts[phase] >= self.plan["limits"]["phases"][phase] or len(self.calls) >= self.plan["limits"]["maximum_calls"]:
            raise RuntimeError("Model call limit or previous transport failure.")
        messages = [{"role": "system", "content": system}, {"role": "user", "content": json.dumps(data, ensure_ascii=False)}]
        if len(json.dumps(messages, ensure_ascii=False).encode()) > MAX_INPUT_BYTES:
            raise ValueError("Model input byte limit exceeded.")
        payload = {**self.plan["parameters"], "model": self.plan["models"][phase], "messages": messages}
        row = {"sequence": len(self.calls) + 1, "case_id": self.current_id, "phase": phase,
               "model": payload["model"], "request_sha256": value_hash(payload), "status": "started",
               "transport_kind": "real" if self.clients[phase].is_real else "fake", "usage": None}
        self.calls.append(row); self.counts[phase] += 1
        self.append(dict(row))
        started = time.perf_counter()
        try:
            response = self.clients[phase].complete(payload)
            usage = response.get("usage", {})
            row["usage"] = {key: usage[key] for key in ("prompt_tokens", "completion_tokens", "total_tokens")
                            if isinstance(usage, dict) and type(usage.get(key)) is int and usage[key] >= 0}
            choice = response["choices"][0]
            text = choice["message"]["content"]
            if isinstance(text, str):
                row["response_sha256"] = hashlib.sha256(text.encode()).hexdigest()
                save(self.output / "responses" / (str(row["sequence"]).zfill(4) + ".json"),
                     {"case_id": self.current_id, "phase": phase, "content": text})
            if choice.get("finish_reason") != "stop" or choice["message"].get("tool_calls") or not isinstance(text, str) or not text.strip():
                raise ValueError("Incomplete model response or unexpected tool call.")
            if phase == "verification":
                parsed = json.loads(text)
                if not valid_verifier_output(parsed):
                    raise ValueError("Verifier protocol failure, not a supported refusal.")
            row["status"] = "completed"
            return text
        except Exception as exc:
            row["status"] = "failed"; row["error_type"] = type(exc).__name__; self.failed = True
            raise
        finally:
            row["latency_ms"] = round((time.perf_counter() - started) * 1000, 3)
            self.append(dict(row))

    def callback(self, phase: str):
        def query(system, data, **_kwargs):
            return self.call(phase, system, data)
        return query


def summary(rows: list[dict]) -> dict:
    result = {}
    for arm in ("lexical", "hybrid"):
        eligible = [row for row in rows if arm in row]
        answered = [row for row in eligible if row[arm].get("status") == "answered"]
        unavailable = [row for row in eligible if not row["answerable"]]
        references = [(row, identifier) for row in answered for identifier in re.findall(r"\[(HVK:[^\]]+)\]", row[arm]["answer"])]
        result[arm] = {"evaluated_queries": len(eligible), "answered": len(answered),
                      "errors": sum(row[arm].get("status") == "error" for row in eligible),
                      "unanswerable_evaluated": len(unavailable),
                      "unanswerable_wrong_answer_rate": sum(row[arm].get("status") == "answered" for row in unavailable) / len(unavailable) if unavailable else None,
                      "unanswerable_correct_refusal_rate": sum(row[arm].get("status") == "insufficient_evidence" for row in unavailable) / len(unavailable) if unavailable else None,
                      "verified_answer_rate": len(answered) / len(eligible) if eligible else None,
                      "citation_membership_rate": sum(identifier in row[arm]["gated_citation_ids"] for row, identifier in references) / len(references) if references else None,
                      "human_gold_citation_relevance_rate": sum(identifier in row["qrels"] for row, identifier in references) / len(references) if references else None}
    return result


def execute_run(path: Path, output: Path, *, approved_sha256: str, reviewed_sha256: str,
                clients: dict | None = None, approval_root: Path = APPROVAL_ROOT) -> dict:
    plan = validate_preflight(path)
    if approved_sha256 != plan["sha256"] or reviewed_sha256 != plan["sha256"]:
        raise ValueError("Current preflight review and explicit paid approval are required.")
    if clients is not None and (set(clients) != set(PHASE_CAPS) or any(client.is_real for client in clients.values())):
        raise ValueError("Injected clients are offline-only; paid clients come from the bound endpoint.")
    config = configured_web_model() if clients is None else None
    if config and (config.api_base.rstrip("/") + "/chat/completions" != plan["endpoint"] or config.model != plan["models"]["generation"]):
        raise ValueError("Configured provider changed after approval.")
    save(approval_root / (plan["sha256"] + ".json"), {"preflight_sha256": plan["sha256"], "output": str(output.resolve())})
    output.mkdir(parents=True, exist_ok=False)
    if clients is None:
        clients = {phase: ModelHTTPClient(plan["endpoint"], config.api_key) for phase in PHASE_CAPS}
    calls = BoundedCalls(plan, output, clients)
    gold = json.loads(Path(plan["inputs"]["gold"]).read_text(encoding="utf-8"))
    contexts = {row["query_id"]: row for row in json.loads(Path(plan["inputs"]["contexts"]).read_text(encoding="utf-8"))["queries"]}
    rows, judgments, calibration = [], [], []
    status, client, corpus, phase = "completed", None, None, "calibration"
    try:
        for pair_index, pair in enumerate(plan["calibration"]):
            calls.current_id = pair["pair_id"]
            for dimension_index, dimension in enumerate(JUDGE_DIMENSIONS):
                better_is_a = (pair_index + dimension_index) % 2 == 0
                data = {key: pair[key] for key in ("query", "evidence", "reference_claims", "refusal_target")}
                data.update(answer_A=pair["better"] if better_is_a else pair["worse"], answer_B=pair["worse"] if better_is_a else pair["better"])
                judged = _validate_judge_output(json.loads(calls.call("calibration", build_judge_system_prompt(dimension), data)), dimension,
                                                {item["citation_id"] for item in pair["evidence"]})
                calibration.append({"pair_id": pair["pair_id"], "dimension": dimension, "passed": judged["winner"] == ("A" if better_is_a else "B"), "judgment": judged})
        if not all(item["passed"] for item in calibration):
            raise ValueError("Real Judge calibration failed; main evaluation stopped.")
        phase = "web_setup"
        with patch.dict(os.environ, {"HEALTHVAULT_UI_DB_PATH": str(output / "import-demo.sqlite3"), "HEALTHVAULT_KNOWLEDGE_MODE": "lexical"}):
            from fastapi.testclient import TestClient
            from healthvault.web.api import create_app
        corpus = SQLiteFTSHealthKnowledgeRepository()
        controller = FrozenController(output / "web.sqlite3", corpus)
        controller.knowledge_release_root = output / "releases"
        def answerer(question, mode, evidence):
            return answer_with_model(question, mode, evidence, query=calls.callback("generation"), profile_id="evaluation")
        app = create_app(controller, model_answerer=answerer, model_answer_verifier=calls.callback("verification"),
                         skill_evolution_root=output / "skills")
        client = TestClient(app, raise_server_exceptions=False)
        registered = client.post("/api/auth/register", json={"username": "evaluation-account", "password": secrets.token_urlsafe(24)})
        registered.raise_for_status()
        client.headers["X-CSRF-Token"] = registered.json()["csrf_token"]
        for item in gold["queries"]:
            phase = "web_answer"
            calls.current_id = item["query_id"]
            row = {"query_id": item["query_id"], "answerable": item["answerable"], "qrels": item["qrels"]}
            rows.append(row)
            for arm in ("lexical", "hybrid"):
                controller.select(contexts[item["query_id"]][arm])
                response = client.post("/api/answers/knowledge", json={"question": item["query"], "model_consent": True, "allow_web_search": False})
                row[arm] = response.json() if response.status_code == 200 else {"status": "error", "http_status": response.status_code}
                row[arm]["gated_citation_ids"] = [hit["citation_id"] for hit in controller.last_result["results"]]
                if calls.failed:
                    raise RuntimeError("Model transport failed; no automatic retry.")
            save(output / (item["query_id"] + ".json"), row)
        by_id = {row["query_id"]: row for row in rows}
        for scheduled in _balanced_pairwise_schedule([row["query_id"] for row in rows]):
            phase = "judge"
            item = next(item for item in gold["queries"] if item["query_id"] == scheduled["query_id"])
            row = by_id[item["query_id"]]
            calls.current_id = item["query_id"]
            if any(row[arm]["status"] == "error" for arm in ("lexical", "hybrid")):
                judgments.append({"query_id": item["query_id"], "dimension": scheduled["dimension"], "status": "not_scored_due_to_answer_error"})
                continue
            evidence = {hit["citation_id"]: {"citation_id": hit["citation_id"], "title": hit["title"], "evidence_text": hit["excerpt"]} for arm in ("lexical", "hybrid") for hit in contexts[item["query_id"]][arm]}
            data = {"query": item["query"], "evidence": list(evidence.values()), "reference_claims": item["reference_claims"], "refusal_target": item["refusal_target"],
                    "answer_A": row["lexical" if scheduled["answer_a"] == "baseline" else "hybrid"]["answer"],
                    "answer_B": row["lexical" if scheduled["answer_b"] == "baseline" else "hybrid"]["answer"]}
            judged = _validate_judge_output(json.loads(calls.call("judge", build_judge_system_prompt(scheduled["dimension"]), data)), scheduled["dimension"], set(evidence))
            judgments.append({**scheduled, "status": "scored", "judgment": judged})
    except Exception as exc:
        status = "stopped_on_error"
        error_type = type(exc).__name__
    finally:
        if client: client.close()
        if corpus: corpus.close()
    if status == "completed" and any(row.get(arm, {}).get("status") == "error" for row in rows for arm in ("lexical", "hybrid")):
        status = "completed_with_errors"
    known_usage = [row["usage"] for row in calls.calls if row.get("usage") and all(key in row["usage"] for key in ("prompt_tokens", "completion_tokens"))]
    result = {"protocol": PROTOCOL, "preflight_sha256": plan["sha256"], "status": status,
              "mode": "real_model_regression" if any(row["transport_kind"] == "real" for row in calls.calls) else "offline_contract_check",
              "api_calls": sum(row["transport_kind"] == "real" for row in calls.calls), "transport_attempts": len(calls.calls),
              "untouched_held_out": False, "completed_queries": sum(all(arm in row for arm in ("lexical", "hybrid")) for row in rows),
              "planned_queries": len(gold["queries"]),
              "rows": rows, "calls": calls.calls, "calibration": calibration, "judgments": judgments, "metrics": summary(rows),
              "provider_usage": {"known_calls": len(known_usage), "unknown_calls": len(calls.calls) - len(known_usage),
                                 "input_tokens": sum(item["prompt_tokens"] for item in known_usage), "output_tokens": sum(item["completion_tokens"] for item in known_usage)},
              "human_review_status": "pending", "human_review_min_queries": 10,
              **({"error_type": error_type, "error_phase": phase} if status == "stopped_on_error" else {})}
    completed_rows = [row for row in rows if all(arm in row for arm in ("lexical", "hybrid"))]
    result["gated_retrieval_metrics"] = {arm: aggregate_retrieval_metrics((set(row["qrels"]), row[arm]["gated_citation_ids"]) for row in completed_rows)
                                         for arm in ("lexical", "hybrid")}
    save(output / "results.json", result)
    selected = sorted(rows, key=lambda row: value_hash([plan["sha256"], row["query_id"]]))[:10]
    selected_ids = {row["query_id"] for row in selected}
    selected_ids.update(row["query_id"] for row in rows if not row["answerable"] or any(row.get(arm, {}).get("status") == "error" for arm in ("lexical", "hybrid")))
    winners = {}
    for judged in judgments:
        if judged["status"] != "scored" or judged["judgment"]["winner"] == "tie": continue
        arm = judged["answer_a"] if judged["judgment"]["winner"] == "A" else judged["answer_b"]
        winners.setdefault(judged["query_id"], set()).add(arm)
    selected_ids.update(identifier for identifier, arms in winners.items() if len(arms) > 1)
    review_cases, key = [], []
    for index, row in enumerate(rows):
        if row["query_id"] not in selected_ids or not all(arm in row for arm in ("lexical", "hybrid")): continue
        a, b = ("lexical", "hybrid") if index % 2 == 0 else ("hybrid", "lexical")
        item = next(item for item in gold["queries"] if item["query_id"] == row["query_id"])
        review_cases.append({"query_id": item["query_id"], "query": item["query"], "answerable": item["answerable"],
                             "reference_claims": item["reference_claims"], "refusal_target": item["refusal_target"],
                             "answer_A": row[a], "answer_B": row[b], "human_verdict": None})
        key.append({"query_id": item["query_id"], "answer_A": a, "answer_B": b})
    save(output / "human-review.json", {"status": "pending", "preflight_sha256": plan["sha256"], "cases": review_cases})
    save(output / "unblinding-key.json", {"preflight_sha256": plan["sha256"], "cases": key})
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepare", action="store_true")
    parser.add_argument("--preflight", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--retrieval-root", type=Path, default=ROOT / ".task_outputs/task-066-current")
    parser.add_argument("--gold", type=Path, default=GOLD)
    parser.add_argument("--approved-sha256")
    parser.add_argument("--reviewed-sha256")
    args = parser.parse_args()
    if args.prepare:
        plan = prepare_run(args.output, args.retrieval_root, api_base=configured_web_model().api_base, gold_path=args.gold)
        result = {"status": "prepared", "preflight_sha256": plan["sha256"], "maximum_api_calls": plan["limits"]["maximum_calls"], "api_calls": 0}
    else:
        if not args.preflight or not args.approved_sha256 or not args.reviewed_sha256:
            parser.error("Execution requires a preflight, its independent review and explicit paid approval.")
        result = execute_run(args.preflight, args.output, approved_sha256=args.approved_sha256, reviewed_sha256=args.reviewed_sha256)
        result = {key: result[key] for key in ("status", "mode", "api_calls", "transport_attempts", "completed_queries")}
    os.write(1, (json.dumps(result, ensure_ascii=False) + "\n").encode())
    return 0 if result["status"] in {"prepared", "completed"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
