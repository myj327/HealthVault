"""Create a separately identified Gold from bound human or user-requested assistant review."""
from __future__ import annotations

import copy
import json
from datetime import datetime, timezone
from pathlib import Path

from healthvault.evaluation.current_delivery import digest, save
from healthvault.infrastructure.sqlite_fts_health_knowledge import DEFAULT_CHUNK_ARTIFACT_PATH
from healthvault.rag.corpus import load_chunk_artifact

VERDICTS = {"correct", "incomplete", "unsupported", "citation_error", "appropriate_refusal", "inappropriate_refusal"}
REQUIRED = {"MQ-001", "RAG-027", "RAG-042", "MQ-016"}


def publish_reviewed_gold(source_gold: Path, materials: Path, decisions_path: Path, output: Path,
                          *, human_confirmed_by: str) -> dict:
    """The caller supplies the human's explicit confirmation; agents must not supply it themselves."""
    return _publish(source_gold, materials, decisions_path, output, confirmed_by=human_confirmed_by, review_kind="human")


def publish_agent_reviewed_gold(source_gold: Path, materials: Path, decisions_path: Path, output: Path,
                                *, user_authorization: str) -> dict:
    if not user_authorization.strip():
        raise ValueError("Explicit user authorization for assistant review is required.")
    return _publish(source_gold, materials, decisions_path, output, confirmed_by="Codex", review_kind="assistant",
                    user_authorization=user_authorization)


def _publish(source_gold: Path, materials: Path, decisions_path: Path, output: Path,
             *, confirmed_by: str, review_kind: str, user_authorization: str | None = None) -> dict:
    decisions = json.loads(decisions_path.read_text(encoding="utf-8"))
    old = json.loads(source_gold.read_text(encoding="utf-8"))
    if (not confirmed_by.strip() or decisions.get("status") != ("human_signed" if review_kind == "human" else "agent_reviewed")
            or decisions.get("signed_by") != confirmed_by):
        raise ValueError("Explicit human confirmation matching the signed review is required.")
    if decisions.get("source_gold_sha256") != digest(source_gold) or decisions.get("materials_sha256") != digest(materials):
        raise ValueError("Human review is bound to different materials or Gold.")
    rows = decisions.get("cases")
    if not isinstance(rows, list) or len(rows) < 10:
        raise ValueError("At least ten completed human reviews are required.")
    known = {q["query_id"]: q for q in old["queries"]}
    reviewed = {}
    ids = {chunk.chunk_id for chunk in load_chunk_artifact(DEFAULT_CHUNK_ARTIFACT_PATH).chunks}
    for row in rows:
        if (not isinstance(row, dict) or row.get("query_id") not in known or row["query_id"] in reviewed
                or not isinstance(row.get("answer_A"), str) or row["answer_A"] not in VERDICTS
                or not isinstance(row.get("answer_B"), str) or row["answer_B"] not in VERDICTS
                or not isinstance(row.get("gold_action"), str) or row["gold_action"] not in {"keep", "amend"}):
            raise ValueError("Incomplete, duplicated, or unknown human review case.")
        if row["gold_action"] == "amend":
            qrels, claims, answerable = row.get("qrels"), row.get("reference_claims"), row.get("answerable")
            if (type(answerable) is not bool or not isinstance(qrels, list) or not all(isinstance(i, str) for i in qrels)
                    or len(set(qrels)) != len(qrels) or set(qrels) - ids or not isinstance(claims, list)
                    or not isinstance(row.get("notes"), str) or not row["notes"].strip()):
                raise ValueError("A Gold amendment needs a supported label, citations, and human rationale.")
            for index, claim in enumerate(claims, 1):
                if (not isinstance(claim, dict) or set(claim) != {"claim_id", "text"}
                        or claim["claim_id"] != f"RC{index}" or not isinstance(claim["text"], str)
                        or not claim["text"].strip() or "\n" in claim["text"]):
                    raise ValueError("Reference claims must be nonempty individual facts with ordered IDs.")
            if answerable and (not qrels or not claims or row.get("refusal_target") is not None):
                raise ValueError("Answerable Gold requires relevant evidence and reference facts.")
            if not answerable and (qrels or claims or not isinstance(row.get("refusal_target"), str) or not row["refusal_target"].strip()):
                raise ValueError("Unanswerable Gold requires an explicit refusal target and no relevant facts.")
        reviewed[row["query_id"]] = row
    if not REQUIRED <= set(reviewed):
        raise ValueError("The four identified RAG counterexamples must receive human review.")
    new = copy.deepcopy(old)
    amendments = []
    for query in new["queries"]:
        row = reviewed.get(query["query_id"])
        if row is None or row["gold_action"] == "keep":
            continue
        before = {k: copy.deepcopy(query[k]) for k in ("answerable", "qrels", "reference_claims", "refusal_target")}
        query.update({k: copy.deepcopy(row[k]) for k in before})
        query["review_action"], query["review_notes"] = "correct", row["notes"]
        query["reviewer_kind"] = review_kind
        if review_kind == "human":
            query["human_action"], query["human_notes"] = "correct", row["notes"]
        query["previous_evidence_judgments"] = query["evidence_judgments"]
        query["evidence_judgments"] = [{"citation_id": i, "relevance": "relevant", "source": "TASK-078 explicit " + review_kind + " review"} for i in query["qrels"]]
        query["supplemental_qrels"] = []
        amendments.append({"query_id": query["query_id"], "before": before, "after": {k: query[k] for k in before}, "notes": row["notes"]})
    confirmation = {"confirmed_by": confirmed_by, "reviewer_kind": review_kind, "confirmed_at": datetime.now(timezone.utc).isoformat(),
                    "source_review_file": str(decisions_path.resolve()), "source_review_file_sha256": digest(decisions_path),
                    "materials_sha256": digest(materials), "inherited_gold_sha256": digest(source_gold),
                    "reviewed_query_ids": sorted(reviewed), "user_authorization": user_authorization}
    new.update(eval_version=f"rag-eval-r1-task078-{review_kind}-v1", task_version="task-078-gold-review-v1",
               gold_status="signed" if review_kind == "human" else "agent_reviewed",
               human_confirmation=confirmation if review_kind == "human" else None,
               review_confirmation=confirmation,
               status=f"{review_kind} revisions; explicit inherited labels; iteration, not untouched held-out.")
    counts = lambda g: {"answerable": sum(q["answerable"] for q in g["queries"]), "unanswerable": sum(not q["answerable"] for q in g["queries"])}
    audit = {"source_gold_sha256": digest(source_gold), "decisions_sha256": digest(decisions_path),
             "human_confirmed_by": confirmed_by if review_kind == "human" else None,
             "reviewer_kind": review_kind, "user_authorization": user_authorization, "reviewed_count": len(reviewed),
             "old_denominators": counts(old), "new_denominators": counts(new), "amendments": amendments,
             "metric_policy": "Report label-denominator changes separately from system improvements."}
    new["review_summary"] = audit
    output.mkdir(parents=True, exist_ok=False)
    save(output / "gold.json", new)
    save(output / "amendments.json", audit)
    return audit
