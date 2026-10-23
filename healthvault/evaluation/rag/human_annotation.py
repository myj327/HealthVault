"""Build the v3-human draft and two offline human-annotation views."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
from typing import Any

from healthvault.infrastructure.sqlite_fts_health_knowledge import (
    SQLiteFTSHealthKnowledgeRepository,
)
from healthvault.rag.corpus import DEFAULT_CORPUS_ROOT, load_chunk_artifact
from healthvault.evaluation.used_cases import normalized_query_sha256, validate_used_case_registry


FIXTURE_ROOT = Path(__file__).parents[3] / "tests" / "fixtures" / "rag_eval"
DEFAULT_CANDIDATE_POOL_PATH = FIXTURE_ROOT / "public-question-candidates-v1.json"
DEFAULT_V3_DRAFT_PATH = FIXTURE_ROOT / "rag-eval-r1-v3-human-draft.json"
DEFAULT_POOLED_DRAFT_PATH = FIXTURE_ROOT / "rag-eval-r1-v3-human-pooled-draft.json"
DEFAULT_BLIND_REVIEW_PATH = FIXTURE_ROOT / "rag-eval-r1-v3-human-blind-review.csv"
DEFAULT_PROVISIONAL_PATH = FIXTURE_ROOT / "rag-eval-r1-v3-human-provisional-evidence.csv"
DEFAULT_ANNOTATION_MANIFEST_PATH = FIXTURE_ROOT / "rag-eval-r1-v3-human-annotation-manifest.json"
DEFAULT_USED_CASE_REGISTRY_PATH = FIXTURE_ROOT.parent / "evaluation" / "used-case-registry-v1.json"
V3_VERSION = "rag-eval-r1-v3-human"

_CITATIONS = {
    "diagnosis_overview": "HVK:r1-2026-08-24-v1:niddk-diabetes-tests:002:46658a772415",
    "diagnosis_confirmation": "HVK:r1-2026-08-24-v1:niddk-diabetes-prediabetes-tests:000:47dd80442ae5",
    "a1c_diagnosis": "HVK:r1-2026-08-24-v1:niddk-a1c-test:002:458398560009",
    "a1c_definition": "HVK:r1-2026-08-24-v1:niddk-a1c-test:000:614e744d12c8",
    "cholesterol_definition": "HVK:r1-2026-08-24-v1:medlineplus-cholesterol-levels:000:2843363a035b",
    "cholesterol_screening": "HVK:r1-2026-08-24-v1:nhlbi-cholesterol-diagnosis:001:d00c9be0722a",
}

_PROVISIONAL_ANSWERABLE: dict[str, tuple[tuple[str, str], ...]] = {
    "MQ-001": (
        ("RC1", "Doctors use a variety of tests to diagnose diabetes and prediabetes."),
        ("RC2", "Diagnosis of type 2 diabetes is definitively made by blood tests."),
    ),
    "MQ-002": (
        ("RC1", "Doctors use a variety of tests to diagnose diabetes and prediabetes."),
    ),
    "MQ-005": (
        ("RC1", "Doctors use a variety of tests to diagnose diabetes and prediabetes."),
    ),
    "MQ-009": (
        ("RC1", "Health care professionals can use the A1C test alone or in combination with other diabetes tests to diagnose type 2 diabetes and prediabetes."),
    ),
    "MQ-011": (
        ("RC1", "The A1C test is also the primary test used for diabetes management."),
    ),
    "MQ-014": (
        ("RC1", "The A1C test can be used to diagnose type 2 diabetes and prediabetes.1"),
    ),
    "MQ-016": (
        ("RC1", "A cholesterol test is a blood test that measures the amount of cholesterol and triglycerides (a type of fat) in your blood."),
    ),
    "MQ-019": (
        ("RC1", "Your healthcare provider may order a blood test called a panel to screen for unhealthy cholesterol levels."),
    ),
}

_ALLOWED_BY_CASE = {
    "MQ-001": (_CITATIONS["diagnosis_overview"], _CITATIONS["diagnosis_confirmation"]),
    "MQ-002": (_CITATIONS["diagnosis_overview"],),
    "MQ-005": (_CITATIONS["diagnosis_overview"],),
    "MQ-009": (_CITATIONS["a1c_diagnosis"],),
    "MQ-011": (_CITATIONS["a1c_definition"],),
    "MQ-014": (_CITATIONS["a1c_definition"],),
    "MQ-016": (_CITATIONS["cholesterol_definition"],),
    "MQ-019": (_CITATIONS["cholesterol_screening"],),
}


def _canonical_bytes(value: Any) -> bytes:
    return (json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode("utf-8")


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _load_json(path: str | Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return value


def _validate_candidate_pool(pool: dict[str, Any]) -> list[dict[str, Any]]:
    required = {
        "schema_version", "pool_version", "acquired_on", "scope", "sources",
        "selection_summary", "scope_gaps", "candidates",
    }
    if set(pool) != required or pool["schema_version"] != 1:
        raise ValueError("public candidate pool must use the exact schema")
    candidates = pool["candidates"]
    if not isinstance(candidates, list) or len(candidates) < 30:
        raise ValueError("public candidate pool is too small")
    selected = [item for item in candidates if item.get("disposition") == "selected_for_v3_draft"]
    reserved = [item for item in candidates if item.get("disposition") == "reserved_metadata_only"]
    if len(selected) < 20 or len(reserved) < 10 or len(reserved) / len(candidates) < 0.20:
        raise ValueError("public candidate reserve does not meet the 20 percent / 10 case rule")
    ids: set[str] = set()
    for item in candidates:
        if item.get("candidate_id") in ids:
            raise ValueError("duplicate public candidate ID")
        ids.add(item["candidate_id"])
        digest = item.get("question_sha256")
        question = item.get("question")
        if question is not None and _sha256_bytes(question.encode("utf-8")) != digest:
            raise ValueError("public candidate question hash mismatch")
        if not item.get("source_file_sha256") or not item.get("upstream_url") or not item.get("original_id"):
            raise ValueError("public candidate provenance is incomplete")
    return selected


def build_v3_human_draft(
    candidate_pool_path: str | Path = DEFAULT_CANDIDATE_POOL_PATH,
    *,
    corpus_root: str | Path = DEFAULT_CORPUS_ROOT,
) -> dict[str, Any]:
    pool = _load_json(candidate_pool_path)
    selected = _validate_candidate_pool(pool)
    root = Path(corpus_root)
    manifest = _load_json(root / "manifest.json")
    artifacts = {
        strategy: load_chunk_artifact(root / "chunks" / f"{strategy}.json")
        for strategy in ("natural_section", "hybrid_v1")
    }
    chunk_by_strategy = {
        strategy: {chunk.chunk_id: chunk for chunk in artifact.chunks}
        for strategy, artifact in artifacts.items()
    }
    queries = []
    for ordinal, item in enumerate(selected, 1):
        candidate_id = item["candidate_id"]
        answerable = candidate_id in _PROVISIONAL_ANSWERABLE
        allowed = list(_ALLOWED_BY_CASE.get(candidate_id, ()))
        for strategy in chunk_by_strategy:
            if any(citation not in chunk_by_strategy[strategy] for citation in allowed):
                raise ValueError(f"provisional citation absent from {strategy}")
        claims = [
            {"claim_id": claim_id, "text": text}
            for claim_id, text in _PROVISIONAL_ANSWERABLE.get(candidate_id, ())
        ]
        for strategy in chunk_by_strategy:
            bodies = " ".join(chunk_by_strategy[strategy][citation].body for citation in allowed)
            if any(claim["text"] not in bodies for claim in claims):
                raise ValueError(f"provisional claim absent from {strategy} evidence")
        queries.append(
            {
                "query_id": f"RAG-H-{ordinal:03d}",
                "case_family_id": f"medquad:{item['original_id']}",
                "query": item["question"],
                "language": "en",
                "topic": item["topic"],
                "split": "pending_human",
                "public_provenance": {
                    "dataset": item["dataset"],
                    "dataset_version": pool["sources"][0]["version"],
                    "original_id": item["original_id"],
                    "original_split": item["original_split"],
                    "upstream_url": item["upstream_url"],
                    "source_file_sha256": item["source_file_sha256"],
                    "question_sha256": item["question_sha256"],
                    "license": pool["sources"][0]["license"],
                    "license_url": pool["sources"][0]["license_url"],
                },
                "provisional": {
                    "answerable": answerable,
                    "reference_claims": claims,
                    "allowed_citations": {
                        "natural_section": allowed,
                        "hybrid_v1": allowed,
                    },
                    "refusal_target": None if answerable else (
                        "Refuse because the TASK-013 corpus does not contain sufficient evidence "
                        "to answer this public question without treatment, causation, symptom, "
                        "patient-specific, or other out-of-scope inference."
                    ),
                    "rationale": (
                        "Exact atomic claims are present in both versioned artifacts."
                        if answerable else
                        "No complete in-scope answer is supported by the six-topic TASK-013 corpus."
                    ),
                },
                "human_review": {
                    "status": "pending",
                    "action": None,
                    "reviewer": None,
                    "reviewed_on": None,
                    "notes": None,
                },
            }
        )
    return {
        "schema_version": 1,
        "eval_version": V3_VERSION,
        "status": "draft_pending_human_annotation",
        "gold_status": "not_signed",
        "created_on": pool["acquired_on"],
        "corpus_version": manifest["corpus_version"],
        "corpus_sha256": manifest["corpus_sha256"],
        "aliases_sha256": manifest["aliases_sha256"],
        "artifact_sha256": {
            strategy: manifest["strategies"][strategy]["artifact_sha256"]
            for strategy in ("natural_section", "hybrid_v1")
        },
        "public_candidate_pool_sha256": _sha256_bytes(Path(candidate_pool_path).read_bytes()),
        "human_annotation_gate": {
            "required_actions": ["approve", "correct", "exclude"],
            "executor_may_sign_gold": False,
            "ai_reviewer_may_sign_gold": False,
            "freeze_allowed": False,
        },
        "queries": queries,
    }


def build_annotation_package(
    draft: dict[str, Any],
    *,
    corpus_root: str | Path = DEFAULT_CORPUS_ROOT,
    k: int = 5,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    if draft.get("status") != "draft_pending_human_annotation" or draft.get("gold_status") != "not_signed":
        raise ValueError("only an unsigned human-annotation draft may be pooled")
    root = Path(corpus_root)
    chunks = {}
    repositories = {}
    try:
        for strategy in ("natural_section", "hybrid_v1"):
            artifact = load_chunk_artifact(root / "chunks" / f"{strategy}.json")
            chunks[strategy] = {chunk.chunk_id: chunk for chunk in artifact.chunks}
            repositories[strategy] = SQLiteFTSHealthKnowledgeRepository(
                artifact_path=root / "chunks" / f"{strategy}.json",
                aliases_path=root / "aliases" / "aliases.json",
            )
        pooled_queries = []
        blind_rows: list[dict[str, Any]] = []
        suggestion_rows: list[dict[str, Any]] = []
        for query in draft["queries"]:
            union: dict[str, dict[str, Any]] = {}
            for strategy, repository in repositories.items():
                for rank, hit in enumerate(repository.search(query["query"], k), 1):
                    citation_id = hit.document.citation_id
                    entry = union.setdefault(
                        citation_id,
                        {
                            "citation_id": citation_id,
                            "strategies": [],
                            "ranks": {},
                            "evidence_text": hit.document.summary,
                            "source_url": hit.document.source.url,
                            "title_path": list(hit.document.title_path),
                        },
                    )
                    entry["strategies"].append(strategy)
                    entry["ranks"][strategy] = rank
            candidates = sorted(
                union.values(),
                key=lambda item: hashlib.sha256(
                    f"{V3_VERSION}|{query['query_id']}|{item['citation_id']}".encode("utf-8")
                ).hexdigest(),
            )
            pooled_queries.append({"query_id": query["query_id"], "candidates": candidates})
            review_candidates = candidates or [{
                "citation_id": "", "evidence_text": "", "source_url": "", "title_path": []
            }]
            allowed = {
                citation
                for values in query["provisional"]["allowed_citations"].values()
                for citation in values
            }
            for candidate in review_candidates:
                pool_item_id = hashlib.sha256(
                    f"{V3_VERSION}|{query['query_id']}|{candidate['citation_id'] or 'NO_HIT'}".encode("utf-8")
                ).hexdigest()[:16]
                blind_rows.append(
                    {
                        "pool_item_id": pool_item_id,
                        "query_id": query["query_id"],
                        "query": query["query"],
                        "topic": query["topic"],
                        "candidate_citation_id": candidate["citation_id"],
                        "candidate_title_path": " / ".join(candidate["title_path"]),
                        "evidence_text": candidate["evidence_text"],
                        "human_answerable": "",
                        "human_relevance": "",
                        "human_reference_claims": "",
                        "human_action": "",
                        "human_notes": "",
                    }
                )
                suggestion_rows.append(
                    {
                        "pool_item_id": pool_item_id,
                        "query_id": query["query_id"],
                        "original_id": query["public_provenance"]["original_id"],
                        "provisional_answerable": query["provisional"]["answerable"],
                        "provisional_relevance": (
                            "relevant" if candidate["citation_id"] in allowed else "not_relevant"
                        ),
                        "candidate_citation_id": candidate["citation_id"],
                        "evidence_text": candidate["evidence_text"],
                        "source_url": candidate["source_url"],
                        "reference_claims": json.dumps(
                            query["provisional"]["reference_claims"], ensure_ascii=False
                        ),
                        "refusal_target": query["provisional"]["refusal_target"] or "",
                        "rationale": query["provisional"]["rationale"],
                    }
                )
        return (
            {
                "schema_version": 1,
                "eval_version": V3_VERSION,
                "status": "pooled_draft_pending_human_annotation",
                "top_k_per_strategy": k,
                "pooling": "union_of_natural_section_and_hybrid_v1_top_k",
                "queries": pooled_queries,
            },
            blind_rows,
            suggestion_rows,
        )
    finally:
        for repository in repositories.values():
            repository.close()


def build_used_case_registry(candidate_pool_path: str | Path = DEFAULT_CANDIDATE_POOL_PATH) -> dict[str, Any]:
    pool = _load_json(candidate_pool_path)
    selected = _validate_candidate_pool(pool)
    reserved = [
        item for item in pool["candidates"]
        if item["disposition"] == "reserved_metadata_only"
    ]
    tracks = [
        {"track": "rag", "case_version": V3_VERSION, "purpose": "retrieval_and_generation"},
        {"track": "agent_e2e", "case_version": "agent-e2e-r1-v1", "purpose": "ordinary_end_to_end"},
        {"track": "long_fault", "case_version": "long-fault-r1-v1", "purpose": "long_horizon_and_faults"},
        {"track": "skill_evolution", "case_version": "skill-evolution-r1-v1", "purpose": "skill_evolution"},
    ]
    entries = []
    for ordinal, item in enumerate(selected, 1):
        query = item["question"]
        entries.append(
            {
                "case_id": f"RAG-H-{ordinal:03d}",
                "case_family_id": f"medquad:{item['original_id']}",
                "track": "rag",
                "case_version": V3_VERSION,
                "split": "pending_human",
                "held_out": False,
                "query": query,
                "query_sha256": normalized_query_sha256(query),
                "upstream_dataset": item["dataset"],
                "upstream_id": item["original_id"],
                "status": "draft_pending_human_annotation",
                "source": item["upstream_url"],
            }
        )
    for ordinal, item in enumerate(reserved, 1):
        track = "long_fault" if ordinal <= len(reserved) // 2 else "skill_evolution"
        version = "long-fault-r1-v1" if track == "long_fault" else "skill-evolution-r1-v1"
        redacted = f"WITHHELD CHQA QUESTION {item['original_id']}"
        entries.append(
            {
                "case_id": f"RES-{ordinal:03d}",
                "case_family_id": f"chqa:{item['original_id']}",
                "track": track,
                "case_version": version,
                "split": "reserve",
                "held_out": False,
                "query": redacted,
                "query_sha256": normalized_query_sha256(redacted),
                "upstream_dataset": item["dataset"],
                "upstream_id": item["original_id"],
                "status": "reserved_metadata_only_license_pending",
                "source": item["upstream_url"],
            }
        )
    return validate_used_case_registry(
        {
            "schema_version": 1,
            "registry_version": "used-cases-r1-v1",
            "created_on": pool["acquired_on"],
            "normalization": "nfkc-casefold-alnum-v1",
            "tracks": tracks,
            "entries": entries,
        }
    )


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_default_annotation_package(*, corpus_root: str | Path = DEFAULT_CORPUS_ROOT) -> dict[str, Any]:
    draft = build_v3_human_draft(corpus_root=corpus_root)
    pooled, blind, provisional = build_annotation_package(draft, corpus_root=corpus_root)
    DEFAULT_V3_DRAFT_PATH.write_bytes(_canonical_bytes(draft))
    DEFAULT_POOLED_DRAFT_PATH.write_bytes(_canonical_bytes(pooled))
    _write_csv(DEFAULT_BLIND_REVIEW_PATH, blind)
    _write_csv(DEFAULT_PROVISIONAL_PATH, provisional)
    registry = build_used_case_registry()
    DEFAULT_USED_CASE_REGISTRY_PATH.parent.mkdir(parents=True, exist_ok=True)
    DEFAULT_USED_CASE_REGISTRY_PATH.write_bytes(_canonical_bytes(registry))
    paths = (
        DEFAULT_V3_DRAFT_PATH, DEFAULT_POOLED_DRAFT_PATH,
        DEFAULT_BLIND_REVIEW_PATH, DEFAULT_PROVISIONAL_PATH,
        DEFAULT_USED_CASE_REGISTRY_PATH,
    )
    manifest = {
        "schema_version": 1,
        "eval_version": V3_VERSION,
        "status": "human_annotation_gate",
        "gold_status": "not_signed",
        "query_count": len(draft["queries"]),
        "blind_row_count": len(blind),
        "provisional_row_count": len(provisional),
        "files": {path.name: _sha256_bytes(path.read_bytes()) for path in paths},
        "api_calls": 0,
        "input_tokens": 0,
        "output_tokens": 0,
        "cost_usd": 0.0,
    }
    DEFAULT_ANNOTATION_MANIFEST_PATH.write_bytes(_canonical_bytes(manifest))
    return manifest
