"""Offline TASK-026 Query Rewrite + BM25 human-annotation gate."""

from __future__ import annotations

import argparse
import csv
import hashlib
import inspect
import json
import os
import re
import unicodedata
import zipfile
from datetime import date
from pathlib import Path
from posixpath import normpath
from typing import Any
from xml.etree import ElementTree

from healthvault.evaluation.rag.dataset import (
    DEFAULT_DATASET_PATH,
    load_query_relevance_dataset,
)
from healthvault.evaluation.rag.human_annotation import (
    DEFAULT_CANDIDATE_POOL_PATH,
    _ALLOWED_BY_CASE,
    _PROVISIONAL_ANSWERABLE,
    _load_json,
    _validate_candidate_pool,
)
from healthvault.evaluation.used_cases import find_held_out_collisions
from healthvault.evaluation.rag.metrics import aggregate_retrieval_metrics
from healthvault.evaluation.rag.generation import _deterministic_generation_metrics
from healthvault.infrastructure.sqlite_fts_health_knowledge import (
    SQLiteFTSHealthKnowledgeRepository,
)
from healthvault.rag.corpus import DEFAULT_CORPUS_ROOT, load_chunk_artifact


EVAL_VERSION = "rag-eval-r1-v3-human"
TASK_VERSION = "task-026-offline-v1"
TOP_K = 5
PRIMARY_CASE_IDS = (
    "MQ-001", "MQ-006", "MQ-011", "MQ-016", "MQ-019", "MQ-020",
    "RAG-021", "RAG-022", "RAG-023", "RAG-027", "RAG-028", "RAG-029",
    "RAG-031", "RAG-033", "RAG-034", "RAG-037", "RAG-039", "RAG-041",
    "RAG-042", "RAG-054", "RAG-056", "RAG-057", "RAG-059", "RAG-060",
)
SUBSTITUTE_CASE_IDS = (
    "MQ-017", "MQ-018", "RAG-024", "RAG-032", "RAG-036",
    "RAG-043", "RAG-052", "RAG-055", "RAG-058",
)
REWRITE_DEVELOPMENT_QUERY_IDS = (
    "RAG-001", "RAG-002", "RAG-003", "RAG-004", "RAG-005", "RAG-006",
    "RAG-007", "RAG-008", "RAG-009", "RAG-010", "RAG-011", "RAG-012",
    "RAG-013", "RAG-015", "RAG-016", "RAG-019", "RAG-020", "RAG-047",
    "RAG-048", "RAG-053",
)
DEFAULT_OUTPUT_DIR = Path(__file__).parent / "task-026"
_PROJECT_ROOT = Path(__file__).parents[3]
DEFAULT_SIGNED_WORKBOOK = (
    _PROJECT_ROOT / "outputs/task-026-cn/task-026-blind-review-cn-final.xlsx"
)
DEFAULT_GOLD_PATH = DEFAULT_OUTPUT_DIR / "task-026-human-reviewed-gold.json"
DEFAULT_RANKINGS_PATH = DEFAULT_OUTPUT_DIR / "task-026-rankings.json"
DEFAULT_RETRIEVAL_RESULT_PATH = DEFAULT_OUTPUT_DIR / "task-026-retrieval-results.json"
DEFAULT_TASK_027_V3_PATH = (
    Path(__file__).parent / "task-027/task-027-v3-retrieval-results.json"
)
_TASK_024_PATH = _PROJECT_ROOT / "healthvault/evaluation/skill/task-024-heldout-r2-v1.json"
_TASK_025_PATH = _PROJECT_ROOT / "healthvault/evaluation/skill/task-025-heldout-r3-v1.json"

GENERATION_PROMPT_VERSION = "rag-generation-r1-v3-human-v4"
JUDGE_PROMPT_VERSION = "rag-pairwise-judge-r1-v3-human-v3"
GENERATION_INSTRUCTIONS = (
    "Answer only from INPUT_JSON.query and INPUT_JSON.evidence. Return one JSON "
    "object with exactly status, answer, claims, and refusal_reason. status must be "
    "answered or refused. Each claim must contain exactly claim_id, text, and "
    "citation_ids, and may cite only supplied evidence IDs. Refuse when the supplied "
    "evidence is insufficient. For answered, use a non-empty answer, one or more "
    "claims, and null refusal_reason. For refused, use answer as an empty string, "
    "claims as an empty array, and a non-empty refusal_reason; never attach claims "
    "to a refused response. Do not use outside knowledge or hidden labels."
)
JUDGE_INSTRUCTIONS = {
    "faithfulness": (
        "Compare blind Answer A and Answer B only for evidence faithfulness. Prefer "
        "the answer whose claims are directly supported by the supplied evidence. "
        "Do not infer arm identity or independently decide medical facts."
    ),
    "completeness": (
        "Compare blind Answer A and Answer B only for completeness against the "
        "human-reviewed reference claims or refusal target. Do not score style, infer "
        "arm identity, or independently decide medical facts."
    ),
}
JUDGE_DIMENSIONS = ("faithfulness", "completeness")

GENERATION_RESPONSE_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": ["status", "answer", "claims", "refusal_reason"],
    "properties": {
        "status": {"type": "string", "enum": ["answered", "refused"]},
        "answer": {"type": "string"},
        "claims": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["claim_id", "text", "citation_ids"],
                "properties": {
                    "claim_id": {"type": "string", "minLength": 1},
                    "text": {"type": "string", "minLength": 1},
                    "citation_ids": {
                        "type": "array",
                        "items": {"type": "string", "minLength": 1},
                        "uniqueItems": True,
                    },
                },
            },
        },
        "refusal_reason": {"type": ["string", "null"]},
    },
}


def build_generator_system_prompt() -> str:
    """Bind the evidence-only instruction and exact JSON shape into the request."""

    schema = json.dumps(
        GENERATION_RESPONSE_SCHEMA,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return (
        f"{GENERATION_INSTRUCTIONS} Return exactly one JSON object matching this "
        f"schema, with no Markdown or additional fields: {schema}"
    )


def _judge_response_schema(dimension: str) -> dict[str, Any]:
    return {
        "type": "object",
        "additionalProperties": False,
        "required": ["dimension", "winner", "reason", "evidence_citation_ids"],
        "properties": {
            "dimension": {"type": "string", "const": dimension},
            "winner": {"type": "string", "enum": ["A", "B", "tie"]},
            "reason": {"type": "string", "minLength": 1},
            "evidence_citation_ids": {
                "type": "array",
                "items": {"type": "string", "minLength": 1},
                "uniqueItems": True,
            },
        },
    }


def build_judge_system_prompt(dimension: str) -> str:
    """Bind the dimension-specific rubric and exact JSON shape into the request."""

    schema = json.dumps(
        _judge_response_schema(dimension),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return (
        f"{JUDGE_INSTRUCTIONS[dimension]} Return exactly one JSON object matching "
        f"this schema, with no Markdown or additional fields: {schema}"
    )


_DEV_CALIBRATION_PAIRS = (
    {
        "pair_id": "DEV-CAL-001",
        "query": "What may be consumed during a required blood-test fast?",
        "evidence": [{"citation_id": "DEV-CAL-001:E1", "evidence_text": "During the fast, plain water is allowed; coffee and other drinks are not."}],
        "reference_claims": [{"claim_id": "RC1", "text": "Plain water is allowed during the required fast; other drinks are not."}],
        "refusal_target": None,
        "better": "Plain water may be consumed [DEV-CAL-001:E1].",
        "worse": "Coffee with milk is allowed during the fast.",
    },
    {
        "pair_id": "DEV-CAL-002",
        "query": "What period does an A1C result summarize?",
        "evidence": [{"citation_id": "DEV-CAL-002:E1", "evidence_text": "A1C reflects average blood glucose over about the past three months."}],
        "reference_claims": [{"claim_id": "RC1", "text": "A1C reflects average blood glucose over about the past three months."}],
        "refusal_target": None,
        "better": "It summarizes average blood glucose over about three months [DEV-CAL-002:E1].",
        "worse": "It shows only the most recent day.",
    },
    {
        "pair_id": "DEV-CAL-003",
        "query": "What should a patient bring to a doctor visit?",
        "evidence": [{"citation_id": "DEV-CAL-003:E1", "evidence_text": "Bring a list of questions and an up-to-date medicine list."}],
        "reference_claims": [
            {"claim_id": "RC1", "text": "Bring a list of questions."},
            {"claim_id": "RC2", "text": "Bring an up-to-date medicine list."},
        ],
        "refusal_target": None,
        "better": "Bring questions and an up-to-date medicine list [DEV-CAL-003:E1].",
        "worse": "Bring only a payment receipt.",
    },
    {
        "pair_id": "DEV-CAL-004",
        "query": "What thyroid-ultrasound reference range applies?",
        "evidence": [],
        "reference_claims": [],
        "refusal_target": "Refuse because no thyroid-ultrasound evidence is supplied.",
        "better": "The supplied evidence is insufficient, so I cannot provide a range.",
        "worse": "The normal range is exactly 1 to 2 units.",
    },
)

_WORKBOOK_HEADERS = (
    "题号", "中文问题", "英文原问题", "主题", "候选引用ID", "英文标题路径",
    "中文证据译文（AI辅助）", "官方英文证据原文", "可回答？", "证据相关性",
    "原子主张（JSON）", "拒答目标", "操作", "备注", "pool_item_id",
)

# These citations were explicitly identified by the user in the signed workbook
# notes as relevant official evidence outside the blinded Top-5 union. MQ-001's
# noted provisional-only citation is intentionally absent because the user
# explicitly marked it unrelated and requested its removal.
_HUMAN_SUPPLEMENTAL_QRELS: dict[str, tuple[str, ...]] = {
    "MQ-011": (
        "HVK:r1-2026-08-24-v1:niddk-a1c-test:000:614e744d12c8",
    ),
    "RAG-021": (
        "HVK:r1-2026-08-24-v1:medlineplus-fasting-blood-test:001:123cc8fea845",
        "HVK:r1-2026-08-24-v1:niddk-diabetes-tests:003:9a52fd667ccc",
    ),
    "RAG-022": (
        "HVK:r1-2026-08-24-v1:medlineplus-fasting-blood-test:000:ee35a2417bdc",
        "HVK:r1-2026-08-24-v1:medlineplus-fasting-blood-test:004:7d4efe5fcffc",
        "HVK:r1-2026-08-24-v1:niddk-diabetes-tests:003:9a52fd667ccc",
    ),
    "RAG-027": (
        "HVK:r1-2026-08-24-v1:niddk-diabetes-prediabetes-tests:003:ee350877b4db",
    ),
    "RAG-028": (
        "HVK:r1-2026-08-24-v1:niddk-a1c-test:002:458398560009",
    ),
    "RAG-033": (
        "HVK:r1-2026-08-24-v1:medlineplus-lab-preparation:001:46d6705b79b0",
    ),
    "RAG-034": (
        "HVK:r1-2026-08-24-v1:medlineplus-lab-preparation:001:46d6705b79b0",
        "HVK:r1-2026-08-24-v1:medlineplus-fasting-blood-test:000:ee35a2417bdc",
    ),
    "RAG-042": (
        "HVK:r1-2026-08-24-v1:medlineplus-talking-with-doctor:000:7f2bb9d2992a",
    ),
}

# Frozen from the non-Gold rag-eval-r1-v2 dev split. Rules only append lexical
# equivalents; retrieval, aliases, corpus, ranking and Top-K remain unchanged.
REWRITE_RULES: tuple[dict[str, Any], ...] = (
    {
        "rule_id": "fasting-blood-test-v1",
        "triggers": ("blood-test fast", "fasting blood test", "空腹抽血", "化验前空腹"),
        "append": "fasting for a blood test fasting instructions",
    },
    {
        "rule_id": "lipid-profile-v1",
        "triggers": ("lipid profile", "blood cholesterol", "high blood cholesterol", "血脂 profile"),
        "append": "lipid panel lipoprotein panel LDL HDL triglycerides",
    },
    {
        "rule_id": "a1c-v1",
        "triggers": ("a1c", "hba1c", "糖化血红蛋白"),
        "append": "A1C HbA1c hemoglobin A1C glycated hemoglobin",
    },
    {
        "rule_id": "laboratory-preparation-v1",
        "triggers": ("lab prep", "laboratory work", "特别准备", "化验准备"),
        "append": "lab preparation prepare for a lab test test preparation",
    },
    {
        "rule_id": "reference-range-v1",
        "triggers": ("reference range", "参考范围"),
        "append": "lab results understand results laboratory reference range",
    },
    {
        "rule_id": "doctor-visit-v1",
        "triggers": ("seeing my doctor", "doctor visit", "家人陪同", "就诊时"),
        "append": "doctor visit appointment preparation questions for doctor",
    },
)

_GUIDE = """# TASK-026 人工盲审说明

状态：Human Annotation Gate。当前材料不是 Gold，执行者和 AI Reviewer 均未签署结论。

1. 先只打开 `task-026-blind-review.csv`，不要先看 provisional 文件。
2. 每个 `query_id` 的所有行是一道题的 baseline/variant Top-5 去重并集；顺序已确定性打乱，不包含方案身份、原排名或改写查询。
3. 对同一道题填写一致的 `human_answerable`：`true`、`false` 或 `uncertain`。
4. 对每个候选证据填写 `human_relevance`：`relevant`、`not_relevant` 或 `uncertain`。相关证据应能直接支持答案中的至少一个原子主张。
5. answerable 题在 `human_reference_claims` 填 JSON 数组，例如 `[ {"claim_id":"RC1","text":"..."} ]`；只写证据原文直接支持的原子主张。
6. unanswerable 题在 `human_refusal_target` 写清应拒绝的目标；不能用相似但不回答问题的 chunk 代替。
7. `human_action` 逐题填写 `approve`、`correct` 或 `exclude`；`exclude` 时在 notes 写理由，替补只按任务卡固定顺序使用。
8. 完成独立判断后，再打开 `task-026-provisional-evidence.csv` 核对建议、官方 URL 和理由；建议仅供参考，可全部纠正。
9. 把填写后的盲审表交回执行者。只有用户逐题确认后才会冻结 dataset/hash 并计算检索指标。
"""


def _canonical_bytes(value: Any) -> bytes:
    return (json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode(
        "utf-8"
    )


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _normalized(value: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", value).casefold().split())


def deterministic_query_rewrite(query: str) -> str:
    """Append frozen dev-derived equivalents without inspecting labels or hits."""

    if not isinstance(query, str) or not query.strip():
        raise ValueError("query must be non-empty text")
    rewritten = query.replace("(are)", "").replace("  ", " ")
    normalized = _normalized(query)
    additions: list[str] = []
    for rule in REWRITE_RULES:
        if any(_normalized(trigger) in normalized for trigger in rule["triggers"]):
            addition = rule["append"]
            if _normalized(addition) not in normalized and addition not in additions:
                additions.append(addition)
    return rewritten if not additions else f"{rewritten} {' '.join(additions)}"


def _public_provenance(item: dict[str, Any], pool: dict[str, Any]) -> dict[str, Any]:
    source = next(source for source in pool["sources"] if source["dataset"] == item["dataset"])
    return {
        "kind": "public_question",
        "dataset": item["dataset"],
        "dataset_version": source["version"],
        "original_id": item["original_id"],
        "upstream_url": item["upstream_url"],
        "question_sha256": item["question_sha256"],
        "source_file_sha256": item["source_file_sha256"],
        "license": source["license"],
        "license_url": source["license_url"],
    }


def _mq_record(item: dict[str, Any], pool: dict[str, Any]) -> dict[str, Any]:
    claims = [
        {"claim_id": claim_id, "text": text}
        for claim_id, text in _PROVISIONAL_ANSWERABLE.get(item["candidate_id"], ())
    ]
    allowed = list(_ALLOWED_BY_CASE.get(item["candidate_id"], ()))
    answerable = bool(claims)
    return {
        "query_id": item["candidate_id"],
        "case_family_id": f"medquad:{item['original_id']}",
        "query": item["question"],
        "language": "en",
        "topic": item["topic"],
        "public_provenance": _public_provenance(item, pool),
        "provisional": {
            "answerable": answerable,
            "allowed_citations": allowed,
            "reference_claims": claims,
            "refusal_target": None if answerable else (
                "Refuse because the fixed six-topic corpus does not contain enough "
                "evidence to answer the public question without unsupported inference."
            ),
            "rationale": (
                "TASK-014 provisional claims are directly present in the cited official chunks."
                if answerable
                else "The question is outside, or broader than, the fixed corpus evidence."
            ),
        },
    }


def _rag_record(query: Any) -> dict[str, Any]:
    claims = [
        {"claim_id": f"RC{index}", "text": text}
        for index, text in enumerate(query.reference_claims, 1)
    ]
    return {
        "query_id": query.query_id,
        "case_family_id": f"rag-eval-r1-v2:{query.query_id}",
        "query": query.query,
        "language": query.language,
        "topic": query.category,
        "public_provenance": {
            "kind": "existing_rag_candidate",
            "dataset": "rag-eval-r1-v2",
            "dataset_version": "rag-eval-r1-v2",
            "original_id": query.query_id,
            "upstream_url": None,
            "question_sha256": _sha256(query.query.encode("utf-8")),
            "source_file_sha256": _sha256(Path(DEFAULT_DATASET_PATH).read_bytes()),
            "license": "synthetic evaluation query",
            "license_url": None,
        },
        "provisional": {
            "answerable": query.answerable,
            "allowed_citations": list(query.relevance["hybrid_v1"].citation_ids),
            "reference_claims": claims,
            "refusal_target": None if query.answerable else (
                "Refuse because the fixed corpus cannot support this out-of-scope, "
                "patient-specific, diagnostic, treatment, or pathology request."
            ),
            "rationale": (
                "Copied as a provisional suggestion from the existing v2 evidence labels; "
                "TASK-026 still requires independent human confirmation."
            ),
        },
    }


def _external_case_queries() -> list[dict[str, str]]:
    values: list[dict[str, str]] = []
    for path, version in ((_TASK_024_PATH, "task-024"), (_TASK_025_PATH, "task-025")):
        raw = _load_json(path)
        for item in raw["cases"]:
            values.append(
                {
                    "case_id": item["case_id"],
                    "case_family_id": f"{version}:{item['case_id']}",
                    "track": "skill_evolution",
                    "case_version": version,
                    "held_out": True,
                    "query": item["prompt"],
                    "upstream_dataset": version,
                    "upstream_id": item["case_id"],
                }
            )
    return values


def _used_case_collisions(records: list[dict[str, Any]]) -> list[dict[str, str]]:
    collisions: list[dict[str, str]] = []
    external = _external_case_queries()
    for item in records:
        current = {
            "case_id": item["query_id"],
            "case_family_id": item["case_family_id"],
            "track": "rag",
            "case_version": EVAL_VERSION,
            "held_out": True,
            "query": item["query"],
            "upstream_dataset": item["public_provenance"]["dataset"],
            "upstream_id": item["public_provenance"]["original_id"],
        }
        for other in external:
            collisions.extend(find_held_out_collisions([current, other]))
    return collisions


def build_task_026_draft(
    *,
    candidate_pool_path: str | Path = DEFAULT_CANDIDATE_POOL_PATH,
    dataset_path: str | Path = DEFAULT_DATASET_PATH,
    corpus_root: str | Path = DEFAULT_CORPUS_ROOT,
) -> dict[str, Any]:
    pool = _load_json(candidate_pool_path)
    _validate_candidate_pool(pool)
    dataset = load_query_relevance_dataset(dataset_path, corpus_root)
    mq = {item["candidate_id"]: item for item in pool["candidates"]}
    rag = {query.query_id: query for query in dataset.queries}
    records: list[dict[str, Any]] = []
    for case_id in PRIMARY_CASE_IDS:
        if case_id.startswith("MQ-"):
            records.append(_mq_record(mq[case_id], pool))
        else:
            records.append(_rag_record(rag[case_id]))
    collisions = _used_case_collisions(records)
    if collisions:
        raise ValueError(f"TASK-026 used-case collision: {collisions[0]}")
    manifest = _load_json(Path(corpus_root) / "manifest.json")
    return {
        "schema_version": 1,
        "task_version": TASK_VERSION,
        "eval_version": EVAL_VERSION,
        "status": "draft_pending_human_annotation",
        "gold_status": "not_signed",
        "selection": {
            "primary_case_ids": list(PRIMARY_CASE_IDS),
            "substitute_order": list(SUBSTITUTE_CASE_IDS),
            "replacement_allowed_only_after_human_exclude": True,
        },
        "corpus": {
            "corpus_version": manifest["corpus_version"],
            "corpus_sha256": manifest["corpus_sha256"],
            "strategy": "hybrid_v1",
            "artifact_sha256": manifest["strategies"]["hybrid_v1"]["artifact_sha256"],
            "aliases_sha256": manifest["aliases_sha256"],
            "top_k": TOP_K,
        },
        "rewrite_development": {
            "source_eval_version": dataset.eval_version,
            "query_ids": list(REWRITE_DEVELOPMENT_QUERY_IDS),
            "gold_query_ids_used": [],
            "rules_sha256": _sha256(_canonical_bytes(REWRITE_RULES)),
        },
        "source_hashes": {
            "candidate_pool_sha256": _sha256(Path(candidate_pool_path).read_bytes()),
            "rag_eval_v2_sha256": _sha256(Path(dataset_path).read_bytes()),
        },
        "used_case_check": {
            "compared_against": ["task-024-heldout-r2-v1", "task-025-heldout-r3-v1"],
            "method": "nfkc exact plus used_cases near-synonym detector",
            "collisions": collisions,
        },
        "human_annotation_gate": {
            "required_actions": ["approve", "correct", "exclude"],
            "executor_may_sign_gold": False,
            "ai_reviewer_may_sign_gold": False,
            "freeze_allowed": False,
        },
        "queries": records,
    }


def _hit_record(hit: Any) -> dict[str, Any]:
    document = hit.document
    return {
        "citation_id": document.citation_id,
        "title_path": list(document.title_path),
        "evidence_text": document.summary,
        "source_url": document.source.url,
        "publisher": document.source.publisher,
        "body_sha256": document.body_sha256,
    }


def _chunk_record(chunk: Any) -> dict[str, Any]:
    return {
        "citation_id": chunk.citation_id,
        "title_path": list(chunk.title_path),
        "evidence_text": chunk.body,
        "source_url": chunk.url,
        "publisher": chunk.publisher,
        "body_sha256": chunk.body_sha256,
    }


def build_human_annotation_package(
    draft: dict[str, Any],
    *,
    corpus_root: str | Path = DEFAULT_CORPUS_ROOT,
) -> dict[str, Any]:
    if draft.get("gold_status") != "not_signed":
        raise ValueError("only an unsigned draft may enter TASK-026 pooling")
    root = Path(corpus_root)
    artifact = load_chunk_artifact(root / "chunks/hybrid_v1.json")
    chunk_by_citation = {chunk.citation_id: chunk for chunk in artifact.chunks}
    repository = SQLiteFTSHealthKnowledgeRepository(
        artifact_path=root / "chunks/hybrid_v1.json",
        aliases_path=root / "aliases/aliases.json",
    )
    try:
        ranking_queries = []
        pooled_queries = []
        blind_rows: list[dict[str, Any]] = []
        suggestion_rows: list[dict[str, Any]] = []
        for query in draft["queries"]:
            rewritten = deterministic_query_rewrite(query["query"])
            baseline_hits = repository.search(query["query"], TOP_K)
            variant_hits = repository.search(rewritten, TOP_K)
            ranking_queries.append(
                {
                    "query_id": query["query_id"],
                    "baseline": {
                        "query": query["query"],
                        "citation_ids": [hit.document.citation_id for hit in baseline_hits],
                        "scores": [hit.score for hit in baseline_hits],
                    },
                    "variant": {
                        "query": rewritten,
                        "citation_ids": [hit.document.citation_id for hit in variant_hits],
                        "scores": [hit.score for hit in variant_hits],
                    },
                }
            )
            union: dict[str, dict[str, Any]] = {}
            for hit in (*baseline_hits, *variant_hits):
                union.setdefault(hit.document.citation_id, _hit_record(hit))
            candidates = sorted(
                union.values(),
                key=lambda item: _sha256(
                    f"{EVAL_VERSION}|{query['query_id']}|{item['citation_id']}".encode("utf-8")
                ),
            )
            pooled_queries.append(
                {"query_id": query["query_id"], "candidates": candidates}
            )
            review_candidates = candidates or [
                {
                    "citation_id": "",
                    "title_path": [],
                    "evidence_text": "",
                    "source_url": "",
                    "publisher": "",
                    "body_sha256": "",
                }
            ]
            allowed = set(query["provisional"]["allowed_citations"])
            claims = json.dumps(
                query["provisional"]["reference_claims"], ensure_ascii=False
            )
            for candidate in review_candidates:
                pool_item_id = _sha256(
                    f"{EVAL_VERSION}|{query['query_id']}|{candidate['citation_id'] or 'NO_HIT'}".encode(
                        "utf-8"
                    )
                )[:16]
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
                        "human_refusal_target": "",
                        "human_action": "",
                        "human_notes": "",
                    }
                )
                suggestion_rows.append(
                    {
                        "pool_item_id": pool_item_id,
                        "query_id": query["query_id"],
                        "evidence_role": "pooled_candidate",
                        "in_blind_pool": "true",
                        "source_kind": query["public_provenance"]["kind"],
                        "source_original_id": query["public_provenance"]["original_id"],
                        "provisional_answerable": str(
                            query["provisional"]["answerable"]
                        ).lower(),
                        "provisional_relevance": (
                            "relevant"
                            if candidate["citation_id"] in allowed
                            else "not_relevant"
                        ),
                        "candidate_citation_id": candidate["citation_id"],
                        "candidate_title_path": " / ".join(candidate["title_path"]),
                        "evidence_text": candidate["evidence_text"],
                        "source_url": candidate["source_url"],
                        "provisional_qrels": json.dumps(
                            query["provisional"]["allowed_citations"], ensure_ascii=False
                        ),
                        "reference_claims": claims,
                        "refusal_target": query["provisional"]["refusal_target"] or "",
                        "rationale": query["provisional"]["rationale"],
                    }
                )
            for citation_id in sorted(allowed - set(union)):
                reference = _chunk_record(chunk_by_citation[citation_id])
                suggestion_rows.append(
                    {
                        "pool_item_id": "",
                        "query_id": query["query_id"],
                        "evidence_role": "provisional_reference_not_retrieved",
                        "in_blind_pool": "false",
                        "source_kind": query["public_provenance"]["kind"],
                        "source_original_id": query["public_provenance"]["original_id"],
                        "provisional_answerable": str(
                            query["provisional"]["answerable"]
                        ).lower(),
                        "provisional_relevance": "relevant",
                        "candidate_citation_id": reference["citation_id"],
                        "candidate_title_path": " / ".join(reference["title_path"]),
                        "evidence_text": reference["evidence_text"],
                        "source_url": reference["source_url"],
                        "provisional_qrels": json.dumps(
                            query["provisional"]["allowed_citations"], ensure_ascii=False
                        ),
                        "reference_claims": claims,
                        "refusal_target": query["provisional"]["refusal_target"] or "",
                        "rationale": (
                            f"{query['provisional']['rationale']} This cited reference was not "
                            "retrieved by either Top-5 arm and is not part of the blind pool."
                        ),
                    }
                )
        repository_config = dict(repository.index_metadata)
    finally:
        repository.close()
    rules_sha = _sha256(_canonical_bytes(REWRITE_RULES))
    rankings = {
        "schema_version": 1,
        "task_version": TASK_VERSION,
        "eval_version": EVAL_VERSION,
        "status": "provisional_pending_human_annotation",
        "corpus_strategy": "hybrid_v1",
        "top_k": TOP_K,
        "arms": {
            "baseline": {
                "query_transform": "identity",
                "repository": repository_config,
            },
            "variant": {
                "query_transform": "deterministic_query_rewrite",
                "rewrite_rules_sha256": rules_sha,
                "repository": repository_config,
            },
        },
        "queries": ranking_queries,
    }
    pooled = {
        "schema_version": 1,
        "task_version": TASK_VERSION,
        "eval_version": EVAL_VERSION,
        "status": "pooled_blind_pending_human_annotation",
        "pooling": "deduplicated_union_of_baseline_and_variant_top_5",
        "order": "sha256_blinded",
        "arm_identity_present": False,
        "rank_present": False,
        "queries": pooled_queries,
    }
    return {
        "rankings": rankings,
        "pooled": pooled,
        "blind_rows": blind_rows,
        "suggestion_rows": suggestion_rows,
    }


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_human_annotation_package(
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    *,
    candidate_pool_path: str | Path = DEFAULT_CANDIDATE_POOL_PATH,
    dataset_path: str | Path = DEFAULT_DATASET_PATH,
    corpus_root: str | Path = DEFAULT_CORPUS_ROOT,
) -> dict[str, Any]:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    draft = build_task_026_draft(
        candidate_pool_path=candidate_pool_path,
        dataset_path=dataset_path,
        corpus_root=corpus_root,
    )
    package = build_human_annotation_package(draft, corpus_root=corpus_root)
    paths = {
        "task-026-draft.json": _canonical_bytes(draft),
        "task-026-rankings.json": _canonical_bytes(package["rankings"]),
        "task-026-pooled-blind.json": _canonical_bytes(package["pooled"]),
        "task-026-human-annotation-guide.md": _GUIDE.encode("utf-8"),
    }
    for name, content in paths.items():
        (output / name).write_bytes(content)
    blind_path = output / "task-026-blind-review.csv"
    suggestion_path = output / "task-026-provisional-evidence.csv"
    _write_csv(blind_path, package["blind_rows"])
    _write_csv(suggestion_path, package["suggestion_rows"])
    all_paths = [output / name for name in paths] + [blind_path, suggestion_path]
    source_hash = _sha256(inspect.getsource(deterministic_query_rewrite).encode("utf-8"))
    manifest = {
        "schema_version": 1,
        "task_version": TASK_VERSION,
        "eval_version": EVAL_VERSION,
        "status": "human_annotation_gate",
        "gold_status": "not_signed",
        "query_count": len(draft["queries"]),
        "blind_row_count": len(package["blind_rows"]),
        "provisional_row_count": len(package["suggestion_rows"]),
        "corpus": draft["corpus"],
        "baseline": "sqlite_fts5_bm25_trigram",
        "variant": "deterministic_query_rewrite_plus_same_bm25",
        "rewrite_rules_sha256": draft["rewrite_development"]["rules_sha256"],
        "rewrite_function_sha256": source_hash,
        "used_case_collision_count": 0,
        "files": {path.name: _sha256(path.read_bytes()) for path in all_paths},
        "human_gold_signed": False,
        "api_calls": 0,
        "input_tokens": 0,
        "output_tokens": 0,
    }
    (output / "task-026-preflight.json").write_bytes(_canonical_bytes(manifest))
    return manifest


def _xlsx_column_index(reference: str) -> int:
    match = re.match(r"([A-Z]+)", reference)
    if match is None:
        raise ValueError(f"invalid XLSX cell reference: {reference}")
    value = 0
    for character in match.group(1):
        value = value * 26 + ord(character) - ord("A") + 1
    return value - 1


def _xlsx_sheet_rows(path: str | Path, sheet_name: str) -> list[list[str]]:
    """Read plain cell values from one OOXML worksheet without a runtime dependency."""

    workbook_path = Path(path)
    with zipfile.ZipFile(workbook_path) as archive:
        workbook = ElementTree.fromstring(archive.read("xl/workbook.xml"))
        relationships = ElementTree.fromstring(
            archive.read("xl/_rels/workbook.xml.rels")
        )
        relationship_targets = {
            item.attrib["Id"]: item.attrib["Target"] for item in relationships
        }
        relationship_id = None
        for sheet in workbook.findall(
            ".//{http://schemas.openxmlformats.org/spreadsheetml/2006/main}sheet"
        ):
            if sheet.attrib.get("name") == sheet_name:
                relationship_id = sheet.attrib.get(
                    "{http://schemas.openxmlformats.org/officeDocument/2006/relationships}id"
                )
                break
        if relationship_id is None:
            raise ValueError(f"XLSX worksheet is missing: {sheet_name}")
        target = relationship_targets[relationship_id].lstrip("/")
        worksheet_path = target if target.startswith("xl/") else normpath(f"xl/{target}")
        shared_strings: list[str] = []
        if "xl/sharedStrings.xml" in archive.namelist():
            shared = ElementTree.fromstring(archive.read("xl/sharedStrings.xml"))
            for item in shared.findall(
                ".//{http://schemas.openxmlformats.org/spreadsheetml/2006/main}si"
            ):
                shared_strings.append("".join(node.text or "" for node in item.iter() if node.tag.endswith("}t")))
        worksheet = ElementTree.fromstring(archive.read(worksheet_path))
        rows: list[list[str]] = []
        namespace = "{http://schemas.openxmlformats.org/spreadsheetml/2006/main}"
        for row in worksheet.findall(f".//{namespace}sheetData/{namespace}row"):
            values: dict[int, str] = {}
            for cell in row.findall(f"{namespace}c"):
                index = _xlsx_column_index(cell.attrib["r"])
                cell_type = cell.attrib.get("t")
                if cell_type == "inlineStr":
                    value = "".join(
                        node.text or "" for node in cell.iter() if node.tag.endswith("}t")
                    )
                else:
                    raw = cell.find(f"{namespace}v")
                    value = "" if raw is None or raw.text is None else raw.text
                    if cell_type == "s" and value:
                        value = shared_strings[int(value)]
                    elif cell_type == "b":
                        value = "true" if value == "1" else "false"
                values[index] = value
            if values:
                rows.append([values.get(index, "") for index in range(max(values) + 1)])
    return rows


def _parse_reference_claims(raw: str, query_id: str) -> list[dict[str, str]]:
    try:
        claims = json.loads(raw)
    except json.JSONDecodeError as error:
        raise ValueError(f"{query_id} reference claims must be valid JSON") from error
    if not isinstance(claims, list) or not claims:
        raise ValueError(f"{query_id} answerable item requires reference claims")
    parsed: list[dict[str, str]] = []
    for claim in claims:
        if not isinstance(claim, dict) or set(claim) != {"claim_id", "text"}:
            raise ValueError(f"{query_id} claim must use claim_id and text")
        if not all(isinstance(claim[field], str) and claim[field].strip() for field in claim):
            raise ValueError(f"{query_id} claim fields must be non-empty strings")
        parsed.append({"claim_id": claim["claim_id"], "text": claim["text"]})
    claim_ids = [claim["claim_id"] for claim in parsed]
    if len(claim_ids) != len(set(claim_ids)):
        raise ValueError(f"{query_id} claim IDs must be unique")
    return parsed


def build_human_reviewed_gold(
    workbook_path: str | Path = DEFAULT_SIGNED_WORKBOOK,
    *,
    corpus_root: str | Path = DEFAULT_CORPUS_ROOT,
    signed_on: str | None = None,
) -> dict[str, Any]:
    """Validate the user-signed workbook and freeze its decisions as JSON Gold."""

    source = Path(workbook_path)
    resolved_source = source.resolve()
    try:
        source_label = resolved_source.relative_to(_PROJECT_ROOT.resolve()).as_posix()
    except ValueError:
        source_label = resolved_source.as_posix()
    rows = _xlsx_sheet_rows(source, "中文盲审表")
    if not rows or tuple(rows[0]) != _WORKBOOK_HEADERS:
        raise ValueError("signed workbook does not use the frozen TASK-026 columns")
    draft = build_task_026_draft(corpus_root=corpus_root)
    draft_by_id = {item["query_id"]: item for item in draft["queries"]}
    grouped: dict[str, list[dict[str, str]]] = {}
    for values in rows[1:]:
        padded = values + [""] * (len(_WORKBOOK_HEADERS) - len(values))
        row = dict(zip(_WORKBOOK_HEADERS, padded, strict=True))
        query_id = row["题号"].strip()
        if query_id not in draft_by_id:
            raise ValueError(f"unexpected signed query: {query_id}")
        grouped.setdefault(query_id, []).append(row)
    if tuple(grouped) != PRIMARY_CASE_IDS:
        raise ValueError("signed workbook must contain the fixed 24 TASK-026 queries in order")

    artifact = load_chunk_artifact(Path(corpus_root) / "chunks/hybrid_v1.json")
    known_citations = {chunk.citation_id for chunk in artifact.chunks}
    gold_queries: list[dict[str, Any]] = []
    action_counts = {"approve": 0, "correct": 0, "exclude": 0}
    relevance_counts = {"relevant": 0, "not_relevant": 0, "uncertain": 0}
    for query_id in PRIMARY_CASE_IDS:
        query_rows = grouped[query_id]
        answerable_values = {
            _normalized(row["可回答？"])
            for row in query_rows
            if row["可回答？"].strip()
        }
        action_values = {
            _normalized(row["操作"])
            for row in query_rows
            if row["操作"].strip()
        }
        claim_values = {
            row["原子主张（JSON）"].strip()
            for row in query_rows
            if row["原子主张（JSON）"].strip()
        }
        refusal_values = {
            row["拒答目标"].strip()
            for row in query_rows
            if row["拒答目标"].strip()
        }
        note_values = {
            row["备注"].strip() for row in query_rows if row["备注"].strip()
        }
        if len(answerable_values) != 1 or answerable_values.pop() not in {"true", "false"}:
            raise ValueError(f"{query_id} must have one final true/false answerability")
        answerable = _normalized(query_rows[0]["可回答？"]) == "true"
        if len(action_values) != 1 or next(iter(action_values)) not in action_counts:
            raise ValueError(f"{query_id} must have one valid human action")
        action = next(iter(action_values))
        action_counts[action] += 1
        if action == "exclude":
            raise ValueError("excluded queries require the frozen substitute flow")
        judgments = []
        qrels: list[str] = []
        for row in query_rows:
            relevance = _normalized(row["证据相关性"])
            if relevance not in relevance_counts:
                raise ValueError(f"{query_id} has an unresolved relevance label")
            relevance_counts[relevance] += 1
            citation_id = row["候选引用ID"].strip()
            if citation_id and citation_id not in known_citations:
                raise ValueError(f"{query_id} references an unknown citation")
            if relevance == "relevant" and citation_id:
                qrels.append(citation_id)
            judgments.append(
                {
                    "pool_item_id": row["pool_item_id"].strip(),
                    "citation_id": citation_id,
                    "relevance": relevance,
                }
            )
        supplemental = list(_HUMAN_SUPPLEMENTAL_QRELS.get(query_id, ()))
        if any(citation not in known_citations for citation in supplemental):
            raise ValueError(f"{query_id} has an unknown supplemental citation")
        qrels = list(dict.fromkeys([*qrels, *supplemental]))
        if answerable:
            if not qrels:
                raise ValueError(f"{query_id} answerable item requires qrels")
            if len(claim_values) != 1:
                raise ValueError(f"{query_id} requires consistent reference claims")
            claims = _parse_reference_claims(next(iter(claim_values)), query_id)
            refusal_target = None
        else:
            if qrels:
                raise ValueError(f"{query_id} unanswerable item cannot have qrels")
            if claim_values or len(refusal_values) != 1:
                raise ValueError(f"{query_id} unanswerable item requires only a refusal target")
            claims = []
            refusal_target = next(iter(refusal_values))
        source_query = draft_by_id[query_id]
        gold_queries.append(
            {
                "query_id": query_id,
                "case_family_id": source_query["case_family_id"],
                "query": source_query["query"],
                "language": source_query["language"],
                "topic": source_query["topic"],
                "answerable": answerable,
                "qrels": qrels,
                "supplemental_qrels": supplemental,
                "reference_claims": claims,
                "refusal_target": refusal_target,
                "human_action": action,
                "human_notes": next(iter(note_values)) if len(note_values) == 1 else "",
                "evidence_judgments": judgments,
                "public_provenance": source_query["public_provenance"],
            }
        )
    return {
        "schema_version": 1,
        "eval_version": EVAL_VERSION,
        "task_version": TASK_VERSION,
        "status": "human_reviewed_gold",
        "gold_status": "signed",
        "human_confirmation": {
            "signed_by": "user",
            "signed_on": signed_on or date.today().isoformat(),
            "confirmation_text": (
                "确认以 task-026-blind-review-cn-final.xlsx 作为 human-reviewed Gold，"
                "并开始 TASK-026 离线检索评测。"
            ),
            "source_workbook": source_label,
            "source_workbook_sha256": _sha256(source.read_bytes()),
        },
        "corpus": draft["corpus"],
        "rewrite_development": draft["rewrite_development"],
        "source_hashes": draft["source_hashes"],
        "used_case_check": draft["used_case_check"],
        "summary": {
            "query_count": len(gold_queries),
            "answerable_count": sum(query["answerable"] for query in gold_queries),
            "unanswerable_count": sum(not query["answerable"] for query in gold_queries),
            "actions": action_counts,
            "evidence_judgments": relevance_counts,
        },
        "queries": gold_queries,
    }


def evaluate_signed_gold(
    gold: dict[str, Any], rankings: dict[str, Any]
) -> dict[str, Any]:
    """Evaluate the two frozen TASK-026 retrieval arms against one signed Gold."""

    gold_by_id = {query["query_id"]: query for query in gold["queries"]}
    rankings_by_id = {query["query_id"]: query for query in rankings["queries"]}
    if set(gold_by_id) != set(rankings_by_id):
        raise ValueError("rankings and signed Gold query IDs differ")
    arm_results: dict[str, Any] = {}
    query_results: list[dict[str, Any]] = []
    for query_id in PRIMARY_CASE_IDS:
        gold_query = gold_by_id[query_id]
        ranking = rankings_by_id[query_id]
        query_results.append(
            {
                "query_id": query_id,
                "answerable": gold_query["answerable"],
                "qrels": gold_query["qrels"],
                "baseline": ranking["baseline"]["citation_ids"],
                "variant": ranking["variant"]["citation_ids"],
            }
        )
    for arm in ("baseline", "variant"):
        metrics = aggregate_retrieval_metrics(
            (set(item["qrels"]), item[arm]) for item in query_results
        )
        arm_results[arm] = {
            "query_transform": rankings["arms"][arm]["query_transform"],
            "metrics": {
                key: metrics[key]
                for key in (
                    "query_count", "answerable_count", "unanswerable_count",
                    "recall_at_5", "mrr_at_5", "unanswerable_false_positive_rate",
                )
            },
        }
    deltas = {
        metric: round(
            (arm_results["variant"]["metrics"][metric] - arm_results["baseline"]["metrics"][metric]) * 100,
            2,
        )
        for metric in ("recall_at_5", "mrr_at_5", "unanswerable_false_positive_rate")
    }
    return {
        "schema_version": 1,
        "eval_version": EVAL_VERSION,
        "status": "completed_offline",
        "gold_sha256": _sha256(_canonical_bytes(gold)),
        "corpus": gold["corpus"],
        "comparison": "baseline_to_deterministic_query_rewrite_plus_same_bm25",
        "top_k": TOP_K,
        "arms": arm_results,
        "variant_minus_baseline_percentage_points": deltas,
        "queries": query_results,
        "api_calls": 0,
        "input_tokens": 0,
        "output_tokens": 0,
    }


def build_generator_request_payload(
    query: str, evidence: list[dict[str, Any]]
) -> dict[str, Any]:
    """Expose exactly the raw query and retrieved evidence to the Generator."""

    return {"query": query, "evidence": evidence}


def _validate_judge_output(
    value: Any, dimension: str, available_citations: set[str]
) -> dict[str, Any]:
    required = {"dimension", "winner", "reason", "evidence_citation_ids"}
    if not isinstance(value, dict) or set(value) != required:
        raise ValueError("Judge output must use the exact schema")
    if value["dimension"] != dimension or value["winner"] not in {"A", "B", "tie"}:
        raise ValueError("Judge dimension or winner is invalid")
    if not isinstance(value["reason"], str) or not value["reason"].strip():
        raise ValueError("Judge reason must be non-empty")
    citations = value["evidence_citation_ids"]
    if (
        not isinstance(citations, list)
        or len(citations) != len(set(citations))
        or not all(isinstance(item, str) and item in available_citations for item in citations)
    ):
        raise ValueError("Judge evidence citations must be unique and supplied")
    return value


def _balanced_pairwise_schedule(query_ids: list[str]) -> list[dict[str, Any]]:
    schedule = []
    for query_index, query_id in enumerate(query_ids):
        for dimension_index, dimension in enumerate(JUDGE_DIMENSIONS):
            baseline_is_a = (query_index + dimension_index) % 2 == 0
            schedule.append(
                {
                    "query_id": query_id,
                    "dimension": dimension,
                    "answer_a": "baseline" if baseline_is_a else "variant",
                    "answer_b": "variant" if baseline_is_a else "baseline",
                }
            )
    return schedule


def build_generation_contexts(
    gold: dict[str, Any],
    rankings: dict[str, Any],
    final_retrieval_result: dict[str, Any],
    *,
    corpus_root: str | Path = DEFAULT_CORPUS_ROOT,
) -> dict[str, Any]:
    """Bind baseline Top-5 and the final TASK-027 v3 variant Top-5 to evidence."""

    if gold.get("gold_status") != "signed" or len(gold.get("queries", ())) != 24:
        raise ValueError("generation requires the signed 24-query TASK-026 Gold")
    if (
        final_retrieval_result.get("task_version") != "task-027-gate-v3"
        or not final_retrieval_result.get("answerable_top5_unchanged")
        or final_retrieval_result.get("api_calls") != 0
    ):
        raise ValueError("generation requires the final zero-API TASK-027 v3 result")
    gold_sha256 = _sha256(_canonical_bytes(gold))
    if final_retrieval_result.get("source_hashes", {}).get("gold_sha256") != gold_sha256:
        raise ValueError("TASK-027 v3 and TASK-026 Gold hashes differ")

    ranking_by_id = {item["query_id"]: item for item in rankings["queries"]}
    final_by_id = {item["query_id"]: item for item in final_retrieval_result["queries"]}
    expected_ids = {item["query_id"] for item in gold["queries"]}
    if set(ranking_by_id) != expected_ids or set(final_by_id) != expected_ids:
        raise ValueError("Gold, baseline rankings, and TASK-027 v3 query IDs differ")

    root = Path(corpus_root)
    artifact_path = root / "chunks/hybrid_v1.json"
    artifact = load_chunk_artifact(artifact_path)
    chunk_by_id = {chunk.citation_id: chunk for chunk in artifact.chunks}

    def evidence_for(citation_ids: list[str]) -> list[dict[str, Any]]:
        try:
            return [_chunk_record(chunk_by_id[citation_id]) for citation_id in citation_ids]
        except KeyError as exc:
            raise ValueError("retrieval result references an unknown citation") from exc

    queries = []
    for gold_query in gold["queries"]:
        query_id = gold_query["query_id"]
        queries.append(
            {
                "query_id": query_id,
                "query": gold_query["query"],
                "baseline": evidence_for(
                    list(ranking_by_id[query_id]["baseline"]["citation_ids"])
                ),
                "variant": evidence_for(list(final_by_id[query_id]["citation_ids"])),
            }
        )
    return {
        "schema_version": 1,
        "context_version": "task-026-generation-contexts-task-027-v3",
        "query_count": len(queries),
        "arms": ["baseline", "variant"],
        "top_k": TOP_K,
        "generator_visible_fields": ["query", "evidence"],
        "generator_forbidden_fields": [
            "query_id", "answerable", "qrels", "reference_claims",
            "refusal_target", "expected_citation",
        ],
        "source_bindings": {
            "gold_sha256": gold_sha256,
            "baseline_rankings_sha256": _sha256(_canonical_bytes(rankings)),
            "task_027_v3_retrieval_sha256": _sha256(
                _canonical_bytes(final_retrieval_result)
            ),
            "hybrid_artifact_sha256": _sha256(artifact_path.read_bytes()),
        },
        "queries": queries,
    }


def run_fake_judge_calibration() -> dict[str, Any]:
    """Exercise the frozen blind-order and strict-Schema protocol without transport."""

    judgments = []
    for pair_index, pair in enumerate(_DEV_CALIBRATION_PAIRS):
        citations = {item["citation_id"] for item in pair["evidence"]}
        for dimension_index, dimension in enumerate(JUDGE_DIMENSIONS):
            better_is_a = (pair_index + dimension_index) % 2 == 0
            expected_winner = "A" if better_is_a else "B"
            fake_output = {
                "dimension": dimension,
                "winner": expected_winner,
                "reason": "The preferred answer follows all supplied evidence and the other does not.",
                "evidence_citation_ids": sorted(citations),
            }
            _validate_judge_output(fake_output, dimension, citations)
            judgments.append(
                {
                    "pair_id": pair["pair_id"],
                    "dimension": dimension,
                    "answer_a": pair["better"] if better_is_a else pair["worse"],
                    "answer_b": pair["worse"] if better_is_a else pair["better"],
                    "expected_winner": expected_winner,
                    "fake_output": fake_output,
                    "passed": fake_output["winner"] == expected_winner,
                }
            )
    for dimension in JUDGE_DIMENSIONS:
        dimension_rows = [item for item in judgments if item["dimension"] == dimension]
        if sum(item["expected_winner"] == "A" for item in dimension_rows) != 2:
            raise ValueError("dev calibration order is not balanced")
    return {
        "schema_version": 1,
        "status": "passed_fake_protocol_calibration",
        "calibration_version": "task-026-judge-dev-calibration-v1",
        "pair_count": len(_DEV_CALIBRATION_PAIRS),
        "logical_judge_requests": len(judgments),
        "transport_calls": 0,
        "api_calls": 0,
        "input_tokens": 0,
        "output_tokens": 0,
        "pairs": list(_DEV_CALIBRATION_PAIRS),
        "judgments": judgments,
    }


def build_generation_judge_preflight(
    gold: dict[str, Any],
    retrieval_result: dict[str, Any],
    *,
    rankings: dict[str, Any] | None = None,
    final_retrieval_result: dict[str, Any] | None = None,
    contexts: dict[str, Any] | None = None,
    calibration: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Bind frozen inputs and model-separated protocols without authorizing transport."""

    rankings = rankings or _load_json(DEFAULT_RANKINGS_PATH)
    final_retrieval_result = final_retrieval_result or _load_json(
        DEFAULT_TASK_027_V3_PATH
    )
    contexts = contexts or build_generation_contexts(
        gold, rankings, final_retrieval_result
    )
    calibration = calibration or run_fake_judge_calibration()
    query_ids = [item["query_id"] for item in gold["queries"]]
    schedule = _balanced_pairwise_schedule(query_ids)
    for dimension in JUDGE_DIMENSIONS:
        rows = [item for item in schedule if item["dimension"] == dimension]
        if len(rows) != 24 or sum(item["answer_a"] == "baseline" for item in rows) != 12:
            raise ValueError("final Judge A/B order is not balanced")

    generator_config = {
        "client_id": "task-026-generator-only",
        "provider": "zhipu",
        "api_base": "https://open.bigmodel.cn/api/paas/v4",
        "api_key_environment": "HEALTHVAULT_OPENAI_API_KEY",
        "model": "glm-5.3-flash",
        "do_sample": False,
        "temperature": 0.0,
        "thinking": {"type": "enabled"},
        "reasoning_effort": "low",
        "max_output_tokens": 2048,
        "response_format": "json_object",
    }
    judge_config = {
        "client_id": "task-026-judge-only",
        "provider": "zhipu",
        "api_base": "https://open.bigmodel.cn/api/paas/v4",
        "api_key_environment": "HEALTHVAULT_OPENAI_API_KEY",
        "model": "glm-5.3",
        "do_sample": False,
        "temperature": 0.0,
        "thinking": {"type": "enabled"},
        "reasoning_effort": "low",
        "max_output_tokens": 2048,
        "response_format": "json_object",
        "shared_client_with_generator": False,
        "shared_context_with_generator": False,
        "context_reuse_between_calls": False,
    }
    prompt_hashes = {
        "generation": _sha256(build_generator_system_prompt().encode("utf-8")),
        **{
            f"judge_{dimension}": _sha256(
                build_judge_system_prompt(dimension).encode("utf-8")
            )
            for dimension in JUDGE_DIMENSIONS
        },
    }
    schema_hashes = {
        "generation": _sha256(_canonical_bytes(GENERATION_RESPONSE_SCHEMA)),
        **{
            f"judge_{dimension}": _sha256(
                _canonical_bytes(_judge_response_schema(dimension))
            )
            for dimension in JUDGE_DIMENSIONS
        },
    }
    generation_requests = len(query_ids) * 2
    final_judge_requests = len(schedule)
    calibration_judge_requests = calibration["logical_judge_requests"]
    judge_requests = final_judge_requests + calibration_judge_requests
    config = {
        "generator_config": generator_config,
        "judge_config": judge_config,
        "prompt_hashes": prompt_hashes,
        "schema_hashes": schema_hashes,
        "contexts_sha256": _sha256(_canonical_bytes(contexts)),
        "calibration_sha256": _sha256(_canonical_bytes(calibration)),
        "pairwise_schedule_sha256": _sha256(_canonical_bytes(schedule)),
        "gold_sha256": _sha256(_canonical_bytes(gold)),
        "offline_retrieval_result_sha256": _sha256(_canonical_bytes(retrieval_result)),
        "task_027_v3_retrieval_sha256": _sha256(
            _canonical_bytes(final_retrieval_result)
        ),
        "execution_runner_sha256": _sha256(
            Path(__file__).with_name("task_026_run.py").read_bytes()
        ),
        "maximum_model_calls": generation_requests + judge_requests,
        "maximum_transport_attempts": (generation_requests + judge_requests) * 2,
    }
    return {
        "schema_version": 2,
        "status": "api_gate_awaiting_user_approval",
        "execute_authorized": False,
        "api_key_value_recorded": False,
        "generator": generator_config,
        "judge": judge_config,
        "client_isolation": "separate client instances and fresh per-call contexts",
        "generator_input_contract": {
            "visible_fields": ["query", "evidence"],
            "forbidden_fields": [
                "query_id", "answerable", "qrels", "reference_claims",
                "refusal_target", "expected_citation",
            ],
        },
        "deterministic_validation_order": [
            "generation_json_schema", "answer_or_refusal_state", "claim_id",
            "citation_format", "citation_resolvable", "allowed_citation",
        ],
        "generation_response_schema": GENERATION_RESPONSE_SCHEMA,
        "judge_response_schemas": {
            dimension: _judge_response_schema(dimension)
            for dimension in JUDGE_DIMENSIONS
        },
        "prompt_versions": {
            "generation": GENERATION_PROMPT_VERSION,
            "judge": JUDGE_PROMPT_VERSION,
        },
        "prompt_sha256": prompt_hashes,
        "schema_sha256": schema_hashes,
        "context_bindings": contexts["source_bindings"],
        "contexts_sha256": config["contexts_sha256"],
        "dev_calibration": {
            "status": calibration["status"],
            "pair_count": calibration["pair_count"],
            "logical_judge_requests": calibration_judge_requests,
            "sha256": config["calibration_sha256"],
            "real_model_calibration_pending_approval": True,
        },
        "pairwise_order": {
            "blind_labels": ["A", "B"],
            "assignment_visible_to_judge": False,
            "schedule": schedule,
            "schedule_sha256": config["pairwise_schedule_sha256"],
            "baseline_as_a_per_dimension": 12,
            "variant_as_a_per_dimension": 12,
        },
        "queries_per_arm": len(query_ids),
        "logical_generation_requests": generation_requests,
        "logical_final_judge_requests": final_judge_requests,
        "logical_calibration_judge_requests": calibration_judge_requests,
        "logical_judge_requests": judge_requests,
        "maximum_model_calls": config["maximum_model_calls"],
        "maximum_transport_attempts": config["maximum_transport_attempts"],
        "max_transport_retries_per_call": 1,
        "execution_entrypoint": "python -m healthvault.evaluation.rag.task_026 --execute --approved --preflight-sha256 <sha256>",
        "execution_runner_sha256": config["execution_runner_sha256"],
        "maximum_logical_output_tokens": (
            generation_requests * generator_config["max_output_tokens"]
            + judge_requests * judge_config["max_output_tokens"]
        ),
        "run_config_sha256": _sha256(_canonical_bytes(config)),
        "deterministic_validator_sha256": _sha256(
            inspect.getsource(_deterministic_generation_metrics).encode("utf-8")
            + inspect.getsource(_validate_judge_output).encode("utf-8")
        ),
        "api_calls": 0,
        "input_tokens": 0,
        "output_tokens": 0,
        "cost_usd": 0.0,
    }


def write_generation_judge_preflight(
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    *,
    gold_path: str | Path = DEFAULT_GOLD_PATH,
    rankings_path: str | Path = DEFAULT_RANKINGS_PATH,
    retrieval_result_path: str | Path = DEFAULT_RETRIEVAL_RESULT_PATH,
    final_retrieval_path: str | Path = DEFAULT_TASK_027_V3_PATH,
) -> dict[str, Any]:
    """Write the zero-API contexts, fake calibration, preflight, and manifest."""

    gold = _load_json(gold_path)
    rankings = _load_json(rankings_path)
    retrieval_result = _load_json(retrieval_result_path)
    final_retrieval_result = _load_json(final_retrieval_path)
    contexts = build_generation_contexts(gold, rankings, final_retrieval_result)
    calibration = run_fake_judge_calibration()
    preflight = build_generation_judge_preflight(
        gold,
        retrieval_result,
        rankings=rankings,
        final_retrieval_result=final_retrieval_result,
        contexts=contexts,
        calibration=calibration,
    )
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    files = {
        "task-026-generation-contexts-v3.json": contexts,
        "task-026-judge-dev-calibration-v1.json": calibration,
        "task-026-generation-judge-preflight.json": preflight,
    }
    for name, value in files.items():
        (output / name).write_bytes(_canonical_bytes(value))
    manifest = {
        "schema_version": 1,
        "status": "api_gate_awaiting_user_approval",
        "run_config_sha256": preflight["run_config_sha256"],
        "files": {
            name: _sha256((output / name).read_bytes()) for name in files
        },
        "api_calls": 0,
        "input_tokens": 0,
        "output_tokens": 0,
        "cost_usd": 0.0,
    }
    manifest_name = "task-026-generation-judge-run-manifest.json"
    (output / manifest_name).write_bytes(_canonical_bytes(manifest))
    return {
        "contexts": contexts,
        "calibration": calibration,
        "preflight": preflight,
        "manifest": manifest,
    }


def _render_retrieval_summary(result: dict[str, Any]) -> str:
    baseline = result["arms"]["baseline"]["metrics"]
    variant = result["arms"]["variant"]["metrics"]
    delta = result["variant_minus_baseline_percentage_points"]
    return "\n".join(
        (
            f"# TASK-026 offline retrieval: {result['eval_version']}",
            "",
            "- Gold: user-signed human-reviewed dataset, 24 queries",
            "- Corpus: TASK-013 hybrid_v1; Top-K=5",
            "- Variant: deterministic Query Rewrite + the same BM25 repository",
            "- API calls / input tokens / output tokens: 0 / 0 / 0",
            "",
            "| Core metric | baseline | variant | variant - baseline (pp) |",
            "|---|---:|---:|---:|",
            f"| Recall@5 | {baseline['recall_at_5']} | {variant['recall_at_5']} | {delta['recall_at_5']} |",
            f"| MRR@5 | {baseline['mrr_at_5']} | {variant['mrr_at_5']} | {delta['mrr_at_5']} |",
            "| Unanswerable false-positive rate | "
            f"{baseline['unanswerable_false_positive_rate']} | "
            f"{variant['unanswerable_false_positive_rate']} | "
            f"{delta['unanswerable_false_positive_rate']} |",
            "",
        )
    )


def finalize_signed_gold(
    workbook_path: str | Path = DEFAULT_SIGNED_WORKBOOK,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    *,
    signed_on: str | None = None,
) -> dict[str, Any]:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    gold = build_human_reviewed_gold(workbook_path, signed_on=signed_on)
    rankings_path = output / "task-026-rankings.json"
    rankings = _load_json(rankings_path)
    result = evaluate_signed_gold(gold, rankings)
    preflight = build_generation_judge_preflight(gold, result, rankings=rankings)
    files = {
        "task-026-human-reviewed-gold.json": gold,
        "task-026-retrieval-results.json": result,
        "task-026-generation-judge-preflight.json": preflight,
    }
    for name, value in files.items():
        (output / name).write_bytes(_canonical_bytes(value))
    summary_name = "task-026-retrieval-summary.md"
    (output / summary_name).write_text(
        _render_retrieval_summary(result), encoding="utf-8", newline="\n"
    )
    evaluation_config = {
        "eval_version": EVAL_VERSION,
        "gold_sha256": _sha256(_canonical_bytes(gold)),
        "corpus": gold["corpus"],
        "top_k": TOP_K,
        "baseline": "sqlite_fts5_bm25_trigram",
        "variant": "deterministic_query_rewrite_plus_same_bm25",
        "rewrite_rules_sha256": gold["rewrite_development"]["rules_sha256"],
    }
    manifest = {
        "schema_version": 1,
        "task_version": TASK_VERSION,
        "eval_version": EVAL_VERSION,
        "status": "api_gate_awaiting_user_approval",
        "gold_status": "signed",
        "files": {
            name: _sha256((output / name).read_bytes())
            for name in (*files, summary_name)
        },
        "source_workbook_sha256": gold["human_confirmation"]["source_workbook_sha256"],
        "evaluation_source_sha256": _sha256(Path(__file__).read_bytes()),
        "evaluation_config_sha256": _sha256(_canonical_bytes(evaluation_config)),
        "rewrite_rules_sha256": gold["rewrite_development"]["rules_sha256"],
        "rewrite_function_sha256": _sha256(
            inspect.getsource(deterministic_query_rewrite).encode("utf-8")
        ),
        "used_case_collision_count": len(gold["used_case_check"]["collisions"]),
        "api_calls": 0,
        "input_tokens": 0,
        "output_tokens": 0,
    }
    (output / "task-026-run-manifest.json").write_bytes(_canonical_bytes(manifest))
    return {"gold": gold, "retrieval": result, "preflight": preflight, "manifest": manifest}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build the offline TASK-026 annotation gate")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--finalize-gold",
        type=Path,
        help="freeze a user-signed TASK-026 workbook and run the one offline evaluation",
    )
    parser.add_argument(
        "--generation-preflight",
        action="store_true",
        help="write the TASK-027-v3-bound zero-API generation/Judge preflight",
    )
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--approved", action="store_true")
    parser.add_argument("--preflight-sha256", default="")
    parser.add_argument(
        "--preflight-path",
        type=Path,
        default=DEFAULT_OUTPUT_DIR / "task-026-generation-judge-preflight.json",
    )
    parser.add_argument(
        "--run-output",
        type=Path,
        default=DEFAULT_OUTPUT_DIR / "generation-judge-run-v5",
    )
    args = parser.parse_args(argv)
    if args.execute:
        from healthvault.evaluation.rag.task_026_run import execute_task026_sync

        execute_task026_sync(
            preflight_path=args.preflight_path,
            provided_preflight_sha256=args.preflight_sha256,
            execute=True,
            approved=args.approved,
            api_key=os.environ.get("HEALTHVAULT_OPENAI_API_KEY", ""),
            output_dir=args.run_output,
        )
        print(f"status=completed_once output={args.run_output}")
        return 0
    if args.generation_preflight:
        completed = write_generation_judge_preflight(args.output)
        preflight = completed["preflight"]
        print(
            "status=api_gate_awaiting_user_approval api_calls=0 "
            f"generation_calls={preflight['logical_generation_requests']} "
            f"judge_calls={preflight['logical_judge_requests']} "
            f"maximum_model_calls={preflight['maximum_model_calls']}"
        )
        print(f"preflight={args.output / 'task-026-generation-judge-preflight.json'}")
        return 0
    if args.finalize_gold is not None:
        completed = finalize_signed_gold(args.finalize_gold, args.output)
        baseline = completed["retrieval"]["arms"]["baseline"]["metrics"]
        variant = completed["retrieval"]["arms"]["variant"]["metrics"]
        print(
            "status=api_gate gold_status=signed api_calls=0 "
            f"baseline_recall_at_5={baseline['recall_at_5']} "
            f"variant_recall_at_5={variant['recall_at_5']}"
        )
        print(f"gold={args.output / 'task-026-human-reviewed-gold.json'}")
        print(f"preflight={args.output / 'task-026-generation-judge-preflight.json'}")
        return 0
    manifest = write_human_annotation_package(args.output)
    print(
        f"status={manifest['status']} queries={manifest['query_count']} "
        "api_calls=0 gold_status=not_signed"
    )
    print(f"blind_review={args.output / 'task-026-blind-review.csv'}")
    print(f"provisional={args.output / 'task-026-provisional-evidence.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
