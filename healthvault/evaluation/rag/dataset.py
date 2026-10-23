"""Strict loader for the frozen query-relevance examination set."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from healthvault.rag.corpus import DEFAULT_CORPUS_ROOT, load_chunk_artifact


_FIXTURE_ROOT = Path(__file__).parents[3] / "tests" / "fixtures" / "rag_eval"
DEFAULT_DATASET_PATH = _FIXTURE_ROOT / "rag-eval-r1-v2.json"
LEGACY_DATASET_PATH = _FIXTURE_ROOT / "rag-eval-r1-v1.json"
TASK013_QUERY_REGISTRY_PATH = _FIXTURE_ROOT / "task-013-query-registry.json"
_STRATEGIES = ("natural_section", "hybrid_v1")
_TOP_FIELDS_V1 = {
    "schema_version", "eval_version", "frozen_on", "corpus_version",
    "corpus_sha256", "aliases_sha256", "artifact_sha256",
    "frozen_test_sha256", "queries",
}
_TOP_FIELDS_V2 = _TOP_FIELDS_V1 | {"task013_query_registry_sha256"}
_QUERY_FIELDS_V1 = {
    "query_id", "query", "language", "category", "answerable", "split",
    "rewrite_category", "relevance", "reference_claims",
}
_QUERY_FIELDS_V2 = _QUERY_FIELDS_V1 | {"judgment_pool"}
_RELEVANCE_FIELDS = {"chunk_ids", "citation_ids"}
_LANGUAGES = {"en", "zh", "mixed"}
_SPLITS = {"dev", "test"}
_REWRITE_CATEGORIES = {"direct", "synonym", "mixed_language", "unanswerable"}


@dataclass(frozen=True)
class Relevance:
    chunk_ids: tuple[str, ...]
    citation_ids: tuple[str, ...]


@dataclass(frozen=True)
class RagQuery:
    query_id: str
    query: str
    language: str
    category: str
    answerable: bool
    split: str
    rewrite_category: str
    relevance: dict[str, Relevance]
    judgment_pool: dict[str, tuple[str, ...]]
    reference_claims: tuple[str, ...]


@dataclass(frozen=True)
class QueryRelevanceDataset:
    schema_version: int
    eval_version: str
    frozen_on: str
    corpus_version: str
    corpus_sha256: str
    aliases_sha256: str
    artifact_sha256: dict[str, str]
    frozen_test_sha256: str
    dataset_sha256: str
    validity_status: str
    invalidated_reason: str | None
    task013_query_registry_sha256: str | None
    queries: tuple[RagQuery, ...]


def _exact_dict(value: Any, fields: set[str], label: str) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != fields:
        raise ValueError(f"{label} must use the exact schema")
    return value


def _text(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} must be a non-empty string")
    return value


def _sha256(value: Any, field: str) -> str:
    text = _text(value, field)
    if len(text) != 64 or any(char not in "0123456789abcdef" for char in text):
        raise ValueError(f"{field} must be a lowercase SHA-256 digest")
    return text


def _text_list(value: Any, field: str, *, allow_empty: bool) -> tuple[str, ...]:
    if not isinstance(value, list) or (not allow_empty and not value):
        raise ValueError(f"{field} must be a list")
    values = tuple(_text(item, field) for item in value)
    if len(set(values)) != len(values):
        raise ValueError(f"{field} must not contain duplicates")
    return values


def _canonical_test_hash(raw_queries: list[dict[str, Any]]) -> str:
    test_queries = [query for query in raw_queries if query.get("split") == "test"]
    payload = json.dumps(
        test_queries, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _normalized(value: str) -> str:
    return " ".join(value.split()).casefold()


def load_query_relevance_dataset(
    path: str | Path = DEFAULT_DATASET_PATH,
    corpus_root: str | Path = DEFAULT_CORPUS_ROOT,
) -> QueryRelevanceDataset:
    dataset_path = Path(path)
    raw_bytes = dataset_path.read_bytes()
    decoded = json.loads(raw_bytes.decode("utf-8"))
    if not isinstance(decoded, dict):
        raise ValueError("dataset must use the exact schema")
    eval_version = decoded.get("eval_version")
    if eval_version == "rag-eval-r1-v1":
        raw = _exact_dict(decoded, _TOP_FIELDS_V1, "dataset")
        query_fields = _QUERY_FIELDS_V1
        validity_status = "invalidated"
        invalidated_reason = (
            "Invalidated after checkpoint review: held-out leakage and incomplete qrels."
        )
        registry_hash = None
    elif eval_version == "rag-eval-r1-v2":
        raw = _exact_dict(decoded, _TOP_FIELDS_V2, "dataset")
        query_fields = _QUERY_FIELDS_V2
        validity_status = "active"
        invalidated_reason = None
        registry_bytes = Path(TASK013_QUERY_REGISTRY_PATH).read_bytes()
        registry_hash = _sha256(
            raw["task013_query_registry_sha256"], "task013_query_registry_sha256"
        )
        if hashlib.sha256(registry_bytes).hexdigest() != registry_hash:
            raise ValueError("TASK-013 query registry hash mismatch")
    else:
        raise ValueError("Unsupported RAG evaluation schema or version")
    if raw["schema_version"] != 1:
        raise ValueError("Unsupported RAG evaluation schema or version")
    root = Path(corpus_root)
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    if raw["corpus_version"] != manifest["corpus_version"]:
        raise ValueError("Evaluation and corpus versions differ")
    if _sha256(raw["corpus_sha256"], "corpus_sha256") != manifest["corpus_sha256"]:
        raise ValueError("Evaluation corpus hash mismatch")
    if _sha256(raw["aliases_sha256"], "aliases_sha256") != manifest["aliases_sha256"]:
        raise ValueError("Evaluation alias hash mismatch")
    artifact_hashes = _exact_dict(raw["artifact_sha256"], set(_STRATEGIES), "artifact hashes")
    chunks = {}
    for strategy in _STRATEGIES:
        expected_hash = manifest["strategies"][strategy]["artifact_sha256"]
        if _sha256(artifact_hashes[strategy], strategy) != expected_hash:
            raise ValueError(f"Evaluation {strategy} artifact hash mismatch")
        artifact = load_chunk_artifact(root / "chunks" / f"{strategy}.json")
        chunks[strategy] = {chunk.chunk_id: chunk for chunk in artifact.chunks}
    if not isinstance(raw["queries"], list):
        raise ValueError("queries must be a list")
    frozen_hash = _sha256(raw["frozen_test_sha256"], "frozen_test_sha256")
    if frozen_hash != _canonical_test_hash(raw["queries"]):
        raise ValueError("Frozen test hash mismatch")
    queries = []
    for item in raw["queries"]:
        item = _exact_dict(item, query_fields, "query")
        query_id = _text(item["query_id"], "query_id")
        language = _text(item["language"], "language")
        split = _text(item["split"], "split")
        rewrite = _text(item["rewrite_category"], "rewrite_category")
        if language not in _LANGUAGES or split not in _SPLITS or rewrite not in _REWRITE_CATEGORIES:
            raise ValueError("Invalid language, split, or rewrite category")
        if not isinstance(item["answerable"], bool):
            raise ValueError("answerable must be boolean")
        relevance_raw = _exact_dict(item["relevance"], set(_STRATEGIES), "relevance")
        relevance = {}
        for strategy in _STRATEGIES:
            strategy_raw = _exact_dict(
                relevance_raw[strategy], _RELEVANCE_FIELDS, f"{strategy} relevance"
            )
            chunk_ids = _text_list(
                strategy_raw["chunk_ids"], "chunk_ids", allow_empty=not item["answerable"]
            )
            citation_ids = _text_list(
                strategy_raw["citation_ids"], "citation_ids", allow_empty=not item["answerable"]
            )
            if chunk_ids != citation_ids:
                raise ValueError("Expected chunk and citation IDs must be equal and ordered")
            if item["answerable"] != bool(chunk_ids):
                raise ValueError("Answerability and relevance must agree")
            if any(chunk_id not in chunks[strategy] for chunk_id in chunk_ids):
                raise ValueError(f"Unknown expected {strategy} chunk")
            relevance[strategy] = Relevance(chunk_ids, citation_ids)
        judgment_pool: dict[str, tuple[str, ...]] = {}
        if eval_version == "rag-eval-r1-v2":
            pool_raw = _exact_dict(
                item["judgment_pool"], set(_STRATEGIES), "judgment_pool"
            )
            for strategy in _STRATEGIES:
                pool_ids = _text_list(
                    pool_raw[strategy], f"{strategy} judgment pool", allow_empty=True
                )
                if any(chunk_id not in chunks[strategy] for chunk_id in pool_ids):
                    raise ValueError(f"Unknown pooled {strategy} chunk")
                if not set(relevance[strategy].chunk_ids).issubset(pool_ids):
                    raise ValueError("Relevant evidence must be in the judgment pool")
                judgment_pool[strategy] = pool_ids
        else:
            judgment_pool = {
                strategy: relevance[strategy].chunk_ids for strategy in _STRATEGIES
            }
        claims = _text_list(
            item["reference_claims"], "reference_claims", allow_empty=not item["answerable"]
        )
        if not item["answerable"] and claims:
            raise ValueError("Unanswerable queries cannot contain reference claims")
        for strategy in _STRATEGIES:
            bodies = " ".join(
                chunks[strategy][chunk_id].body
                for chunk_id in relevance[strategy].chunk_ids
            )
            if any(_normalized(claim) not in _normalized(bodies) for claim in claims):
                raise ValueError(f"Reference claim is absent from {strategy} evidence")
        queries.append(
            RagQuery(
                query_id=query_id,
                query=_text(item["query"], "query"),
                language=language,
                category=_text(item["category"], "category"),
                answerable=item["answerable"],
                split=split,
                rewrite_category=rewrite,
                relevance=relevance,
                judgment_pool=judgment_pool,
                reference_claims=claims,
            )
        )
    if len(queries) != 60 or len({query.query_id for query in queries}) != 60:
        raise ValueError("The fixed evaluation set must contain 60 unique queries")
    expected_counts = {
        "dev": 20,
        "test": 40,
        "answerable": 50,
        "direct": 40,
        "rewrite": 10,
        "unanswerable": 10,
    }
    actual_counts = {
        "dev": sum(query.split == "dev" for query in queries),
        "test": sum(query.split == "test" for query in queries),
        "answerable": sum(query.answerable for query in queries),
        "direct": sum(query.answerable and query.rewrite_category == "direct" for query in queries),
        "rewrite": sum(query.answerable and query.rewrite_category != "direct" for query in queries),
        "unanswerable": sum(not query.answerable for query in queries),
    }
    if actual_counts != expected_counts:
        raise ValueError("Evaluation distribution does not match the frozen design")
    if eval_version == "rag-eval-r1-v2":
        registry = json.loads(Path(TASK013_QUERY_REGISTRY_PATH).read_text(encoding="utf-8"))
        excluded = {_normalized(query) for query in registry["queries"]}
        leaked = {
            _normalized(query.query)
            for query in queries
            if query.split == "test" and _normalized(query.query) in excluded
        }
        if leaked:
            raise ValueError("Held-out test contains a TASK-013 query")
    for attribute in ("language", "category"):
        values = {getattr(query, attribute) for query in queries if query.answerable}
        for value in values:
            splits = {
                query.split
                for query in queries
                if query.answerable and getattr(query, attribute) == value
            }
            if splits != _SPLITS:
                raise ValueError(f"{attribute}={value} is not stratified across splits")
    return QueryRelevanceDataset(
        schema_version=1,
        eval_version=raw["eval_version"],
        frozen_on=_text(raw["frozen_on"], "frozen_on"),
        corpus_version=raw["corpus_version"],
        corpus_sha256=raw["corpus_sha256"],
        aliases_sha256=raw["aliases_sha256"],
        artifact_sha256=dict(artifact_hashes),
        frozen_test_sha256=frozen_hash,
        dataset_sha256=hashlib.sha256(raw_bytes).hexdigest(),
        validity_status=validity_status,
        invalidated_reason=invalidated_reason,
        task013_query_registry_sha256=registry_hash,
        queries=tuple(queries),
    )
