"""Validation and byte-stable rendering for the versioned knowledge corpus."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from healthvault.rag.chunking import (
    CORPUS_VERSION,
    MAXIMUM,
    MINIMUM,
    OVERLAP,
    TARGET_MAX,
    TARGET_MIN,
    TOKEN_ESTIMATOR_VERSION,
    KnowledgeChunk,
    SourceBlock,
    SourceSection,
    SourceSnapshot,
    build_chunks,
    estimate_tokens,
    make_chunk_id,
)


DEFAULT_CORPUS_ROOT = Path(__file__).parents[1] / "knowledge" / CORPUS_VERSION

_HEX_64_RE = re.compile(r"[0-9a-f]{64}\Z")
_HEX_12_RE = re.compile(r"[0-9a-f]{12}\Z")
_ARTIFACT_FIELDS = {"schema_version", "corpus_version", "strategy", "chunks"}
_CHUNK_FIELDS = set(KnowledgeChunk.__dataclass_fields__)
_SOURCE_FIELDS = {
    "schema_version", "corpus_version", "source_id", "publisher", "title",
    "version", "url", "language", "retrieved_on", "reuse_basis",
    "source_sha256", "title_path_whitelist", "sections",
}
_SECTION_FIELDS = {"title_path", "topic_tags", "blocks"}
_BLOCK_FIELDS = {"kind", "text"}
_BLOCK_WITH_HEADER_FIELDS = {"kind", "text", "table_header"}
_MANIFEST_FIELDS = {
    "schema_version", "corpus_version", "default_strategy", "source_count",
    "sources", "aliases_sha256", "chunking", "strategies", "corpus_sha256",
}
_MANIFEST_SOURCE_FIELDS = {
    "source_id", "publisher", "title", "version", "url", "language",
    "retrieved_on", "reuse_basis", "source_sha256", "title_path_whitelist",
}
_CHUNKING_FIELDS = {
    "token_estimator", "target_estimated_tokens", "minimum_estimated_tokens",
    "maximum_estimated_tokens", "overlap_estimated_tokens",
    "configured_overlap_estimated_tokens",
}
_STRATEGY_FIELDS = {
    "chunk_count", "minimum_estimated_tokens", "maximum_estimated_tokens",
    "mean_estimated_tokens", "target_range_count", "short_intact_count",
    "artifact_sha256",
}
_STRATEGIES = {"natural_section", "hybrid_v1"}
_EXPECTED_CHUNKING = {
    "token_estimator": TOKEN_ESTIMATOR_VERSION,
    "target_estimated_tokens": [TARGET_MIN, TARGET_MAX],
    "minimum_estimated_tokens": MINIMUM,
    "maximum_estimated_tokens": MAXIMUM,
    "overlap_estimated_tokens": [40, 80],
    "configured_overlap_estimated_tokens": OVERLAP,
}


@dataclass(frozen=True)
class ChunkArtifact:
    schema_version: int
    corpus_version: str
    strategy: str
    chunks: tuple[KnowledgeChunk, ...]


@dataclass(frozen=True)
class ParsedChunkId:
    corpus_version: str
    source_id: str
    ordinal: int
    identity_digest: str


def _canonical_bytes(value: Any) -> bytes:
    return (
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")


def _exact_fields(raw: Any, expected: set[str], label: str) -> dict[str, Any]:
    if not isinstance(raw, dict) or set(raw) != expected:
        raise ValueError(f"{label} must use the exact schema")
    return raw


def _required_text(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} must be a non-empty string")
    return value


def _required_int(value: Any, field: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{field} must be an integer >= {minimum}")
    return value


def _required_sha256(value: Any, field: str) -> str:
    if not isinstance(value, str) or _HEX_64_RE.fullmatch(value) is None:
        raise ValueError(f"{field} must be 64 lowercase hexadecimal characters")
    return value


def _required_https(value: Any, field: str) -> str:
    url = _required_text(value, field)
    parsed = urlparse(url)
    if parsed.scheme != "https" or not parsed.netloc:
        raise ValueError(f"{field} must be absolute HTTPS")
    return url


def _required_text_list(value: Any, field: str) -> tuple[str, ...]:
    if not isinstance(value, list) or not value:
        raise ValueError(f"{field} must be a non-empty list")
    result = tuple(_required_text(item, field) for item in value)
    if len(set(result)) != len(result):
        raise ValueError(f"{field} must not contain duplicates")
    return result


def _source_from_dict(
    raw: dict[str, Any], *, expected_corpus_version: str = CORPUS_VERSION
) -> SourceSnapshot:
    _exact_fields(raw, _SOURCE_FIELDS, "source snapshot")
    if (
        _required_int(raw["schema_version"], "source.schema_version", minimum=1) != 1
        or raw["corpus_version"] != expected_corpus_version
    ):
        raise ValueError("Unsupported source snapshot schema or corpus version")
    source_title = _required_text(raw["title"], "source.title")
    if not isinstance(raw["title_path_whitelist"], list) or not raw["title_path_whitelist"]:
        raise ValueError("source.title_path_whitelist must be a non-empty list")
    title_path_whitelist = tuple(
        _required_text_list(path, "source.title_path_whitelist")
        for path in raw["title_path_whitelist"]
    )
    if len(set(title_path_whitelist)) != len(title_path_whitelist):
        raise ValueError("source.title_path_whitelist must not contain duplicates")
    sections = []
    if not isinstance(raw["sections"], list) or not raw["sections"]:
        raise ValueError("source.sections must be a non-empty list")
    for section_raw in raw["sections"]:
        _exact_fields(section_raw, _SECTION_FIELDS, "source section")
        title_path = _required_text_list(section_raw["title_path"], "title_path")
        if title_path[0] != source_title or any(" | " in item for item in title_path):
            raise ValueError("title_path must preserve the real source hierarchy")
        topic_tags = _required_text_list(section_raw["topic_tags"], "topic_tags")
        if not isinstance(section_raw["blocks"], list) or not section_raw["blocks"]:
            raise ValueError("section.blocks must be a non-empty list")
        blocks = []
        for block in section_raw["blocks"]:
            expected = (
                _BLOCK_WITH_HEADER_FIELDS
                if isinstance(block, dict) and "table_header" in block
                else _BLOCK_FIELDS
            )
            _exact_fields(block, expected, "source block")
            kind = _required_text(block["kind"], "block.kind")
            if kind not in {"paragraph", "list", "table"}:
                raise ValueError("Unsupported source block kind")
            table_header = block.get("table_header")
            if table_header is not None:
                table_header = _required_text(table_header, "block.table_header")
                if kind != "table":
                    raise ValueError("Only table blocks may define table_header")
            blocks.append(
                SourceBlock(
                    kind=kind,
                    text=_required_text(block["text"], "block.text"),
                    table_header=table_header,
                )
            )
        sections.append(SourceSection(title_path, topic_tags, tuple(blocks)))
    if tuple(section.title_path for section in sections) != title_path_whitelist:
        raise ValueError("Source sections must exactly match the title path whitelist")
    content = [
        {
            "blocks": [
                {
                    "kind": block.kind,
                    "text": block.text,
                    **(
                        {"table_header": block.table_header}
                        if block.table_header is not None
                        else {}
                    ),
                }
                for block in section.blocks
            ],
            "title_path": list(section.title_path),
            "topic_tags": list(section.topic_tags),
        }
        for section in sections
    ]
    expected_hash = hashlib.sha256(
        json.dumps(
            content,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    source_hash = _required_sha256(raw["source_sha256"], "source_sha256")
    if source_hash != expected_hash:
        raise ValueError("Source snapshot hash mismatch")
    return SourceSnapshot(
        source_id=_required_text(raw["source_id"], "source_id"),
        publisher=_required_text(raw["publisher"], "publisher"),
        title=source_title,
        version=_required_text(raw["version"], "version"),
        url=_required_https(raw["url"], "source.url"),
        language=_required_text(raw["language"], "language"),
        retrieved_on=_required_text(raw["retrieved_on"], "retrieved_on"),
        reuse_basis=_required_text(raw["reuse_basis"], "reuse_basis"),
        source_sha256=source_hash,
        sections=tuple(sections),
        title_path_whitelist=title_path_whitelist,
    )


def load_source_snapshots(
    root: Path = DEFAULT_CORPUS_ROOT, *, expected_corpus_version: str = CORPUS_VERSION
) -> tuple[SourceSnapshot, ...]:
    paths = sorted((Path(root) / "sources").glob("*.json"))
    sources = tuple(
        _source_from_dict(
            json.loads(path.read_text(encoding="utf-8")),
            expected_corpus_version=expected_corpus_version,
        )
        for path in paths
    )
    if not sources or len({source.source_id for source in sources}) != len(sources):
        raise ValueError("Corpus sources are missing or duplicated")
    return sources


def _chunk_dict(chunk: KnowledgeChunk) -> dict[str, Any]:
    value = asdict(chunk)
    value["topic_tags"] = list(chunk.topic_tags)
    value["title_path"] = list(chunk.title_path)
    return value


def _artifact_dict(
    strategy: str, chunks: tuple[KnowledgeChunk, ...], *, corpus_version: str
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "corpus_version": corpus_version,
        "strategy": strategy,
        "chunks": [_chunk_dict(chunk) for chunk in chunks],
    }


def _strategy_stats(chunks: tuple[KnowledgeChunk, ...]) -> dict[str, Any]:
    if not chunks:
        raise ValueError("A chunk artifact must not be empty")
    sizes = [chunk.estimated_tokens for chunk in chunks]
    return {
        "chunk_count": len(chunks),
        "minimum_estimated_tokens": min(sizes),
        "maximum_estimated_tokens": max(sizes),
        "mean_estimated_tokens": round(sum(sizes) / len(sizes), 3),
        "target_range_count": sum(TARGET_MIN <= size <= TARGET_MAX for size in sizes),
        "short_intact_count": sum(
            chunk.strategy == "hybrid_v1" and chunk.estimated_tokens < MINIMUM
            for chunk in chunks
        ),
    }


def render_corpus_artifacts(
    root: Path = DEFAULT_CORPUS_ROOT,
    *,
    corpus_version: str = CORPUS_VERSION,
) -> dict[str, bytes]:
    root = Path(root)
    sources = load_source_snapshots(root, expected_corpus_version=corpus_version)
    alias_bytes = (root / "aliases" / "aliases.json").read_bytes()
    aliases = json.loads(alias_bytes.decode("utf-8"))
    if aliases.get("corpus_version") != corpus_version:
        raise ValueError("Alias corpus version mismatch")
    natural = build_chunks(sources, "natural_section", corpus_version=corpus_version)
    from healthvault.rag.budget import TokenBudget
    budget = TokenBudget.load(root)
    hybrid = build_chunks(sources, "hybrid_v1", corpus_version=corpus_version, token_budget=budget)
    natural_bytes = _canonical_bytes(_artifact_dict("natural_section", natural, corpus_version=corpus_version))
    hybrid_bytes = _canonical_bytes(_artifact_dict("hybrid_v1", hybrid, corpus_version=corpus_version))
    source_manifest = [
        {
            "source_id": source.source_id,
            "publisher": source.publisher,
            "title": source.title,
            "version": source.version,
            "url": source.url,
            "language": source.language,
            "retrieved_on": source.retrieved_on,
            "reuse_basis": source.reuse_basis,
            "source_sha256": source.source_sha256,
            "title_path_whitelist": [
                list(path) for path in source.title_path_whitelist
            ],
        }
        for source in sources
    ]
    digest_input = "\n".join(
        [
            *(source.source_sha256 for source in sources),
            hashlib.sha256(alias_bytes).hexdigest(),
            hashlib.sha256(natural_bytes).hexdigest(),
            hashlib.sha256(hybrid_bytes).hexdigest(),
        ]
    ).encode("ascii")
    policy_hash = hashlib.sha256((root / "chunk-policy.json").read_bytes()).hexdigest() if budget else None
    if policy_hash:
        digest_input += b"\n" + policy_hash.encode("ascii")
    manifest = {
        "schema_version": 1,
        "corpus_version": corpus_version,
        "default_strategy": "hybrid_v1",
        "source_count": len(sources),
        "sources": source_manifest,
        "aliases_sha256": hashlib.sha256(alias_bytes).hexdigest(),
        "chunking": _EXPECTED_CHUNKING if budget is None else {**_EXPECTED_CHUNKING, "policy_sha256": policy_hash},
        "strategies": {
            "natural_section": {
                **_strategy_stats(natural),
                "artifact_sha256": hashlib.sha256(natural_bytes).hexdigest(),
            },
            "hybrid_v1": {
                **_strategy_stats(hybrid),
                "artifact_sha256": hashlib.sha256(hybrid_bytes).hexdigest(),
            },
        },
        "corpus_sha256": hashlib.sha256(digest_input).hexdigest(),
    }
    artifacts = {
        "chunks/natural_section.json": natural_bytes,
        "chunks/hybrid_v1.json": hybrid_bytes,
        "manifest.json": _canonical_bytes(manifest),
    }
    if budget:
        artifacts["chunks/boundary-spans.json"] = _canonical_bytes({"policy": "bounded-v2", "tokenizer": budget.identity,
            "chunks": [{"chunk_id": chunk.chunk_id, "source_id": chunk.source_id, "title_path": list(chunk.title_path),
                        **span} for chunk, span in zip(hybrid, budget.spans)]})
    return artifacts


def _manifest_path_for_artifact(path: Path) -> Path:
    return (
        path.parent.parent / "manifest.json"
        if path.parent.name == "chunks"
        else path.parent / "manifest.json"
    )


def _corpus_root_for_artifact(path: Path) -> Path:
    return path.parent.parent if path.parent.name == "chunks" else path.parent


def _load_manifest(path: Path) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    manifest = _exact_fields(
        json.loads(path.read_text(encoding="utf-8")),
        _MANIFEST_FIELDS,
        "corpus manifest",
    )
    if _required_int(
        manifest["schema_version"], "manifest.schema_version", minimum=1
    ) != 1:
        raise ValueError("Unsupported corpus manifest schema")
    _required_text(manifest["corpus_version"], "manifest.corpus_version")
    if manifest["default_strategy"] not in _STRATEGIES:
        raise ValueError("Invalid default strategy")
    _required_sha256(manifest["aliases_sha256"], "aliases_sha256")
    _required_sha256(manifest["corpus_sha256"], "corpus_sha256")
    policy_hash = manifest["chunking"].get("policy_sha256") if isinstance(manifest["chunking"], dict) else None
    _exact_fields(manifest["chunking"], _CHUNKING_FIELDS | ({"policy_sha256"} if policy_hash else set()), "chunking manifest")
    if policy_hash:
        _required_sha256(policy_hash, "policy_sha256")
    if {key: value for key, value in manifest["chunking"].items() if key != "policy_sha256"} != _EXPECTED_CHUNKING:
        raise ValueError("Manifest chunking configuration mismatch")
    if not isinstance(manifest["sources"], list) or not manifest["sources"]:
        raise ValueError("manifest.sources must be non-empty")
    sources: dict[str, dict[str, Any]] = {}
    source_order = []
    for source in manifest["sources"]:
        _exact_fields(source, _MANIFEST_SOURCE_FIELDS, "manifest source")
        source_id = _required_text(source["source_id"], "manifest source_id")
        if source_id in sources:
            raise ValueError("Duplicate manifest source_id")
        source_order.append(source_id)
        _required_text(source["publisher"], "manifest publisher")
        _required_text(source["title"], "manifest title")
        _required_text(source["version"], "manifest version")
        _required_https(source["url"], "manifest url")
        _required_text(source["language"], "manifest language")
        _required_text(source["retrieved_on"], "manifest retrieved_on")
        _required_text(source["reuse_basis"], "manifest reuse_basis")
        _required_sha256(source["source_sha256"], "manifest source_sha256")
        if not isinstance(source["title_path_whitelist"], list) or not source["title_path_whitelist"]:
            raise ValueError("manifest title_path_whitelist must be non-empty")
        for title_path in source["title_path_whitelist"]:
            _required_text_list(title_path, "manifest title_path_whitelist")
        sources[source_id] = source
    if source_order != sorted(source_order):
        raise ValueError("Manifest sources must use deterministic ordering")
    if _required_int(manifest["source_count"], "source_count", minimum=1) != len(sources):
        raise ValueError("Manifest source_count mismatch")
    if not isinstance(manifest["strategies"], dict) or set(manifest["strategies"]) != _STRATEGIES:
        raise ValueError("Manifest strategies must be exact")
    for strategy, stats in manifest["strategies"].items():
        _exact_fields(stats, _STRATEGY_FIELDS, f"{strategy} stats")
        for field in (
            "chunk_count", "minimum_estimated_tokens",
            "maximum_estimated_tokens", "target_range_count",
            "short_intact_count",
        ):
            _required_int(stats[field], f"{strategy}.{field}")
        if isinstance(stats["mean_estimated_tokens"], bool) or not isinstance(
            stats["mean_estimated_tokens"], (int, float)
        ):
            raise ValueError("Invalid mean_estimated_tokens")
        _required_sha256(stats["artifact_sha256"], f"{strategy}.artifact_sha256")
    return manifest, sources


def _validate_manifest_provenance(
    root: Path,
    manifest: dict[str, Any],
    manifest_sources: dict[str, dict[str, Any]],
) -> tuple[SourceSnapshot, ...]:
    snapshots = load_source_snapshots(
        root, expected_corpus_version=manifest["corpus_version"]
    )
    snapshot_entries = {
        source.source_id: {
            "source_id": source.source_id,
            "publisher": source.publisher,
            "title": source.title,
            "version": source.version,
            "url": source.url,
            "language": source.language,
            "retrieved_on": source.retrieved_on,
            "reuse_basis": source.reuse_basis,
            "source_sha256": source.source_sha256,
            "title_path_whitelist": [
                list(path) for path in source.title_path_whitelist
            ],
        }
        for source in snapshots
    }
    if manifest_sources != snapshot_entries:
        raise ValueError("Manifest source metadata does not match source snapshots")
    alias_bytes = (root / "aliases" / "aliases.json").read_bytes()
    alias_hash = hashlib.sha256(alias_bytes).hexdigest()
    if alias_hash != manifest["aliases_sha256"]:
        raise ValueError("Manifest aliases hash mismatch")
    artifact_hashes = {}
    for strategy in sorted(_STRATEGIES):
        artifact_bytes = (root / "chunks" / f"{strategy}.json").read_bytes()
        artifact_hash = hashlib.sha256(artifact_bytes).hexdigest()
        if artifact_hash != manifest["strategies"][strategy]["artifact_sha256"]:
            raise ValueError(f"Manifest {strategy} artifact hash mismatch")
        artifact_hashes[strategy] = artifact_hash
    digest_input = "\n".join(
        [
            *(source.source_sha256 for source in snapshots),
            alias_hash,
            artifact_hashes["natural_section"],
            artifact_hashes["hybrid_v1"],
        ]
    ).encode("ascii")
    if manifest["chunking"].get("policy_sha256"):
        policy = hashlib.sha256((root / "chunk-policy.json").read_bytes()).hexdigest()
        if policy != manifest["chunking"]["policy_sha256"]:
            raise ValueError("Manifest chunk policy hash mismatch")
        from healthvault.rag.budget import TokenBudget
        TokenBudget.load(root)
        digest_input += b"\n" + policy.encode("ascii")
    if hashlib.sha256(digest_input).hexdigest() != manifest["corpus_sha256"]:
        raise ValueError("Manifest corpus hash mismatch")
    return snapshots


def _chunk_from_dict(
    raw: dict[str, Any],
    *,
    artifact_version: str,
    artifact_strategy: str,
    sources: dict[str, dict[str, Any]],
) -> KnowledgeChunk:
    _exact_fields(raw, _CHUNK_FIELDS, "knowledge chunk")
    corpus_version = _required_text(raw["corpus_version"], "chunk.corpus_version")
    strategy = _required_text(raw["strategy"], "chunk.strategy")
    if corpus_version != artifact_version or strategy != artifact_strategy:
        raise ValueError("Chunk version or strategy mismatch")
    source_id = _required_text(raw["source_id"], "chunk.source_id")
    source_version = _required_text(raw["source_version"], "chunk.source_version")
    publisher = _required_text(raw["publisher"], "chunk.publisher")
    topic_tags = _required_text_list(raw["topic_tags"], "chunk.topic_tags")
    title_path = _required_text_list(raw["title_path"], "chunk.title_path")
    ordinal = _required_int(raw["ordinal"], "chunk.ordinal")
    body = _required_text(raw["body"], "chunk.body")
    estimated_tokens = _required_int(
        raw["estimated_tokens"], "chunk.estimated_tokens", minimum=1
    )
    if estimated_tokens != estimate_tokens(body):
        raise ValueError("chunk.estimated_tokens does not match the body")
    overlap = _required_int(
        raw["overlap_estimated_tokens"], "chunk.overlap_estimated_tokens"
    )
    if artifact_strategy == "natural_section" and overlap != 0:
        raise ValueError("Natural chunks cannot contain overlap")
    if artifact_strategy == "hybrid_v1":
        if estimated_tokens > MAXIMUM:
            raise ValueError("Hybrid chunks must not exceed 600 estimated tokens")
        if estimated_tokens < MINIMUM and overlap != 0:
            raise ValueError("Short-intact hybrid chunks cannot contain overlap")
        if overlap != 0 and not 40 <= overlap <= 80:
            raise ValueError("Hybrid overlap must be zero or 40-80 estimated tokens")
    body_hash = _required_sha256(raw["body_sha256"], "body_sha256")
    if body_hash != hashlib.sha256(body.encode("utf-8")).hexdigest():
        raise ValueError("body_sha256 does not match chunk.body")
    source_hash = _required_sha256(raw["source_sha256"], "source_sha256")
    url = _required_https(raw["url"], "chunk.url")
    language = _required_text(raw["language"], "chunk.language")
    retrieved_on = _required_text(raw["retrieved_on"], "chunk.retrieved_on")
    chunk_id = _required_text(raw["chunk_id"], "chunk.chunk_id")
    citation_id = _required_text(raw["citation_id"], "chunk.citation_id")
    if chunk_id != citation_id:
        raise ValueError("chunk_id and citation_id must be equal")
    parsed = parse_chunk_id(
        chunk_id, expected_corpus_version=artifact_version
    )
    expected_id = make_chunk_id(
        source_id,
        source_version,
        title_path,
        ordinal,
        corpus_version=artifact_version,
    )
    if parsed.source_id != source_id or parsed.ordinal != ordinal or chunk_id != expected_id:
        raise ValueError("chunk_id digest does not match source identity metadata")
    source = sources.get(source_id)
    if source is None:
        raise ValueError("Chunk source_id is absent from the manifest")
    expected_metadata = {
        "publisher": publisher,
        "version": source_version,
        "url": url,
        "language": language,
        "retrieved_on": retrieved_on,
        "source_sha256": source_hash,
    }
    for field, value in expected_metadata.items():
        if source[field] != value:
            raise ValueError(f"Chunk source metadata mismatch: {field}")
    if title_path[0] != source["title"] or any(" | " in item for item in title_path):
        raise ValueError("Chunk title_path is not a real source hierarchy")
    return KnowledgeChunk(
        chunk_id=chunk_id,
        citation_id=citation_id,
        corpus_version=corpus_version,
        strategy=strategy,
        source_id=source_id,
        source_version=source_version,
        publisher=publisher,
        topic_tags=topic_tags,
        title_path=title_path,
        ordinal=ordinal,
        body=body,
        estimated_tokens=estimated_tokens,
        overlap_estimated_tokens=overlap,
        body_sha256=body_hash,
        source_sha256=source_hash,
        url=url,
        language=language,
        retrieved_on=retrieved_on,
    )


def load_chunk_artifact(
    path: str | Path, *, expected_corpus_version: str | None = CORPUS_VERSION
) -> ChunkArtifact:
    artifact_path = Path(path)
    artifact_bytes = artifact_path.read_bytes()
    raw = _exact_fields(
        json.loads(artifact_bytes.decode("utf-8")),
        _ARTIFACT_FIELDS,
        "chunk artifact",
    )
    if _required_int(
        raw["schema_version"], "artifact.schema_version", minimum=1
    ) != 1:
        raise ValueError("Unsupported chunk artifact schema")
    corpus_version = _required_text(raw["corpus_version"], "artifact.corpus_version")
    if expected_corpus_version is not None and corpus_version != expected_corpus_version:
        raise ValueError("Unsupported chunk artifact corpus version")
    strategy = _required_text(raw["strategy"], "artifact.strategy")
    if strategy not in _STRATEGIES:
        raise ValueError("Unsupported chunk artifact strategy")
    manifest, sources = _load_manifest(_manifest_path_for_artifact(artifact_path))
    if manifest["corpus_version"] != corpus_version:
        raise ValueError("Artifact and manifest corpus versions differ")
    expected_artifact_hash = manifest["strategies"][strategy]["artifact_sha256"]
    if hashlib.sha256(artifact_bytes).hexdigest() != expected_artifact_hash:
        raise ValueError("manifest artifact hash does not match artifact bytes")
    if not isinstance(raw["chunks"], list) or not raw["chunks"]:
        raise ValueError("Chunk artifact must contain non-empty chunks")
    chunks = tuple(
        _chunk_from_dict(
            item,
            artifact_version=corpus_version,
            artifact_strategy=strategy,
            sources=sources,
        )
        for item in raw["chunks"]
    )
    chunk_ids = [chunk.chunk_id for chunk in chunks]
    citation_ids = [chunk.citation_id for chunk in chunks]
    if len(set(chunk_ids)) != len(chunks) or len(set(citation_ids)) != len(chunks):
        raise ValueError("Chunk and citation IDs must be unique")
    if list(chunks) != sorted(chunks, key=lambda chunk: (chunk.source_id, chunk.ordinal)):
        raise ValueError("Chunks must use deterministic source/ordinal ordering")
    ordinals: dict[str, list[int]] = {}
    for chunk in chunks:
        ordinals.setdefault(chunk.source_id, []).append(chunk.ordinal)
    if any(values != list(range(len(values))) for values in ordinals.values()):
        raise ValueError("Chunk ordinals must be contiguous per source")
    expected_stats = _strategy_stats(chunks)
    manifest_stats = manifest["strategies"][strategy]
    if any(manifest_stats[field] != value for field, value in expected_stats.items()):
        raise ValueError("Artifact statistics do not match the manifest")
    snapshots = _validate_manifest_provenance(
        _corpus_root_for_artifact(artifact_path), manifest, sources
    )
    from healthvault.rag.budget import TokenBudget
    budget = TokenBudget.load(_corpus_root_for_artifact(artifact_path)) if strategy == "hybrid_v1" else None
    expected_chunks = build_chunks(snapshots, strategy, corpus_version=corpus_version, token_budget=budget)
    if chunks != expected_chunks:
        raise ValueError("Chunk artifact is not deterministically derived from source snapshots")
    if budget:
        expected_spans = {"policy": "bounded-v2", "tokenizer": budget.identity,
            "chunks": [{"chunk_id": chunk.chunk_id, "source_id": chunk.source_id, "title_path": list(chunk.title_path),
                        **span} for chunk, span in zip(chunks, budget.spans)]}
        if json.loads((_corpus_root_for_artifact(artifact_path) / "chunks" / "boundary-spans.json").read_text(encoding="utf-8")) != expected_spans:
            raise ValueError("Chunk boundary spans do not match the source text.")
    return ChunkArtifact(1, corpus_version, strategy, chunks)


def parse_chunk_id(
    value: str, *, expected_corpus_version: str | None = CORPUS_VERSION
) -> ParsedChunkId:
    if not isinstance(value, str):
        raise ValueError("Invalid HealthVault knowledge chunk ID")
    parts = value.split(":")
    if (
        len(parts) != 5
        or parts[0] != "HVK"
        or (expected_corpus_version is not None and parts[1] != expected_corpus_version)
    ):
        raise ValueError("Invalid HealthVault knowledge chunk ID")
    if not parts[2] or re.fullmatch(r"\d{3}", parts[3]) is None:
        raise ValueError("Invalid HealthVault knowledge chunk ordinal")
    if _HEX_12_RE.fullmatch(parts[4]) is None:
        raise ValueError("Invalid HealthVault knowledge chunk digest")
    return ParsedChunkId(parts[1], parts[2], int(parts[3]), parts[4])
