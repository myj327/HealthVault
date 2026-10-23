"""Optional dense + lexical retrieval with deterministic RRF and safe fallback."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Mapping, Sequence

from healthvault.domain.knowledge import KnowledgeHit
from healthvault.infrastructure.sqlite_fts_health_knowledge import (
    SQLiteFTSHealthKnowledgeRepository,
)
from healthvault.repositories.dense_retrieval import (
    DenseRetrievalError,
    Embedder,
    VectorIndex,
)


RRF_K = 60
INDEX_VERSION = "hybrid-r1"


@dataclass(frozen=True)
class VectorIndexManifest:
    index_version: str
    corpus_sha256: str
    corpus_version: str
    model_name: str
    dimension: int
    chunk_ids_sha256: str
    model_sha256: str | None = None
    index_sha256: str | None = None

    def to_dict(self) -> dict[str, Any]:
        value = {
            "index_version": self.index_version,
            "corpus_sha256": self.corpus_sha256,
            "corpus_version": self.corpus_version,
            "model_name": self.model_name,
            "dimension": self.dimension,
            "chunk_ids_sha256": self.chunk_ids_sha256,
        }
        value.update({key: item for key, item in (("model_sha256", self.model_sha256), ("index_sha256", self.index_sha256)) if item is not None})
        return value

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "VectorIndexManifest":
        required = {"index_version", "corpus_sha256", "corpus_version", "model_name", "dimension", "chunk_ids_sha256"}
        if not required.issubset(value) or set(value) - required - {"model_sha256", "index_sha256"} or not isinstance(value["dimension"], int):
            raise DenseRetrievalError("dense_index_invalid: invalid index manifest")
        return cls(**dict(value))


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _chunk_ids_sha256(chunk_ids: Sequence[str]) -> str:
    return hashlib.sha256("\n".join(chunk_ids).encode("utf-8")).hexdigest()


@lru_cache(maxsize=2)
def _local_embedder(model_path: str, model_name: str, model_sha256: str):
    from healthvault.infrastructure.bge_faiss import BGEM3Embedder
    return BGEM3Embedder(model_path, model_name=model_name)


def rrf_fuse(
    lexical_chunk_ids: Sequence[str], dense_chunk_ids: Sequence[str], *, limit: int, k: int = RRF_K
) -> tuple[tuple[str, float], ...]:
    """Fuse two ordered lists using stable chunk-ID tie-breaking and de-duplication."""
    if limit <= 0 or k <= 0:
        return ()
    scores: dict[str, float] = {}
    for ranked in (lexical_chunk_ids, dense_chunk_ids):
        for rank, chunk_id in enumerate(ranked, 1):
            if chunk_id:
                scores[chunk_id] = scores.get(chunk_id, 0.0) + 1.0 / (k + rank)
    return tuple(sorted(scores.items(), key=lambda item: (-item[1], item[0]))[:limit])


class HybridHealthKnowledgeRepository:
    """Preserve lexical behavior unless a fully matching local vector index is injected."""

    def __init__(
        self,
        lexical: SQLiteFTSHealthKnowledgeRepository,
        *,
        embedder: Embedder | None = None,
        vector_index: VectorIndex | None = None,
        fallback_reason: str | None = None,
    ) -> None:
        self._lexical = lexical
        self._embedder = embedder
        self._vector_index = vector_index
        self._fallback_reason = fallback_reason
        usable = embedder is not None and vector_index is not None and embedder.dimension == vector_index.dimension
        if not usable and fallback_reason is None:
            self._fallback_reason = "dense_index_unavailable"
        self._usable = usable
        self.last_search_metadata: dict[str, Any] | None = None
        self.index_metadata = dict(lexical.index_metadata) | {
            "engine": "hybrid_rrf" if usable else lexical.index_metadata["engine"],
            "engine_id": "bge_m3_faiss_rrf" if usable else lexical.index_metadata["engine_id"],
            "retrieval_mode": "hybrid" if usable else "lexical",
            "dense_fallback_reason": None if usable else self._fallback_reason,
        }

    @classmethod
    def from_local_artifacts(
        cls,
        lexical: SQLiteFTSHealthKnowledgeRepository,
        *,
        artifact_path: str | Path,
        index_directory: str | Path | None,
        model_path: str | Path | None,
        model_name: str = "BAAI/bge-m3",
    ) -> "HybridHealthKnowledgeRepository":
        if index_directory is None:
            return cls(lexical, fallback_reason="dense_index_missing")
        directory = Path(index_directory); manifest_path = directory / "manifest.json"
        if not manifest_path.is_file() or not (directory / "index.faiss").is_file() or not (directory / "chunk_ids.json").is_file():
            return cls(lexical, fallback_reason="dense_index_missing")
        try:
            manifest = VectorIndexManifest.from_dict(json.loads(manifest_path.read_text(encoding="utf-8")))
            ids = tuple(json.loads((directory / "chunk_ids.json").read_text(encoding="utf-8")))
            if (manifest.index_version != INDEX_VERSION or manifest.corpus_sha256 != _sha256_file(Path(artifact_path)) or manifest.corpus_version != lexical.index_metadata["corpus_version"] or manifest.model_name != model_name or manifest.chunk_ids_sha256 != _chunk_ids_sha256(ids)):
                return cls(lexical, fallback_reason="dense_index_manifest_mismatch")
            if model_path is None:
                return cls(lexical, fallback_reason="dense_model_missing")
            if manifest.index_sha256 and manifest.index_sha256 != _sha256_file(directory / "index.faiss"):
                return cls(lexical, fallback_reason="dense_index_checksum_mismatch")
            if manifest.model_sha256:
                from healthvault.rag.vector_cache import model_fingerprint
                if manifest.model_sha256 != model_fingerprint(Path(model_path)):
                    return cls(lexical, fallback_reason="dense_model_checksum_mismatch")
            from healthvault.infrastructure.bge_faiss import BGEM3Embedder, FaissVectorIndex
            embedder = (_local_embedder(str(Path(model_path).resolve()), model_name, manifest.model_sha256)
                        if manifest.model_sha256 else BGEM3Embedder(model_path, model_name=model_name))
            if embedder.dimension != manifest.dimension:
                return cls(lexical, fallback_reason="dense_index_dimension_mismatch")
            index = FaissVectorIndex.load(directory / "index.faiss", directory / "chunk_ids.json", manifest.dimension)
            return cls(lexical, embedder=embedder, vector_index=index)
        except (OSError, UnicodeError, ValueError, json.JSONDecodeError, DenseRetrievalError) as exc:
            return cls(lexical, fallback_reason=str(exc).split(":", 1)[0])

    def close(self) -> None:
        self._lexical.close()

    def search(self, query: str, limit: int) -> tuple[KnowledgeHit, ...]:
        lexical_hits = self._lexical.search(query, limit)
        self.last_search_metadata = {"retrieval_mode": "lexical", "dense_participated": False,
                                     "dense_fallback_reason": self._fallback_reason if not self._usable else None,
                                     "lexical_chunk_ids": [hit.document.chunk_id for hit in lexical_hits],
                                     "dense_chunk_ids": [], "fused_chunk_ids": []}
        if not self._usable:
            return lexical_hits
        assert self._embedder is not None and self._vector_index is not None
        try:
            vector = self._embedder.embed((query,))[0]
            dense = self._vector_index.search(vector, limit)
        except (DenseRetrievalError, IndexError, ValueError):
            self._fallback_reason = "dense_query_failed"
            self.index_metadata["dense_fallback_reason"] = self._fallback_reason
            self.last_search_metadata["dense_fallback_reason"] = self._fallback_reason
            return lexical_hits
        fused = rrf_fuse(
            [hit.document.chunk_id for hit in lexical_hits],
            [chunk_id for chunk_id, _score in dense],
            limit=limit,
        )
        hits: list[KnowledgeHit] = []
        for rank, (chunk_id, score) in enumerate(fused, 1):
            hit = self._lexical.hit_for_chunk_id(
                chunk_id, rank=rank, score=round(score, 12), matched_terms=()
            )
            if hit is not None:
                hits.append(hit)
        self.index_metadata["dense_fallback_reason"] = None
        self.last_search_metadata.update(retrieval_mode="hybrid", dense_participated=bool(dense),
                                         dense_chunk_ids=[chunk_id for chunk_id, _ in dense],
                                         fused_chunk_ids=[hit.document.chunk_id for hit in hits])
        return tuple(hits)
