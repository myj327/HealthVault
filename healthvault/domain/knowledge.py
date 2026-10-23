"""Domain models for offline health knowledge retrieval."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


EDUCATIONAL_NOTICE = (
    "Educational source retrieval only; not medical advice."
)


@dataclass(frozen=True)
class KnowledgeSource:
    publisher: str
    title: str
    url: str
    retrieved_on: str
    source_id: str = ""
    version: str = ""
    section: str = ""
    language: str = "en"
    reuse_basis: str = ""
    sha256: str = ""
    provenance: dict[str, Any] | None = None


@dataclass(frozen=True)
class KnowledgeDocument:
    citation_id: str
    topic: str
    title: str
    summary: str
    search_terms: tuple[str, ...]
    source: KnowledgeSource
    chunk_id: str = ""
    corpus_version: str = ""
    source_version: str = ""
    topic_tags: tuple[str, ...] = ()
    title_path: tuple[str, ...] = ()
    estimated_tokens: int = 0
    body_sha256: str = ""


@dataclass(frozen=True)
class KnowledgeHit:
    document: KnowledgeDocument
    matched_terms: tuple[str, ...]
    rank: int = 0
    score: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "citation_id": self.document.citation_id,
            "chunk_id": self.document.chunk_id or self.document.citation_id,
            "corpus_version": self.document.corpus_version,
            "topic": self.document.topic,
            "topic_tags": list(self.document.topic_tags),
            "title": self.document.title,
            "title_path": list(self.document.title_path),
            "excerpt": self.document.summary,
            "estimated_tokens": self.document.estimated_tokens,
            "body_sha256": self.document.body_sha256,
            "matched_terms": list(self.matched_terms),
            "rank": self.rank,
            "score": self.score,
            "source": {key:value for key,value in asdict(self.document.source).items() if key!='provenance' or value is not None},
        }


@dataclass(frozen=True)
class KnowledgeSearchIssue:
    code: str
    message: str

    def to_dict(self) -> dict[str, str]:
        return asdict(self)


@dataclass(frozen=True)
class KnowledgeSearchResult:
    status: str
    limit: int | None
    hits: tuple[KnowledgeHit, ...]
    issues: tuple[KnowledgeSearchIssue, ...]
    query: str | None = None

    def to_dict(self) -> dict[str, Any]:
        available_citation_ids = [
            hit.document.citation_id
            for hit in self.hits
            if hit.document.citation_id.startswith("HVK:")
        ]
        result = {"status": self.status}
        if self.query is not None:
            result["query"] = self.query
        result.update(
            {
                "limit": self.limit,
                "result_count": len(self.hits),
                "results": [hit.to_dict() for hit in self.hits],
                "available_citation_ids": available_citation_ids,
                "recommended_citation_ids": available_citation_ids[:1],
                "issues": [issue.to_dict() for issue in self.issues],
                "notice": EDUCATIONAL_NOTICE,
            }
        )
        return result
