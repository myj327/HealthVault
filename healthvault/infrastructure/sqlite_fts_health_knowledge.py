"""In-memory SQLite FTS5/BM25 repository for versioned knowledge chunks."""

from __future__ import annotations

import json
import re
import sqlite3
import unicodedata
from pathlib import Path
from typing import Any

from healthvault.domain.knowledge import KnowledgeDocument, KnowledgeHit, KnowledgeSource
from healthvault.rag.corpus import DEFAULT_CORPUS_ROOT, load_chunk_artifact
from healthvault.repositories.health_knowledge import HealthKnowledgeQueryError


DEFAULT_CHUNK_ARTIFACT_PATH = DEFAULT_CORPUS_ROOT / "chunks" / "hybrid_v1.json"
DEFAULT_ALIASES_PATH = DEFAULT_CORPUS_ROOT / "aliases" / "aliases.json"

_UNSAFE_QUERY_RE = re.compile(
    r"[\"'()*{}:^]|--|\b(?:union|select|insert|update|delete|drop|pragma)\b",
    re.IGNORECASE,
)
_TOKEN_RE = re.compile(r"[a-z0-9]+(?:[-_][a-z0-9]+)*|[\u3400-\u9fff]{3,}")
_STOPWORDS = {
    "about", "and", "check", "does", "for", "from", "have", "how",
    "result", "results", "test", "testing", "the", "what", "with",
}


def _normalize(value: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", value).casefold().split())


def _quoted_match(terms: tuple[str, ...]) -> str:
    return " OR ".join(f'"{term}"' for term in terms)


class SQLiteFTSHealthKnowledgeRepository:
    """Build a transient trigram index and return deterministic BM25 hits."""

    def __init__(
        self,
        artifact_path: str | Path = DEFAULT_CHUNK_ARTIFACT_PATH,
        aliases_path: str | Path = DEFAULT_ALIASES_PATH,
    ) -> None:
        self._artifact_path = Path(artifact_path)
        self._aliases_path = Path(aliases_path)
        provenance = self._artifact_path.parent.parent / 'provenance.json'
        self._source_metadata = {item['source_id']:item for item in json.loads(provenance.read_text(encoding='utf-8'))} if provenance.is_file() else {}
        try:
            # The manifest and snapshots authenticate the artifact.  The default
            # remains R1, while evaluators may explicitly provide another frozen
            # corpus version.
            artifact = load_chunk_artifact(
                self._artifact_path, expected_corpus_version=None
            )
            aliases = json.loads(self._aliases_path.read_text(encoding="utf-8"))
            if aliases.get("corpus_version") != artifact.corpus_version:
                raise ValueError("Alias and chunk corpus versions differ")
            self._alias_groups = tuple(
                (
                    _normalize(group["topic"]),
                    tuple(_normalize(alias) for alias in group["aliases"]),
                )
                for group in aliases["groups"]
            )
            self._chunks = {chunk.chunk_id: chunk for chunk in artifact.chunks}
            # Streamlit caches this repository across script threads.  The index is
            # fully built before exposure and is queried read-only afterwards, so
            # SQLite may safely serialize those shared-connection reads; rebuilding
            # or writing the index from concurrent threads remains unsupported.
            self._connection = sqlite3.connect(":memory:", check_same_thread=False)
            self._connection.execute(
                "CREATE VIRTUAL TABLE chunk_fts USING fts5("
                "chunk_id UNINDEXED, index_text, tokenize=trigram)"
            )
            alias_by_topic = dict(self._alias_groups)
            rows = []
            self._normalized_index_text = {}
            for chunk in artifact.chunks:
                aliases_for_chunk = tuple(
                    alias
                    for topic in chunk.topic_tags
                    for alias in alias_by_topic.get(_normalize(topic), ())
                )
                index_text = "\n".join(
                    (
                        " / ".join(chunk.title_path),
                        " ".join(chunk.topic_tags),
                        " ".join(aliases_for_chunk),
                        chunk.body,
                    )
                )
                self._normalized_index_text[chunk.chunk_id] = _normalize(index_text)
                rows.append((chunk.chunk_id, index_text))
            self._connection.executemany(
                "INSERT INTO chunk_fts(chunk_id, index_text) VALUES (?, ?)", rows
            )
            self._connection.commit()
        except (OSError, UnicodeError, json.JSONDecodeError, KeyError, TypeError, ValueError, sqlite3.Error) as exc:
            connection = getattr(self, "_connection", None)
            if connection is not None:
                connection.close()
            raise HealthKnowledgeQueryError(
                "Health knowledge FTS5 corpus is unavailable or invalid."
            ) from exc
        self.index_metadata = {
            "engine": "fts5",
            "ranking": "bm25",
            "tokenizer": "trigram",
            "engine_id": "sqlite_fts5_bm25_trigram",
            "storage": ":memory:",
            "corpus_version": artifact.corpus_version,
            "strategy": artifact.strategy,
            "chunk_count": len(artifact.chunks),
        }

    def close(self) -> None:
        connection = getattr(self, "_connection", None)
        if connection is not None:
            connection.close()
            self._connection = None

    def hit_for_chunk_id(
        self, chunk_id: str, *, rank: int, score: float, matched_terms: tuple[str, ...]
    ) -> KnowledgeHit | None:
        """Expose an existing corpus chunk without changing its citation contract."""
        chunk = self._chunks.get(chunk_id)
        if chunk is None:
            return None
        source = KnowledgeSource(
            publisher=chunk.publisher,
            provenance=self._source_metadata.get(chunk.source_id),
            title=chunk.title_path[0],
            url=chunk.url,
            retrieved_on=chunk.retrieved_on,
            source_id=chunk.source_id,
            version=chunk.source_version,
            section=" / ".join(chunk.title_path),
            language=chunk.language,
            sha256=chunk.source_sha256,
        )
        return KnowledgeHit(
            document=KnowledgeDocument(
                citation_id=chunk.citation_id,
                topic=chunk.topic_tags[0],
                title=chunk.title_path[-1],
                summary=chunk.body,
                search_terms=(),
                source=source,
                chunk_id=chunk.chunk_id,
                corpus_version=chunk.corpus_version,
                source_version=chunk.source_version,
                topic_tags=chunk.topic_tags,
                title_path=chunk.title_path,
                estimated_tokens=chunk.estimated_tokens,
                body_sha256=chunk.body_sha256,
            ),
            matched_terms=matched_terms,
            rank=rank,
            score=score,
        )

    def _expand_query(
        self, query: str
    ) -> tuple[tuple[str, ...], tuple[str, ...], tuple[str, ...], tuple[str, ...]]:
        normalized = _normalize(query)
        if not normalized or _UNSAFE_QUERY_RE.search(normalized):
            return (), (), (), ()
        group_matches = []
        for topic, aliases in self._alias_groups:
            found = tuple(alias for alias in aliases if alias in normalized)
            if found:
                group_matches.append((topic, aliases, found))
        exact_matches = [
            item for item in group_matches if normalized in item[2]
        ]
        if exact_matches:
            group_matches = exact_matches
        matched_topics = [item[0] for item in group_matches]
        matched_aliases = [alias for item in group_matches for alias in item[2]]
        expanded = [
            term
            for topic, aliases, _found in group_matches
            for term in (*aliases, topic)
        ]
        literal_terms = tuple(
            token for token in _TOKEN_RE.findall(normalized)
            if len(token) >= 3 and token not in _STOPWORDS
        )
        required_terms = () if group_matches else literal_terms
        for token in literal_terms:
            if len(token) >= 3 and token not in _STOPWORDS:
                expanded.append(token)
        terms = tuple(sorted(dict.fromkeys(term for term in expanded if len(term) >= 3)))
        return (
            terms,
            tuple(dict.fromkeys(matched_topics)),
            tuple(sorted(dict.fromkeys(matched_aliases))),
            tuple(dict.fromkeys(required_terms)),
        )

    def search(self, query: str, limit: int) -> tuple[KnowledgeHit, ...]:
        terms, matched_topics, matched_aliases, required_terms = self._expand_query(query)
        if not terms:
            return ()
        try:
            rows = self._connection.execute(
                "SELECT chunk_id, bm25(chunk_fts) "
                "FROM chunk_fts WHERE chunk_fts MATCH ? LIMIT 200",
                (_quoted_match(terms),),
            ).fetchall()
        except sqlite3.Error as exc:
            raise HealthKnowledgeQueryError(
                "Health knowledge query could not be executed."
            ) from exc
        scored: list[tuple[float, str, Any]] = []
        for chunk_id, raw_bm25 in rows:
            if any(
                term not in self._normalized_index_text[chunk_id]
                for term in required_terms
            ):
                continue
            chunk = self._chunks[chunk_id]
            normalized_tags = {_normalize(tag) for tag in chunk.topic_tags}
            topic_matches = sum(topic in normalized_tags for topic in matched_topics)
            score = round(topic_matches * 10.0 + max(0.0, -float(raw_bm25)), 8)
            scored.append((score, chunk_id, chunk))
        scored.sort(key=lambda item: (-item[0], item[1]))
        hits = []
        for rank, (score, _chunk_id, chunk) in enumerate(scored[:limit], 1):
            source = KnowledgeSource(
                publisher=chunk.publisher,
                provenance=self._source_metadata.get(chunk.source_id),
                title=chunk.title_path[0],
                url=chunk.url,
                retrieved_on=chunk.retrieved_on,
                source_id=chunk.source_id,
                version=chunk.source_version,
                section=" / ".join(chunk.title_path),
                language=chunk.language,
                sha256=chunk.source_sha256,
            )
            normalized_tags = tuple(_normalize(tag) for tag in chunk.topic_tags)
            primary_topic = next(
                (
                    topic
                    for topic in matched_topics
                    if topic in normalized_tags
                ),
                chunk.topic_tags[0],
            )
            document = KnowledgeDocument(
                citation_id=chunk.citation_id,
                topic=primary_topic,
                title=chunk.title_path[-1],
                summary=chunk.body,
                search_terms=(),
                source=source,
                chunk_id=chunk.chunk_id,
                corpus_version=chunk.corpus_version,
                source_version=chunk.source_version,
                topic_tags=chunk.topic_tags,
                title_path=chunk.title_path,
                estimated_tokens=chunk.estimated_tokens,
                body_sha256=chunk.body_sha256,
            )
            hits.append(
                KnowledgeHit(
                    document=document,
                    matched_terms=matched_aliases or terms,
                    rank=rank,
                    score=score,
                )
            )
        return tuple(hits)
