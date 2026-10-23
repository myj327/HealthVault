"""Compatibility name for the default offline SQLite FTS knowledge repository."""

from __future__ import annotations

from pathlib import Path

from healthvault.infrastructure.sqlite_fts_health_knowledge import (
    DEFAULT_ALIASES_PATH,
    DEFAULT_CHUNK_ARTIFACT_PATH,
    SQLiteFTSHealthKnowledgeRepository,
)


DEFAULT_HEALTH_KNOWLEDGE_PATH = DEFAULT_CHUNK_ARTIFACT_PATH


class LocalHealthKnowledgeRepository(SQLiteFTSHealthKnowledgeRepository):
    """Preserve the old import while using the versioned FTS5 implementation."""

    def __init__(self, path: str | Path = DEFAULT_HEALTH_KNOWLEDGE_PATH) -> None:
        artifact_path = Path(path)
        aliases_path = (
            artifact_path.parents[1] / "aliases" / "aliases.json"
            if artifact_path.parent.name == "chunks"
            else DEFAULT_ALIASES_PATH
        )
        super().__init__(artifact_path, aliases_path)
