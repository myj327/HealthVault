"""Repository port for curated offline health knowledge."""

from typing import Protocol

from healthvault.domain.knowledge import KnowledgeHit


class HealthKnowledgeQueryError(RuntimeError):
    """Raised when the curated knowledge corpus cannot be queried safely."""


class HealthKnowledgeRepository(Protocol):
    def search(self, query: str, limit: int) -> tuple[KnowledgeHit, ...]:
        """Return deterministic ranked source chunks for one query."""
        ...
