"""Ports for optional local dense retrieval; no model package is imported here."""

from __future__ import annotations

from typing import Protocol, Sequence


class DenseRetrievalError(RuntimeError):
    """Raised when an optional dense index cannot be used safely."""


class Embedder(Protocol):
    """Embed text locally with a fixed, recorded model and dimension."""

    @property
    def model_name(self) -> str:
        ...

    @property
    def dimension(self) -> int:
        ...

    def embed(self, texts: Sequence[str]) -> tuple[tuple[float, ...], ...]:
        ...


class VectorIndex(Protocol):
    """Return stable chunk IDs for one dense query vector."""

    @property
    def dimension(self) -> int:
        ...

    def search(
        self, vector: Sequence[float], limit: int
    ) -> tuple[tuple[str, float], ...]:
        ...
