import json
import tempfile
import unittest
from pathlib import Path

from healthvault.application.search_health_knowledge import search_health_knowledge
from healthvault.domain.knowledge import KnowledgeDocument, KnowledgeHit, KnowledgeSource
from healthvault.infrastructure.hybrid_health_knowledge import (
    HybridHealthKnowledgeRepository,
    rrf_fuse,
)
from healthvault.infrastructure.bge_faiss import BGEM3Embedder
from healthvault.infrastructure.sqlite_fts_health_knowledge import (
    DEFAULT_CHUNK_ARTIFACT_PATH,
    SQLiteFTSHealthKnowledgeRepository,
)
from healthvault.repositories.dense_retrieval import DenseRetrievalError


def _hit(chunk_id, rank=1, score=1.0):
    return KnowledgeHit(
        document=KnowledgeDocument(
            citation_id=f"HVK:test:{chunk_id}", topic="test", title=chunk_id,
            summary=f"evidence for {chunk_id}", search_terms=(),
            source=KnowledgeSource("Test", "Test", "https://example.test", "2026-09-08"),
            chunk_id=chunk_id,
        ),
        matched_terms=("test",), rank=rank, score=score,
    )


class FakeLexical:
    index_metadata = {
        "engine": "fts5", "engine_id": "sqlite_fts", "corpus_version": "test-v1",
    }

    def __init__(self):
        self.hits = {"a": _hit("a"), "b": _hit("b"), "c": _hit("c")}

    def search(self, _query, limit):
        return tuple(self.hits[key] for key in ("a", "b")[:limit])

    def hit_for_chunk_id(self, chunk_id, *, rank, score, matched_terms):
        hit = self.hits.get(chunk_id)
        if hit is None:
            return None
        return KnowledgeHit(hit.document, matched_terms, rank, score)

    def close(self):
        pass


class FakeEmbedder:
    model_name = "fake-bge"
    dimension = 2

    def __init__(self, error=False):
        self.error = error

    def embed(self, _texts):
        if self.error:
            raise DenseRetrievalError("fake failure")
        return ((1.0, 0.0),)


class FakeIndex:
    dimension = 2

    def search(self, _vector, _limit):
        return (("b", 0.9), ("c", 0.8))


class HybridHealthKnowledgeTests(unittest.TestCase):
    def test_rrf_deduplicates_and_tie_breaks_by_chunk_id(self):
        self.assertEqual(
            (("b", 1 / 61 + 1 / 62), ("a", 1 / 61), ("c", 1 / 62)),
            rrf_fuse(("a", "b"), ("b", "c"), limit=3),
        )

    def test_hybrid_fuses_fake_dense_hits_without_changing_citations(self):
        repository = HybridHealthKnowledgeRepository(
            FakeLexical(), embedder=FakeEmbedder(), vector_index=FakeIndex()
        )
        hits = repository.search("test", 3)
        self.assertEqual(["b", "a", "c"], [hit.document.chunk_id for hit in hits])
        self.assertEqual(["HVK:test:b", "HVK:test:a", "HVK:test:c"], [hit.document.citation_id for hit in hits])
        self.assertEqual("hybrid", repository.index_metadata["retrieval_mode"])

    def test_dense_query_error_safely_falls_back_to_lexical(self):
        repository = HybridHealthKnowledgeRepository(
            FakeLexical(), embedder=FakeEmbedder(error=True), vector_index=FakeIndex()
        )
        self.assertEqual(["a", "b"], [hit.document.chunk_id for hit in repository.search("test", 2)])
        self.assertEqual("dense_query_failed", repository.index_metadata["dense_fallback_reason"])

    def test_manifest_mismatch_never_loads_a_vector_index(self):
        lexical = SQLiteFTSHealthKnowledgeRepository()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "manifest.json").write_text(json.dumps({"index_version": "wrong", "corpus_sha256": "x", "corpus_version": "wrong", "model_name": "BAAI/bge-m3", "dimension": 2, "chunk_ids_sha256": "x"}), encoding="utf-8")
            (root / "index.faiss").write_bytes(b"not-read")
            (root / "chunk_ids.json").write_text("[]", encoding="utf-8")
            repository = HybridHealthKnowledgeRepository.from_local_artifacts(
                lexical, artifact_path=DEFAULT_CHUNK_ARTIFACT_PATH,
                index_directory=root, model_path=None,
            )
        self.assertEqual("dense_index_manifest_mismatch", repository.index_metadata["dense_fallback_reason"])
        self.assertEqual("lexical", repository.index_metadata["retrieval_mode"])

    def test_request_gate_prevents_hybrid_repository_search(self):
        class RecordingRepository:
            def __init__(self): self.calls = 0
            def search(self, _query, _limit): self.calls += 1; return ()
        repository = RecordingRepository()
        result = search_health_knowledge({"query": "show my latest A1C result"}, repository).to_dict()
        self.assertEqual("empty", result["status"])
        self.assertEqual("unsupported_personal_record", result["issues"][0]["code"])
        self.assertEqual(0, repository.calls)

    def test_bge_adapter_requires_an_existing_local_model_before_importing_dependencies(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(DenseRetrievalError, "dense_model_missing"):
                BGEM3Embedder(Path(directory) / "not-downloaded")


if __name__ == "__main__":
    unittest.main()
