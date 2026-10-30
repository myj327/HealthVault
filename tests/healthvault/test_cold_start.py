import hashlib
import json
import os
import tempfile
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Event
from unittest.mock import patch

from healthvault.infrastructure import hybrid_health_knowledge as hybrid
from healthvault.rag.vector_cache import model_fingerprint
from tests.healthvault.test_hybrid_health_knowledge import FakeEmbedder, FakeIndex, FakeLexical


class ColdStartTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.model = self.root / 'model'
        self.model.mkdir()
        (self.model / 'config.json').write_text('{"version":1}', encoding='utf-8')
        self.artifact = self.root / 'corpus.json'
        self.artifact.write_text('{}', encoding='utf-8')
        (self.root / 'index.faiss').write_bytes(b'synthetic-index')
        (self.root / 'chunk_ids.json').write_text('["a","b","c"]', encoding='utf-8')
        self.manifest = hybrid.VectorIndexManifest(
            hybrid.INDEX_VERSION, hashlib.sha256(self.artifact.read_bytes()).hexdigest(),
            'test-v1', 'BAAI/bge-m3', 2, hybrid._chunk_ids_sha256(('a', 'b', 'c')),
            model_fingerprint(self.model), hashlib.sha256(b'synthetic-index').hexdigest(),
        )
        (self.root / 'manifest.json').write_text(json.dumps(self.manifest.to_dict()), encoding='utf-8')
        hybrid._local_embedder.cache_clear()
        self.addCleanup(hybrid._local_embedder.cache_clear)
        self.loader = patch('healthvault.infrastructure.bge_faiss.FaissVectorIndex.load', return_value=FakeIndex())
        self.loader.start()
        self.addCleanup(self.loader.stop)

    def repository(self):
        return hybrid.HybridHealthKnowledgeRepository.from_local_artifacts(
            FakeLexical(), artifact_path=self.artifact, index_directory=self.root, model_path=self.model)

    def test_creation_skips_model_hash_load_and_probe(self):
        with patch('healthvault.rag.vector_cache.model_fingerprint', wraps=model_fingerprint) as fingerprint, \
                patch('healthvault.infrastructure.bge_faiss.BGEM3Embedder', return_value=FakeEmbedder()) as load:
            repository = self.repository()
            fingerprint.assert_not_called()
            load.assert_not_called()
            hits = repository.search('test', 3)
            self.assertEqual(['b', 'a', 'c'], [hit.document.chunk_id for hit in hits])
            fingerprint.assert_called_once()
            load.assert_called_once()
            self.assertTrue(repository.last_search_metadata['dense_participated'])

    def test_concurrent_first_searches_share_one_model(self):
        with patch('healthvault.infrastructure.bge_faiss.BGEM3Embedder', return_value=FakeEmbedder()) as load:
            repositories = [self.repository() for _ in range(4)]
            with ThreadPoolExecutor(max_workers=4) as pool:
                results = list(pool.map(lambda repository: repository.search('test', 3), repositories))
            load.assert_called_once()
            self.assertTrue(all(len(result) == 3 for result in results))

    def test_warm_up_reuses_weights_without_creating_search_evidence(self):
        with patch('healthvault.infrastructure.bge_faiss.BGEM3Embedder', return_value=FakeEmbedder()) as load:
            repository = self.repository()
            repository.warm_up()
            self.assertIsNone(repository.last_search_metadata)
            self.repository().search('test', 3)
            load.assert_called_once()

    def test_changed_model_is_rejected_before_using_cached_weights(self):
        with patch('healthvault.infrastructure.bge_faiss.BGEM3Embedder', return_value=FakeEmbedder()) as load:
            self.repository().search('test', 3)
            (self.model / 'config.json').write_text('{"version":2}', encoding='utf-8')
            repository = self.repository()
            self.assertEqual(['a', 'b'], [hit.document.chunk_id for hit in repository.search('test', 3)])
            self.assertEqual('dense_model_checksum_mismatch', repository.last_search_metadata['dense_fallback_reason'])
            self.assertFalse(repository.last_search_metadata['dense_participated'])
            load.assert_called_once()

    def test_wrong_model_dimension_cannot_participate(self):
        embedder = FakeEmbedder()
        embedder.dimension = 3
        with patch('healthvault.infrastructure.bge_faiss.BGEM3Embedder', return_value=embedder):
            repository = self.repository()
            repository.search('test', 3)
            self.assertEqual('dense_index_dimension_mismatch', repository.last_search_metadata['dense_fallback_reason'])
            self.assertFalse(repository.last_search_metadata['dense_participated'])

    def test_invalid_index_still_rejected_without_model_loading(self):
        (self.root / 'index.faiss').write_bytes(b'corrupt-index')
        with patch('healthvault.infrastructure.bge_faiss.BGEM3Embedder') as load:
            repository = self.repository()
            repository.search('test', 3)
            load.assert_not_called()
            self.assertEqual('dense_index_checksum_mismatch', repository.last_search_metadata['dense_fallback_reason'])

    def test_initialization_error_returns_honest_lexical_fallback(self):
        with patch('healthvault.infrastructure.bge_faiss.BGEM3Embedder', side_effect=RuntimeError('synthetic unavailable runtime')):
            repository = self.repository()
            self.assertEqual(['a', 'b'], [hit.document.chunk_id for hit in repository.search('test', 3)])
            self.assertEqual('dense_query_failed', repository.last_search_metadata['dense_fallback_reason'])
            self.assertFalse(repository.last_search_metadata['dense_participated'])

    def test_web_login_and_records_work_while_model_is_warming(self):
        # Use a held warm-up to prove startup never joins the loading thread.
        with patch.dict(os.environ, {'HEALTHVAULT_UI_DB_PATH': str(self.root / 'web.sqlite3'),
                                     'HEALTHVAULT_KNOWLEDGE_MODE': 'lexical'}):
            from fastapi.testclient import TestClient
            from healthvault.ui.controller import HealthVaultUIController
            from healthvault.web.api import create_app
            controller = HealthVaultUIController(self.root / 'web.sqlite3', knowledge_release_root=self.root / 'releases')
            controller.knowledge_repository = self.repository()
            entered, released = Event(), Event()

            def warm_up():
                entered.set()
                released.wait(10)

            controller.knowledge_repository.warm_up = warm_up
            app = create_app(controller, skill_root=self.root / 'skills', skill_evolution_root=self.root / 'evolution')
            try:
                with TestClient(app) as client:
                    self.assertTrue(entered.wait(2))
                    response = client.post('/api/auth/register', json={
                        'username': 'cold-start-user', 'password': 'synthetic-password-123'})
                    self.assertEqual(200, response.status_code)
                    profile = response.json()['profile_id']
                    self.assertEqual(200, client.get(f'/api/profiles/{profile}/timeline').status_code)
                    self.assertFalse(released.is_set())
            finally:
                released.set()


if __name__ == '__main__':
    unittest.main()
