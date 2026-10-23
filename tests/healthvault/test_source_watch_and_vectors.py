import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import httpx
from fastapi.testclient import TestClient

from healthvault.rag.sources import OfficialSourceWatcher, fetch_official
from healthvault.rag.releases import KnowledgeReleaseStore, _bundle_hash
from healthvault.rag.build_hybrid_index import build_index
from healthvault.rag.vector_cache import model_fingerprint
from healthvault.ui.controller import HealthVaultUIController
from healthvault.web.api import create_app
from tests.healthvault.test_knowledge_releases import update_payload


class FakeEmbedder:
    def __init__(self): self.calls = []
    def embed(self, texts):
        self.calls.extend(texts)
        return tuple((1.0, int(hashlib.sha256(text.encode()).hexdigest()[:4],16)/65536) for text in texts)


class SourceWatchVectorTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(); self.root = Path(self.temp.name)
        self.store = KnowledgeReleaseStore(self.root / 'releases')

    def tearDown(self): self.temp.cleanup()

    def test_changed_sources_are_pending_and_conditional_checks_survive_restart(self):
        calls = []; changed = False
        def fetch(url, headers):
            calls.append((url, headers))
            if headers and not changed: return {'status':304,'url':url}
            marker = 'new changed page text' if changed and 'a1c' in url else 'original official page text'
            return {'status':200,'url':url,'body':(f'<html><article><h1>Health document</h1><p>{marker} '+('Details about this document. '*25)+'</p></article></html>').encode(),'etag':marker}
        # Keep fixtures bounded to a single known official source.
        import healthvault.rag.sources as module
        source = next(value for value in module._snapshots(self.store.active_root).values() if value['source_id']=='niddk-a1c-test')
        with patch.object(module, '_snapshots', return_value={source['source_id']:source}):
            watcher = OfficialSourceWatcher(self.store,fetch)
            self.assertEqual(1,watcher.check()['changed'])
            self.assertEqual(0,OfficialSourceWatcher(self.store,fetch).check()['changed'])
            self.assertTrue(calls[-1][1]['If-None-Match'])
            changed = True
            self.assertEqual(1,watcher.check()['changed'])
            record = watcher.list()[0]
            pending = watcher.pending(record['observation_id'])
            self.assertIn('new changed page text',pending['text'])
            self.assertEqual('needs_review',record['status'])
            self.assertFalse((self.store.root/'active.json').exists())

    def test_redirect_boundary_blocks_unapproved_host(self):
        requests = []
        def handle(request):
            requests.append(str(request.url)); return httpx.Response(302, headers={'location':'https://evil.invalid/report'})
        client = httpx.Client(transport=httpx.MockTransport(handle))
        with patch('httpx.Client',return_value=client), patch('healthvault.rag.sources.socket.getaddrinfo',return_value=[(None,None,None,None,('8.8.8.8',443))]), self.assertRaises(ValueError):
            fetch_official('https://www.niddk.nih.gov/test')
        self.assertEqual(1,len(requests))

    def test_vectors_reuse_only_unchanged_text_and_matching_model_and_bundle(self):
        first = self.store.stage(update_payload(), 'admin')
        corpus = self.store._directory(first['candidate_id'])/'corpus'
        model = self.root/'model'; model.mkdir(); (model/'config.json').write_text('{"version":1}')
        embedder = FakeEmbedder(); cache = self.root/'cache'
        manifest = build_index(artifact_path=corpus/'chunks/hybrid_v1.json',output_directory=corpus/'vectors',model_path=model,cache_directory=cache,embedder=embedder)
        count = len(embedder.calls); self.assertGreater(count,0)
        build_index(artifact_path=corpus/'chunks/hybrid_v1.json',output_directory=corpus/'vectors',model_path=model,cache_directory=cache,embedder=embedder)
        self.assertEqual(count,len(embedder.calls))
        payload = update_payload('test-vector-v2')
        payload['source_updates'][0]['sections'][0]['blocks'][0]['text'] += ' Another isolated update.'
        payload['source_updates'][0]['source_sha256'] = hashlib.sha256(json.dumps(payload['source_updates'][0]['sections'],ensure_ascii=False,sort_keys=True,separators=(',',':')).encode()).hexdigest()
        second = self.store.stage(payload,'admin'); other = self.store._directory(second['candidate_id'])/'corpus'
        build_index(artifact_path=other/'chunks/hybrid_v1.json',output_directory=other/'vectors',model_path=model,cache_directory=cache,embedder=embedder)
        stats = json.loads((other/'vectors/build-stats.json').read_text())
        self.assertGreater(stats['reused'],0); self.assertGreater(stats['embedded'],0)
        self.assertLess(stats['embedded'],stats['total'])
        _bundle_hash(other)
        (other/'vectors/index.faiss').write_bytes(b'corrupted')
        with self.assertRaises(ValueError): _bundle_hash(other)
        (model/'config.json').write_text('{"version":2}')
        self.assertNotEqual(manifest.model_sha256,model_fingerprint(model))

    def test_source_check_admin_access(self):
        app = create_app(HealthVaultUIController(self.root/'records.sqlite3',knowledge_release_root=self.store.root), official_source_fetcher=lambda *args: (_ for _ in ()).throw(ValueError('unavailable')))
        with TestClient(app) as client:
            account = client.post('/api/auth/register',json={'username':'source-user','password':'synthetic-password-123'}).json()
            client.headers.update({'X-CSRF-Token':account['csrf_token']})
            self.assertEqual(403,client.post('/api/admin/knowledge/check-sources').status_code)
            app.state.auth_store.set_admin('source-user',enabled=True)
            response = client.post('/api/admin/knowledge/check-sources')
            self.assertEqual(200,response.status_code)
            self.assertTrue(response.json()['errors'])
            self.assertFalse((self.store.root/'active.json').exists())

    def test_update_template_computes_changed_content_hash_and_keeps_other_sources(self):
        payload = self.store.update_template('niddk-a1c-test')
        source = payload['source_updates'][0]
        source['sections'][0]['blocks'][0]['text'] += ' An isolated template marker frostcopper.'
        payload['probes'] = [{'query':'frostcopper','expected_source_id':'niddk-a1c-test'}]
        record = self.store.stage(payload,'admin')
        self.assertEqual('pending_review',record['status'])
        self.assertEqual(['niddk-a1c-test'],record['changed_source_ids'])
        saved = json.loads((self.store._directory(record['candidate_id'])/'corpus/sources/niddk-a1c-test.json').read_text(encoding='utf-8'))
        self.assertEqual(64,len(saved['source_sha256']))
        self.assertEqual('auto',source['source_sha256'])

    def test_official_snapshot_text_keeps_main_table_and_excludes_navigation(self):
        from healthvault.rag.sources import official_text
        value = official_text({'body':b'<html><nav>menu</nav><main><h1>Official report</h1><p>verified fact</p><table><tr><td>label</td><td>detail</td></tr></table><script>private code</script></main><footer>footer</footer></html>'})
        for text in ('Official report','verified fact','label','detail'):
            self.assertIn(text,value)
        for text in ('menu','private code','footer'):
            self.assertNotIn(text,value)

if __name__ == '__main__': unittest.main()
