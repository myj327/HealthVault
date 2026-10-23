import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from healthvault.rag.budget import TokenBudget
from healthvault.rag.chunking import SourceBlock, SourceSection, SourceSnapshot, build_chunks
from healthvault.rag.corpus import DEFAULT_CORPUS_ROOT, load_chunk_artifact, render_corpus_artifacts
from healthvault.rag.releases import KnowledgeReleaseStore
from tests.healthvault.test_knowledge_releases import update_payload


class ChunkBudgetTests(unittest.TestCase):
    def source(self, text, *, kind='paragraph', header=None):
        return SourceSnapshot('test', 'NIDDK', 'Test', 'v1', 'https://www.niddk.nih.gov/test', 'zh', '2026-10-01', 'public domain', 'a'*64,
            (SourceSection(('Test',), ('test',), (SourceBlock(kind, text, header),)),))

    def test_unbroken_unicode_and_cjk_sentences_are_lossless(self):
        for text in ('甲状腺报告内容' * 400, 'a' * 4500, ('first sentence。下一句！' * 220)):
            budget = TokenBudget()
            chunks = build_chunks((self.source(text),), 'hybrid_v1', corpus_version='budget-test-v1', token_budget=budget)
            self.assertGreater(len(chunks), 1)
            self.assertEqual(text, ''.join(text[span['start']:span['end']] for span in budget.spans))
            self.assertTrue(all(budget.fits(chunk.body, 'Test\ntest\n') for chunk in chunks))
            self.assertTrue(all(chunk.overlap_estimated_tokens == 0 for chunk in chunks))

    def test_long_table_row_and_header_have_bounded_continuations(self):
        for header in ('name | description | value', 'header' * 500):
            text = 'row-1 | ' + '内容' * 1800 + ' | 7.5'
            budget = TokenBudget()
            chunks = build_chunks((self.source(text, kind='table', header=header),), 'hybrid_v1', token_budget=budget)
            natural = header + '\n' + text
            self.assertEqual(natural, ''.join(natural[span['start']:span['end']] for span in budget.spans))
            self.assertTrue(all(budget.fits(chunk.body, 'Test\ntest\n') for chunk in chunks))
            self.assertGreater(len(chunks), 1)

    def test_real_local_tokenizer_is_used_without_truncation(self):
        from tokenizers import Tokenizer, models, pre_tokenizers
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'tokenizer.json'
            tokenizer = Tokenizer(models.WordLevel({'[UNK]': 0, 'word': 1}, unk_token='[UNK]'))
            tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
            tokenizer.enable_truncation(max_length=10)
            tokenizer.save(str(path))
            budget = TokenBudget(path)
            self.assertEqual(1500, budget.count('word ' * 1500))
            self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(), budget.identity)
            chunks = build_chunks((self.source('word ' * 1500),), 'hybrid_v1', token_budget=budget)
            self.assertTrue(all(budget.count('Test\ntest\n' + chunk.body) <= 600 for chunk in chunks))

    def test_frozen_corpus_is_byte_identical(self):
        for path, data in render_corpus_artifacts(DEFAULT_CORPUS_ROOT).items():
            self.assertEqual((DEFAULT_CORPUS_ROOT / path).read_bytes(), data)

    def test_release_loads_fallback_and_rejects_span_tampering(self):
        with tempfile.TemporaryDirectory() as directory:
            store = KnowledgeReleaseStore(Path(directory))
            payload = update_payload()
            section = payload['source_updates'][0]['sections'][0]
            section['blocks'][0]['text'] += ' amberquartz' * 700
            source = payload['source_updates'][0]
            source['source_sha256'] = hashlib.sha256(json.dumps(source['sections'], ensure_ascii=False, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
            record = store.stage(payload, 'admin')
            corpus = store.root / 'candidates' / record['candidate_id'] / 'corpus'
            artifact = load_chunk_artifact(corpus / 'chunks' / 'hybrid_v1.json', expected_corpus_version=None)
            self.assertTrue(artifact.chunks)
            spans = corpus / 'chunks' / 'boundary-spans.json'
            value = json.loads(spans.read_text())
            value['chunks'][0]['end'] += 1
            spans.write_text(json.dumps(value))
            with self.assertRaisesRegex(ValueError, 'boundary spans'):
                store.decide(record['candidate_id'], 'approve', 'admin')

if __name__ == '__main__':
    unittest.main()
