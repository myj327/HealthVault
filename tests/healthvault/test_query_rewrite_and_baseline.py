from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from healthvault.application.query_rewrite import prepare_query
from healthvault.infrastructure.hybrid_health_knowledge import _chunk_ids_sha256
from healthvault.rag import retrieval_config as config
from healthvault.rag.corpus import load_chunk_artifact


class QueryRewriteTests(unittest.TestCase):
    def test_rule_resolves_one_topic_and_keeps_original_constraints(self):
        original = '这个指标与去年相比变化了吗？'
        result = prepare_query(original, context_question='解释 HbA1c 是什么', knowledge_only=False)
        self.assertEqual(original, result.original_question)
        self.assertEqual('HbA1c与去年相比变化了吗？', result.resolved_question)
        self.assertEqual('context_rule', result.method)
        explicit = prepare_query('HbA1c 是什么？what does it measure?')
        self.assertEqual('ready', explicit.status)
        self.assertEqual('question_rule', explicit.method)
        compared = prepare_query('这个指标和空腹血糖有什么区别？', context_question='解释 HbA1c')
        self.assertEqual('HbA1c和空腹血糖有什么区别？',compared.resolved_question)

    def test_missing_or_multiple_topics_require_clarification_and_blocked_queries_make_no_model_call(self):
        model = MagicMock()
        for context in ('', '解释 HbA1c 和血糖'):
            result = prepare_query('这个指标是什么意思？', context_question=context, model_query=model)
            self.assertEqual('needs_clarification', result.status)
        self.assertEqual('unsupported', prepare_query('这个指标需要吃什么药？', model_query=model).status)
        model.assert_not_called()

    def test_model_cannot_drop_date_side_negation_or_add_new_topic(self):
        original = '2025年左侧的 HbA1c 不是 7.2 吗？'
        for proposal in ('右侧的 HbA1c 是 7.2 吗？', '2025年左侧的血糖不是7.2吗？'):
            model = MagicMock(return_value=json.dumps({'search_query':proposal,'needs_clarification':False}))
            result = prepare_query(original, model_query=model)
            self.assertEqual(original, result.search_query)
            self.assertEqual(original, result.original_question)
        model = MagicMock(return_value=json.dumps({'search_query':'糖化血红蛋白 是什么？','needs_clarification':False}))
        self.assertEqual('model_validated', prepare_query('HbA1c 是什么？', model_query=model).method)
        self.assertEqual('model_unavailable', prepare_query('HbA1c 是什么？', model_query=MagicMock(side_effect=RuntimeError())).method)


class QueryRewriteWebTests(unittest.TestCase):
    def test_original_question_and_resolved_topic_reach_answer_and_verifier(self):
        from tests.healthvault.test_web_answers import StubController, TestClient
        from healthvault.web.api import create_app

        text = '糖化血红蛋白是一项检测。'
        hit = {'citation_id':'HVK:synthetic-rewrite','title':'糖化血红蛋白','excerpt':text,'source':{'url':''}}
        answerer = MagicMock(return_value=text+' [HVK:synthetic-rewrite]')
        verifier = MagicMock(return_value={'answer_supported':True,'answers_question':True,'claims':[
            {'text':text,'quote':text,'citation_id':hit['citation_id'],'supported':True}]})
        rewrite = MagicMock(return_value=json.dumps({'search_query':'糖化血红蛋白是什么？','needs_clarification':False}))
        original = '这个指标是什么？'
        controller = StubController(hits=[hit])
        with TestClient(create_app(controller, answerer, model_answer_verifier=verifier,
                                   model_query_rewriter=rewrite)) as client:
            response = client.post('/api/answers/knowledge',json={
                'question':original,'context_question':'解释 HbA1c','allow_query_rewrite':True})
            self.assertEqual('answered',response.json()['status'],response.text)
            self.assertEqual(original,response.json()['question'])
            self.assertEqual('HbA1c是什么？',response.json()['resolved_question'])
            self.assertEqual(original,answerer.call_args.args[0])
            self.assertEqual('HbA1c是什么？',answerer.call_args.args[2]['question_context']['resolved_question'])
            self.assertEqual(original,verifier.call_args.args[1]['question'])
            self.assertEqual('HbA1c是什么？',verifier.call_args.args[1]['resolved_question'])
            self.assertEqual(original,rewrite.call_args.args[1]['original_question'])
            history = client.get('/api/profiles/demo-a/answers/history').json()['answers']
            self.assertEqual(original, history[0]['question'])
            self.assertEqual('HbA1c是什么？', history[0]['resolved_question'])
            continued = client.post('/api/answers/knowledge',json={
                'question':'它需要空腹吗？','context_question':history[0]['resolved_question']})
            self.assertEqual('answered',continued.json()['status'],continued.text)
            self.assertEqual('HbA1c需要空腹吗？',continued.json()['resolved_question'])
            for mock in (answerer,verifier,rewrite): mock.reset_mock()
            ambiguous = client.post('/api/answers/knowledge',json={
                'question':original,'context_question':'解释 HbA1c 和血糖','allow_query_rewrite':True})
            self.assertEqual('insufficient_evidence',ambiguous.json()['status'])
            for mock in (answerer,verifier,rewrite): mock.assert_not_called()
        controller.directory.cleanup()


class BaselineVectorTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.addCleanup(self.temporary.cleanup)
        self.source = self.root / 'index'
        self.source.mkdir()
        artifact_path = config.DEFAULT_CORPUS_ROOT / 'chunks/hybrid_v1.json'
        artifact = load_chunk_artifact(artifact_path)
        self.source.joinpath('index.faiss').write_bytes(b'synthetic-vector-index')
        ids = [chunk.chunk_id for chunk in artifact.chunks]
        self.source.joinpath('chunk_ids.json').write_text(json.dumps(ids), encoding='utf-8')
        import hashlib
        self.manifest = {'index_version':'hybrid-r1','corpus_sha256':hashlib.sha256(artifact_path.read_bytes()).hexdigest(),
                         'corpus_version':artifact.corpus_version,'model_name':'BAAI/bge-m3','dimension':1024,
                         'chunk_ids_sha256':_chunk_ids_sha256(ids),'model_sha256':'synthetic-model',
                         'index_sha256':hashlib.sha256(b'synthetic-vector-index').hexdigest()}
        self.write_manifest()
        self.model_patch = patch.object(config,'configured_model_path',return_value=self.root/'model')
        self.fingerprint_patch = patch.object(config,'model_fingerprint',return_value='synthetic-model')
        self.index_patch = patch.object(config.FaissVectorIndex,'load')
        self.model_patch.start(); self.fingerprint_patch.start()
        self.index_loader = self.index_patch.start()
        self.addCleanup(self.model_patch.stop); self.addCleanup(self.fingerprint_patch.stop)
        self.addCleanup(self.index_patch.stop)

    def write_manifest(self):
        self.source.joinpath('manifest.json').write_text(json.dumps(self.manifest),encoding='utf-8')

    def test_matching_index_is_copied_without_changing_baseline(self):
        artifact = config.DEFAULT_CORPUS_ROOT/'chunks/hybrid_v1.json'
        before = artifact.read_bytes()
        value = config.activate_baseline(self.source,self.root/'release')
        self.index_loader.assert_called_once_with(self.source/'index.faiss',self.source/'chunk_ids.json',1024)
        self.assertEqual(before,artifact.read_bytes())
        self.assertEqual(b'synthetic-vector-index',(Path(value['index_directory'])/'index.faiss').read_bytes())
        with patch.dict('os.environ',{},clear=True):
            self.assertEqual('hybrid',config.selection(config.DEFAULT_CORPUS_ROOT,self.root/'release')['knowledge_mode'])

    def test_different_model_or_corpus_and_tampered_index_are_rejected(self):
        for key in ('model_sha256','corpus_sha256'):
            previous=self.manifest[key]; self.manifest[key]='changed'; self.write_manifest()
            with self.assertRaises(ValueError): config.activate_baseline(self.source,self.root/'release')
            self.manifest[key]=previous
        self.write_manifest()
        self.source.joinpath('index.faiss').write_bytes(b'tampered')
        with self.assertRaises(ValueError): config.activate_baseline(self.source,self.root/'release')
        self.assertFalse((self.root/'release/baseline-retrieval.json').exists())

    def test_explicit_lexical_override_and_new_release_index_selection(self):
        config.activate_baseline(self.source,self.root/'release')
        with patch.dict('os.environ',{'HEALTHVAULT_KNOWLEDGE_MODE':'lexical'},clear=True):
            self.assertEqual('lexical',config.selection(config.DEFAULT_CORPUS_ROOT,self.root/'release')['knowledge_mode'])
        corpus=self.root/'new-corpus'
        corpus.joinpath('vectors').mkdir(parents=True)
        corpus.joinpath('vectors/manifest.json').write_text('{}',encoding='utf-8')
        self.assertEqual(str(corpus/'vectors'),config.selection(corpus,self.root/'release')['hybrid_index_path'])


if __name__ == '__main__': unittest.main()
