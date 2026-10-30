import json
import os
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

from healthvault.application.knowledge_support_gate import assess_health_knowledge_evidence
from healthvault.application.search_health_knowledge import search_health_knowledge
from healthvault.application.workflow_contract import knowledge_search_issue
from healthvault.infrastructure.hybrid_health_knowledge import HybridHealthKnowledgeRepository
from healthvault.infrastructure.sqlite_fts_health_knowledge import SQLiteFTSHealthKnowledgeRepository
from healthvault.ui.controller import configured_database_path, PROJECT_ROOT
from healthvault.web.agent_tasks import present_agent_reply, public_task


QUESTION = 'LDL-C 和 HDL-C 分别是什么'


class BasicKnowledgeTests(unittest.TestCase):
    def setUp(self):
        self.repository = SQLiteFTSHealthKnowledgeRepository()
        self.addCleanup(self.repository.close)

    def test_raw_bilingual_questions_return_actual_ldl_and_hdl_definitions(self):
        for query in (QUESTION, QUESTION + '？请给出官方资料来源。',
                      '低密度脂蛋白胆固醇和高密度脂蛋白胆固醇分别是什么',
                      'LDL-C 和 HDL-C 有什么区别', 'What are LDL-C and HDL-C?'):
            with self.subTest(query=query):
                result = search_health_knowledge({'query': query}, self.repository)
                self.assertEqual('ok', result.status)
                body = ' '.join(hit.document.summary.casefold() for hit in result.hits)
                for term in ('ldl', 'hdl', 'low-density', 'high-density'):
                    self.assertIn(term, body)
                self.assertTrue(all(hit.document.source.url.startswith(('https://www.nhlbi.nih.gov/', 'https://medlineplus.gov/')) for hit in result.hits))

    def test_three_actual_failed_searches_now_return_sources(self):
        for query, term in (('LDL-C 低密度脂蛋白胆固醇 是什么', 'ldl'),
                            ('HDL-C 高密度脂蛋白胆固醇 是什么', 'hdl'),
                            ('血脂 胆固醇 脂蛋白', 'cholesterol'),
                            ('LDL-C 低密度脂蛋白胆固醇 定义', 'ldl'),
                            ('HDL-C 高密度脂蛋白胆固醇 定义', 'hdl'),
                            ('血脂检查 胆固醇 脂蛋白 项目 含义', 'cholesterol')):
            with self.subTest(query=query):
                result = search_health_knowledge({'query': query}, self.repository)
                self.assertEqual('ok', result.status)
                self.assertIn(term, ' '.join(hit.document.summary.casefold() for hit in result.hits))

    def test_new_topic_search_after_results_keeps_duplicate_failure_and_budget_guards(self):
        for previous in ('ok', 'partial', 'empty'):
            self.assertIsNone(knowledge_search_issue('hdl', {'ldl'}, previous))
        self.assertEqual('workflow_search_duplicate_query', knowledge_search_issue('LDL', {'ldl'}, 'ok'))
        self.assertEqual('workflow_search_budget_exceeded', knowledge_search_issue('tc', {'ldl','hdl','tg'}, 'ok'))
        self.assertEqual('workflow_search_after_failed_result', knowledge_search_issue('hdl', {'ldl'}, 'error'))

    def test_hybrid_fusion_and_lexical_use_the_same_source_support(self):
        lexical = self.repository.search(QUESTION, 3)
        class Embedder:
            dimension = 2
            def embed(self, texts): return ((1.0, 0.0),)
        class Index:
            dimension = 2
            def search(self, vector, limit): return tuple((hit.document.chunk_id, 1.0) for hit in lexical[:limit])
        hybrid = HybridHealthKnowledgeRepository(self.repository, embedder=Embedder(), vector_index=Index())
        result = search_health_knowledge({'query': QUESTION}, hybrid)
        self.assertEqual('ok', result.status)
        self.assertTrue(hybrid.last_search_metadata['dense_participated'])
        self.assertTrue(all(not hit.matched_terms for hit in result.hits))
        self.assertEqual({hit.document.citation_id for hit in lexical}, {hit.document.citation_id for hit in result.hits})

    def test_query_metadata_and_missing_topics_cannot_bypass_source_checks(self):
        real = self.repository.search('LDL HDL cholesterol', 3)[0]
        unrelated = replace(real, document=replace(real.document, title='Generic test', topic='test',
                            topic_tags=(), summary='A general laboratory test.'), matched_terms=('ldl', 'hdl', '低密度脂蛋白', '高密度脂蛋白'))
        self.assertFalse(assess_health_knowledge_evidence(QUESTION, (unrelated,)).allowed)
        only_ldl = replace(unrelated, document=replace(unrelated.document, summary='LDL means low-density lipoprotein.'))
        self.assertFalse(assess_health_knowledge_evidence(QUESTION, (only_ldl,)).allowed)
        for query in ('A1C 和甲状腺超声分别是什么', '根据血糖结果诊断我得了什么病', 'LDL-C 药物剂量是多少'):
            with self.subTest(query=query):
                self.assertEqual('empty', search_health_knowledge({'query': query}, self.repository).status)

    def test_terminal_json_is_presented_for_new_and_historical_dialogue_only(self):
        for status in ('insufficient_evidence', 'rejected', 'ok'):
            raw = json.dumps(dict(final_status=status, citations=[], refusal=status != 'ok'))
            display, metadata = present_agent_reply(raw)
            self.assertFalse(display.startswith('{'))
            self.assertEqual(status, metadata['final_status'])
            saved = dict(version=1, folded=[], updated_at='synthetic', dialogue=[{'role':'user','text':raw}, {'role':'assistant','text':raw}])
            task = public_task(saved, 'synthetic')
            self.assertEqual(raw, task['dialogue'][0]['text'])
            self.assertEqual(display, task['dialogue'][1]['text'])
            self.assertEqual(raw, saved['dialogue'][1]['text'])
        natural = '这是带来源的正常回答。[HVK:test:source]'
        self.assertEqual((natural, None), present_agent_reply(natural))

    def test_database_path_does_not_follow_working_directory_or_switch_explicit_database(self):
        original = Path.cwd()
        self.addCleanup(os.chdir, original)
        with tempfile.TemporaryDirectory(dir=PROJECT_ROOT/'.task_outputs') as directory:
            for setting in ('', '.task_outputs/interview-demo-20261027/healthvault.sqlite3', str(Path(directory)/'explicit.sqlite3')):
                with self.subTest(setting=setting), patch.dict(os.environ, {'HEALTHVAULT_UI_DB_PATH': setting}):
                    os.chdir(PROJECT_ROOT)
                    expected = configured_database_path()
                    os.chdir(directory)
                    self.assertEqual(expected, configured_database_path())
                    self.assertTrue(expected.is_absolute())
            os.chdir(original)

    def test_http_entry_preserves_answer_sources_and_machine_terminal_status(self):
        from tests.healthvault import test_task114_entrypoint as entry
        fixture = entry.EntryPointTests()
        fixture.setUp()
        self.addCleanup(fixture.doCleanups)
        citation = self.repository.search(QUESTION, 3)[0].document.citation_id
        answer = f'LDL-C 是低密度脂蛋白胆固醇，HDL-C 是高密度脂蛋白胆固醇。[{citation}]'
        row, calls, _ = fixture.collect({'case_id':'lipids', 'profile_id':'r5-empty', 'prompt':QUESTION},
                                      [entry.tool('search_health_knowledge', {'query': QUESTION}), entry.response(answer), entry.verification])
        self.assertEqual(200, row['http_status'])
        self.assertEqual(answer, row['response']['answer'])
        self.assertEqual(answer, row['response']['dialogue'][-1]['text'])
        self.assertIsNone(row['response']['final_result'])
        self.assertEqual(1, len(row['tool_calls']))
        self.assertTrue(row['sources'])
        self.assertTrue(row['database_unchanged'])
        batch = entry.tool('search_health_knowledge', {'query':'LDL-C 低密度脂蛋白胆固醇 定义'})
        second = entry.tool('search_health_knowledge', {'query':'HDL-C 高密度脂蛋白胆固醇 定义'})['choices'][0]['message']['tool_calls'][0]
        second['id'] = 'call-2'
        batch['choices'][0]['message']['tool_calls'].append(second)
        batch_citation = self.repository.search('LDL-C 低密度脂蛋白胆固醇 定义', 3)[0].document.citation_id
        batch_answer = f'LDL-C 是低密度脂蛋白胆固醇，HDL-C 是高密度脂蛋白胆固醇。[{batch_citation}]'
        batch_row, _, _ = fixture.collect({'case_id':'lipids-batch', 'profile_id':'r5-empty', 'prompt':QUESTION},
                                          [batch, entry.response(batch_answer), entry.verification])
        self.assertEqual(200, batch_row['http_status'])
        self.assertEqual(batch_answer, batch_row['response']['answer'])
        self.assertEqual(['ok','ok'], [call['result_status'] for call in batch_row['tool_calls']])
        raw = json.dumps(dict(final_status='insufficient_evidence', citations=[], refusal=True))
        empty, _, client = fixture.collect({'case_id':'empty-lipids', 'profile_id':'r5-empty', 'prompt':QUESTION},
                                          [entry.tool('search_health_knowledge', {'query':QUESTION}), entry.response(raw)],
                                          result={'status':'empty', 'available_citation_ids':[], 'results':[]})
        self.assertEqual(200, empty['http_status'])
        self.assertEqual('insufficient_evidence', empty['response']['final_result']['final_status'])
        self.assertIn('本次检索', empty['response']['answer'])
        self.assertNotIn('final_status', empty['response']['dialogue'][-1]['text'])
        self.assertTrue(empty['database_unchanged'])


if __name__ == '__main__': unittest.main()
