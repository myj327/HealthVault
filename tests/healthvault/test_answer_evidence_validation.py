import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from fastapi.testclient import TestClient
from healthvault.web.answer_validation import verify_answer
from healthvault.web.api import create_app
from healthvault.ui.controller import HealthVaultUIController
from healthvault.infrastructure.sqlite_fts_health_knowledge import SQLiteFTSHealthKnowledgeRepository
from healthvault.tools.search_official_health_web import search_official_health_web


SOURCE = {'citation_id':'REC:synthetic:thyroid','title':'2026-10-01 检查报告','detail':'甲状腺左叶结节 6×4 mm。'}


class AnswerEvidenceValidationTests(unittest.TestCase):
    def verdict(self, text, quote, supported=True, answers=True):
        return {'answer_supported':supported,'answers_question':answers,'claims':[{'text':text,'quote':quote,'citation_id':SOURCE['citation_id'],'supported':supported}]}

    def test_valid_literal_fact_and_date_metadata_are_bound(self):
        text = '左叶结节尺寸为 6×4 mm。'
        result = verify_answer('左叶结节多大？',text+' ['+SOURCE['citation_id']+']',[SOURCE],query=lambda *a,**k:self.verdict(text,SOURCE['detail']))
        self.assertTrue(result['passed'])
        dated = self.verdict('检查日期为2026-10-01。','2026-10-01'); dated['claims'][0]['source_field']='title'
        self.assertTrue(verify_answer('检查日期是什么？','检查日期为2026-10-01。 ['+SOURCE['citation_id']+']',[SOURCE],query=lambda *a,**k:dated)['passed'])

    def test_forged_number_quote_or_insufficient_question_fails(self):
        for text,quote,answers in [('结节是 60 mm。',SOURCE['detail'],True),('结节是 6 mm。','伪造原文',True),('左叶结节 6×4 mm。',SOURCE['detail'],False)]:
            with self.subTest(text=text,quote=quote):
                result = verify_answer('结节增长了吗？',text+' ['+SOURCE['citation_id']+']',[SOURCE],query=lambda *a,**k:self.verdict(text,quote,answers=answers))
                self.assertFalse(result['passed'])

    def test_failed_verification_never_enters_answer_history(self):
        with tempfile.TemporaryDirectory() as directory:
            app = create_app(HealthVaultUIController(Path(directory)/'records.sqlite3'),
                             model_answerer=lambda *a:'结节是 60 mm。 ['+SOURCE['citation_id']+']',
                             model_answer_verifier=lambda *a,**k:self.verdict('结节是 60 mm。',SOURCE['detail']))
            with TestClient(app) as client, patch('healthvault.web.api.retrieve_question_knowledge',return_value={'results':[{'citation_id':SOURCE['citation_id'],'title':SOURCE['title'],'excerpt':SOURCE['detail']}]}):
                account = client.post('/api/auth/register',json={'username':'evidence-user','password':'synthetic-password-123'}).json()
                client.headers.update({'X-CSRF-Token':account['csrf_token']})
                response = client.post('/api/answers/knowledge',json={'question':'甲状腺结节是什么？','model_consent':True})
                self.assertEqual('insufficient_evidence',response.json()['status'])
                self.assertEqual([],app.state.answer_history.list(account['profile_id']))

    def test_known_niddk_source_can_supplement_without_sending_question(self):
        repository = SQLiteFTSHealthKnowledgeRepository()
        calls = []
        def fetch(url, bucket):
            calls.append(url)
            return {'url':url,'text':'The A1C test measures glycated hemoglobin and A1C.','snapshot_sha256':'a'*64}
        try:
            with patch('healthvault.tools.search_official_health_web._cached_official_document',side_effect=fetch), patch('healthvault.tools.search_official_health_web._cached_xml',return_value=b'<nlmSearchResult/>'):
                result = search_official_health_web({'question':'张三的糖化血红蛋白是什么？','local_answer_insufficient':True},repository)
            self.assertEqual('ok',result['status'])
            self.assertTrue(any(item['source']['publisher']=='NIDDK' for item in result['results']))
            self.assertTrue(calls); self.assertTrue(all('张三' not in url and '?' not in url for url in calls))
        finally: repository.close()

if __name__ == '__main__': unittest.main()
