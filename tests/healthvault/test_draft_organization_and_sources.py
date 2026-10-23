import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch
from dataclasses import replace

from fastapi.testclient import TestClient
from healthvault.ocr.pipeline import OCRProcessingError
from healthvault.ui.controller import HealthVaultUIController
from healthvault.web.api import create_app, _public_skill_review
from healthvault.web.answers import visit_evidence
from healthvault.web.drafts import Draft, DraftStore, DraftConflict
from healthvault.web.skill_descriptions import usage_explanation


class DraftOrganizationAndSourcesTests(unittest.TestCase):
    def test_archive_restore_filters_and_profile_ownership_survive_restart(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            app = create_app(HealthVaultUIController(root / 'records.sqlite3', knowledge_release_root=root / 'releases'))
            with TestClient(app) as client:
                account = client.post('/api/auth/register', json={'username': 'draft-owner', 'password': 'synthetic-password-123'}).json()
                profile = account['profile_id']
                client.headers.update({'X-CSRF-Token': account['csrf_token']})
                base = f'/api/profiles/{profile}/uploads'
                text = '报告日期: 2026-10-01\n空腹血糖 5.2 mmol/L 参考 3.9-6.1'
                draft = client.post('/api/drafts/text', json={'profile_id': profile, 'source_id': 'draft-source', 'report_text': text}).json()
                token = draft['token']
                self.assertEqual(1, len(client.get(base + '?status=pending').json()['uploads']))
                self.assertEqual([], client.get(base + '?status=failed').json()['uploads'])
                self.assertEqual(200, client.post(f'/api/drafts/{token}/archive').status_code)
                self.assertEqual([], client.get(base).json()['uploads'])
                self.assertEqual(409, client.put(f'/api/drafts/{token}', json={'profile_id': profile, 'report_text': text, 'reviewed': False}).status_code)
                reopened = TestClient(create_app(HealthVaultUIController(root / 'records.sqlite3', knowledge_release_root=root / 'releases')))
                reopened.cookies.update(client.cookies); reopened.headers.update(client.headers)
                try:
                    self.assertEqual(1, len(reopened.get(base + '?status=archived').json()['uploads']))
                    restored = reopened.post(f'/api/drafts/{token}/restore').json()
                    self.assertIsNone(restored['archived_at'])
                    self.assertEqual(text, restored['original_text'])
                    self.assertEqual(1, len(reopened.get(base).json()['uploads']))
                    confirmed = reopened.post(f'/api/drafts/{token}/confirm', json={'profile_id': profile, 'acknowledged': True,
                        'preview_sha256': restored['preview_sha256'], 'report_text': text})
                    self.assertEqual(200, confirmed.status_code, confirmed.text)
                    self.assertEqual(409, reopened.post(f'/api/drafts/{token}/archive').status_code)
                finally:
                    reopened.close()
                with TestClient(app) as other:
                    other_account = other.post('/api/auth/register', json={'username': 'draft-other', 'password': 'synthetic-password-123'}).json()
                    other.headers.update({'X-CSRF-Token': other_account['csrf_token']})
                    self.assertEqual(404, other.post(f'/api/drafts/{token}/restore').status_code)

    def test_failed_file_keeps_bytes_when_archived_and_duplicate_hint_is_user_scoped(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            controller = HealthVaultUIController(root / 'records.sqlite3', knowledge_release_root=root / 'releases')
            controller.process_image = Mock(side_effect=OCRProcessingError('synthetic failure'))
            app = create_app(controller)
            raw = b'\x89PNG\r\n\x1a\nsynthetic-only'
            with TestClient(app) as client:
                account = client.post('/api/auth/register', json={'username': 'duplicate-owner', 'password': 'synthetic-password-123'}).json()
                client.headers.update({'X-CSRF-Token': account['csrf_token']})
                body = {'profile_id': account['profile_id']}
                files = {'file': ('synthetic.png', raw, 'image/png')}
                first = client.post('/api/drafts/image', data=body, files=files).json()
                second = client.post('/api/drafts/image', data=body, files=files).json()
                self.assertIsNone(first['duplicate_of'])
                self.assertEqual(first['source_id'], second['duplicate_of']['source_id'])
                self.assertEqual(2, len(client.get(f"/api/profiles/{account['profile_id']}/uploads?status=failed").json()['uploads']))
                archived = client.post(f"/api/drafts/{first['token']}/archive").json()
                self.assertEqual('extraction_failed', archived['processing_status'])
                self.assertEqual(raw, client.get(first['image_url']).content)
                with TestClient(app) as other:
                    other_account = other.post('/api/auth/register', json={'username': 'duplicate-other', 'password': 'synthetic-password-123'}).json()
                    other.headers.update({'X-CSRF-Token': other_account['csrf_token']})
                    owned = other.post('/api/drafts/image', data={'profile_id': other_account['profile_id']}, files=files).json()
                    self.assertIsNone(owned['duplicate_of'])

    def test_text_duplicate_is_based_on_original_content_and_updates_do_not_duplicate_self(self):
        with tempfile.TemporaryDirectory() as directory:
            store = DraftStore(Path(directory) / 'uploads.sqlite3')
            first = Draft('first', 'owner', 'source-a', '原文', 'text', original_text='原文')
            store.save(first)
            second = Draft('second', 'owner', 'source-b', '原文', 'text', original_text='原文')
            store.save(second)
            self.assertEqual('source-a', second.duplicate_of['source_id'])
            first.report_text = '已核对的文字'; store.save(first)
            self.assertIsNone(store.get('first').duplicate_of)
            self.assertEqual('原文', store.get('first').original_text)

    def test_inflight_edit_cannot_silently_unarchive_a_draft(self):
        with tempfile.TemporaryDirectory() as directory:
            store = DraftStore(Path(directory) / 'uploads.sqlite3')
            draft = Draft('original', 'owner', 'source', '原文', 'text')
            store.save(draft)
            pending_edit = replace(draft, token='edited', report_text='待保存的编辑')
            draft.archived_at = '2026-10-02T00:00:00Z'; store.save(draft)
            with self.assertRaises(DraftConflict):
                store.save(pending_edit, previous_token='original')
            self.assertTrue(store.get('original').archived_at)
            self.assertEqual('原文', store.get('original').report_text)

    def test_record_and_comparison_sources_link_to_both_original_reports(self):
        records = [{'citation_id': f'REC:{identifier}:glucose', 'report_id': identifier, 'report_date': date,
                    'indicator_code': 'glucose', 'evidence': {'quote': f'空腹血糖 {value} mmol/L', 'line_start': 2, 'span_start': 15}}
                   for identifier, date, value in [('old', '2025-10-01', 5.2), ('new', '2026-10-01', 5.6)]]
        brief = {'record_citations': records, 'indicators': [{'code': 'glucose', 'display_name': '空腹血糖', 'latest': {'report_id': 'new'},
                 'comparison': {'earliest_value': 5.2, 'latest_value': 5.6, 'unit': 'mmol/L', 'absolute_change': 0.4,
                    'earliest_record_citation_id': 'REC:old:glucose', 'latest_record_citation_id': 'REC:new:glucose'}}]}
        _, sources = visit_evidence(brief)
        self.assertEqual('old', sources[0]['report_id'])
        self.assertEqual(15, sources[0]['evidence_locator']['span_start'])
        self.assertEqual(['old', 'new'], [item['report_id'] for item in sources[-1]['report_links']])
        self.assertEqual(records[0]['evidence'], sources[-1]['report_links'][0]['evidence'])

    def test_chinese_skill_explanations_use_actual_sections_and_recorded_failure(self):
        public = _public_skill_review({'patches': [{'target_section': 'Steps', 'target_paragraph': '2. Run the approved tools.'}, {'target_section': 'Verification'}],
                                      'candidate_content': 'private candidate', 'evidence_fingerprint': 'private'})
        self.assertIn('工具选择与执行顺序', public['explanation_zh'])
        self.assertIn('第 2 步', public['explanation_zh'])
        self.assertIn('结果与引用核对', public['explanation_zh'])
        self.assertNotIn('candidate_content', public)
        summary = {'last_action': {'problem_sha256': 'bad', 'actions': ['disable', 'rollback']},
                   'evaluations': [{'skill_sha256': 'bad', 'safety_hard_failures': 1}]}
        self.assertIn('安全失败', usage_explanation(summary))
        self.assertIn('恢复之前版本', usage_explanation(summary))
        self.assertIn('主动回滚', usage_explanation({'last_action': {'actions': ['rollback'], 'reason': 'user_requested'}}))
        self.assertIn('未保存具体原因', usage_explanation({'last_action': {'actions': ['disable']}}))

    def test_confirmed_pdf_field_region_is_accessible_only_to_its_owner(self):
        import pymupdf
        with pymupdf.open() as document:
            document.new_page().insert_text((40, 40), 'Synthetic report only')
            raw = document.tobytes()
        text = 'report_id: source-pdf\nreport_date: 2026-10-01\nfacility: synthetic\nresult|fasting_glucose|5.2|mmol/L|3.9|6.1'
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            app = create_app(HealthVaultUIController(root / 'records.sqlite3', knowledge_release_root=root / 'releases'))
            with TestClient(app) as client:
                account = client.post('/api/auth/register', json={'username': 'pdf-owner', 'password': 'synthetic-password-123'}).json()
                profile = account['profile_id']; client.headers.update({'X-CSRF-Token': account['csrf_token']})
                with patch('healthvault.web.api.extract_pdf_document', return_value={'text': text, 'blocks': [], 'warnings': []}):
                    draft = client.post('/api/drafts/pdf', data={'profile_id': profile}, files={'file': ('synthetic.pdf', raw, 'application/pdf')}).json()
                updated = client.put(f"/api/drafts/{draft['token']}", json={'profile_id': profile, 'report_text': text, 'reviewed': True}).json()
                confirmed = client.post(f"/api/drafts/{updated['token']}/confirm", json={'profile_id': profile, 'report_text': text,
                    'preview_sha256': updated['preview_sha256'], 'acknowledged': True})
                self.assertEqual(200, confirmed.status_code, confirmed.text)
                base = f'/api/profiles/{profile}/reports/source-pdf/pages/'
                rendered = client.get(base + '1')
                self.assertEqual(200, rendered.status_code, rendered.text[:100] if rendered.status_code != 200 else '')
                self.assertTrue(rendered.content.startswith(b'\x89PNG'))
                self.assertEqual(404, client.get(base + '2').status_code)
                with TestClient(app) as other:
                    other.post('/api/auth/register', json={'username': 'pdf-other', 'password': 'synthetic-password-123'})
                    self.assertEqual(404, other.get(base + '1').status_code)
