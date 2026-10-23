import sqlite3
import tempfile
import unittest
from contextlib import closing
from pathlib import Path

from fastapi.testclient import TestClient
from healthvault.ui.controller import HealthVaultUIController
from healthvault.web.api import create_app
from healthvault.web.answer_history import WebAnswerHistory


class WorkflowContinuityTests(unittest.TestCase):
    def test_rescheduling_or_changing_notice_days_resets_acknowledgement(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            controller = HealthVaultUIController(root / 'records.sqlite3', knowledge_release_root=root / 'releases')
            with TestClient(create_app(controller)) as client:
                account = client.post('/api/auth/register', json={'username': 'continuity-owner', 'password': 'synthetic-password-123'}).json()
                client.headers.update({'X-CSRF-Token': account['csrf_token']})
                base = f"/api/profiles/{account['profile_id']}/calendar"
                for index, changes in enumerate(({'scheduled_date': '2000-01-02'}, {'remind_before_days': 2})):
                    created = client.post(base + '/events', json={'idempotency_key': f'notice-{index}', 'title': '复诊', 'scheduled_date': '2000-01-01'})
                    event_id = created.json()['event']['event_id']
                    url = base + '/events/' + event_id
                    self.assertEqual('acknowledged', client.post(url + '/acknowledge').json()['event']['reminder_status'])
                    updated = client.patch(url, json=changes)
                    self.assertEqual(200, updated.status_code, updated.text)
                    self.assertIsNone(updated.json()['event']['reminder_acknowledged_at'])
                    self.assertNotEqual('acknowledged', updated.json()['event']['reminder_status'])
                reopened = TestClient(create_app(HealthVaultUIController(root / 'records.sqlite3', knowledge_release_root=root / 'releases')))
                reopened.cookies.update(client.cookies)
                try:
                    events = reopened.get(base).json()['events']
                    self.assertTrue(all(item['reminder_status'] == 'overdue' for item in events))
                finally:
                    reopened.close()

    def test_title_only_edit_keeps_acknowledgement_and_other_profile_cannot_edit(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            app = create_app(HealthVaultUIController(root / 'records.sqlite3', knowledge_release_root=root / 'releases'))
            with TestClient(app) as client:
                account = client.post('/api/auth/register', json={'username': 'notice-owner', 'password': 'synthetic-password-123'}).json()
                client.headers.update({'X-CSRF-Token': account['csrf_token']})
                base = f"/api/profiles/{account['profile_id']}/calendar"
                event = client.post(base + '/events', json={'idempotency_key': 'notice', 'title': '复诊', 'scheduled_date': '2000-01-01'}).json()['event']
                url = base + '/events/' + event['event_id']
                acknowledged = client.post(url + '/acknowledge').json()['event']['reminder_acknowledged_at']
                updated = client.patch(url, json={'title': '复诊带上报告'}).json()['event']
                self.assertEqual(acknowledged, updated['reminder_acknowledged_at'])
                self.assertEqual('acknowledged', updated['reminder_status'])
                with TestClient(app) as other:
                    account2 = other.post('/api/auth/register', json={'username': 'notice-other', 'password': 'synthetic-password-123'}).json()
                    other.headers.update({'X-CSRF-Token': account2['csrf_token']})
                    own_url = f"/api/profiles/{account2['profile_id']}/calendar/events/{event['event_id']}"
                    self.assertEqual(404, other.patch(own_url, json={'scheduled_date': '2000-02-01'}).status_code)
                self.assertEqual(acknowledged, client.get(base).json()['events'][0]['reminder_acknowledged_at'])

    def test_history_migration_preserves_old_answers_and_resolved_topic_on_restart(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'history.sqlite3'
            with closing(sqlite3.connect(path)) as db, db:
                db.execute('CREATE TABLE web_answer_history (id INTEGER PRIMARY KEY, profile_id TEXT NOT NULL, mode TEXT NOT NULL, question TEXT NOT NULL, answer TEXT NOT NULL, sources_json TEXT NOT NULL, created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP)')
                db.execute("INSERT INTO web_answer_history (profile_id, mode, question, answer, sources_json) VALUES ('owner', 'knowledge', 'HbA1c是什么？', '已确认的回答', '[]')")
            history = WebAnswerHistory(path)
            history.save('owner', 'knowledge', '这个指标是什么？', '已确认的回答', [], resolved_question='HbA1c是什么？')
            history.save('other', 'knowledge', '其他人的问题', '其他人的回答', [])
            restored = WebAnswerHistory(path).list('owner')
            self.assertEqual(2, len(restored))
            self.assertEqual('这个指标是什么？', restored[0]['question'])
            self.assertEqual('HbA1c是什么？', restored[0]['resolved_question'])
            self.assertEqual(restored[1]['question'], restored[1]['resolved_question'])
