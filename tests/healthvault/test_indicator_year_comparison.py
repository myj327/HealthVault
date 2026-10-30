import asyncio
import unittest

from healthvault.application.compare_health_indicators import compare_health_indicators
from healthvault.application.tool_evidence import merge_tool_evidence, tool_evidence
from healthvault.adapters.bearcode.tools import BearCodeToolAdapter
from healthvault.runtime_context import RuntimeContext
from healthvault.tools.catalog import build_healthvault_tool_registry
from tests.healthvault.test_compare_health_indicators import RecordingRepository, make_indicator, make_report


def report(identifier, date, value, unit='mmol/L'):
    return make_report(identifier, date, [make_indicator('ldl_cholesterol', value, unit=unit, source_id=identifier)])


class IndicatorYearComparisonTests(unittest.TestCase):
    def test_requested_internal_years_choose_last_in_each_year_and_keep_sources(self):
        repository = RecordingRepository([
            report('newest', '2026-01-01', 1), report('first', '2020-01-01', 9),
            report('2022-late', '2022-12-31', 4), report('2024', '2024-12-31', 3),
            report('2022-early', '2022-01-01', 8), report('middle', '2023-12-31', 5),
        ])
        result = compare_health_indicators({'profile_id': 'test', 'start_year': 2022, 'end_year': 2024}, repository).to_dict()
        item = result['comparisons'][0]
        self.assertEqual(('2022-late', '2024'), (item['earliest']['report_id'], item['latest']['report_id']))
        self.assertEqual(-1, item['absolute_change'])
        self.assertEqual(-25, item['relative_change_percent'])
        self.assertEqual('fluctuating', item['trend']['direction'])
        self.assertEqual([4, 5, 3], [point['value'] for point in item['trend']['annual_points']])
        sources = merge_tool_evidence([], tool_evidence('compare_health_indicators', result))
        self.assertTrue(set(source['citation_id'] for source in sources).issubset(result['available_citation_ids']))
        trend_source = next(source for source in sources if source['citation_id'] == item['trend']['citation_id'])
        self.assertEqual(['2022-late', 'middle', '2024'], [link['report_id'] for link in trend_source['report_links']])
        other = compare_health_indicators({'profile_id': 'test', 'start_year': 2023, 'end_year': 2024}, repository).to_dict()
        self.assertNotEqual(item['comparison_citation_id'], other['comparisons'][0]['comparison_citation_id'])
        merge_tool_evidence(sources, tool_evidence('compare_health_indicators', other))

    def test_missing_year_invalid_range_and_unit_mismatch_never_substitute(self):
        repository = RecordingRepository([report('2020', '2020-01-01', 2), report('2024', '2024-01-01', 3)])
        for years in ({'start_year': 2021}, {'start_year': True, 'end_year': 2024}, {'start_year': 2024, 'end_year': 2020}):
            result = compare_health_indicators({'profile_id': 'test', **years}, repository)
            self.assertEqual('invalid_request', result.status)
        self.assertEqual([], repository.queries)
        missing = compare_health_indicators({'profile_id': 'test', 'start_year': 2021, 'end_year': 2024}, repository)
        self.assertEqual((), missing.comparisons)
        self.assertEqual('missing_year', missing.issues[0].code)
        self.assertIn('2021', missing.issues[0].message)
        repository.timeline = [report('2020', '2020-01-01', 2), report('2024', '2024-01-01', 30, 'mg/dL')]
        mismatch = compare_health_indicators({'profile_id': 'test', 'start_year': 2020, 'end_year': 2024}, repository)
        self.assertEqual((), mismatch.comparisons)
        self.assertEqual('unit_mismatch', mismatch.issues[0].code)

    def test_slope_uses_actual_time_gaps_and_zero_baseline_has_no_relative_percent(self):
        repository = RecordingRepository([report('2020', '2020-01-01', 0), report('2022', '2022-01-01', 4), report('2024', '2024-01-01', 8)])
        item = compare_health_indicators({'profile_id': 'test'}, repository).to_dict()['comparisons'][0]
        self.assertIsNone(item['relative_change_percent'])
        self.assertEqual('increased', item['trend']['direction'])
        self.assertAlmostEqual(2, item['trend']['slope_per_year'], places=3)
        self.assertEqual([2020, 2022, 2024], [point['year'] for point in item['trend']['annual_points']])
        repository.timeline = [report('a', '2024-01-01', 5), report('b', '2024-12-31', 4)]
        trend = compare_health_indicators({'profile_id': 'test'}, repository).comparisons[0].trend
        self.assertEqual('insufficient_data', trend['direction'])
        self.assertIsNone(trend['slope_per_year'])

    def test_agent_tool_passes_years_and_keeps_profile_bound_by_server(self):
        repository = RecordingRepository([report('2022', '2022-01-01', 4), report('2024', '2024-01-01', 3)])
        adapter = BearCodeToolAdapter(build_healthvault_tool_registry(record_repository=repository),
            allowed_domain_tools={'compare_health_indicators'}, runtime_context=RuntimeContext('bound-profile', 'test'))
        definition = adapter.combined_definitions[0]
        self.assertNotIn('profile_id', definition['input_schema']['properties'])
        import json
        result = json.loads(asyncio.run(adapter.execute('compare_health_indicators',
            {'start_year': 2022, 'end_year': 2024, 'indicator_code': 'ldl_cholesterol'})))
        self.assertEqual('ok', result['status'])
        self.assertEqual(["bound-profile"], repository.queries)
        self.assertEqual(-1, result['comparisons'][0]['absolute_change'])

    def test_web_year_filters_reach_comparison_and_still_require_profile_ownership(self):
        import os
        import tempfile
        from pathlib import Path
        from unittest.mock import patch
        from fastapi.testclient import TestClient
        from healthvault.ui.controller import HealthVaultUIController

        with tempfile.TemporaryDirectory() as directory, patch.dict(os.environ, {
            'HEALTHVAULT_KNOWLEDGE_MODE': 'lexical',
            'HEALTHVAULT_UI_DB_PATH': str(Path(directory) / 'bootstrap.sqlite3'),
        }):
            from healthvault.web.api import create_app
            controller = HealthVaultUIController(Path(directory) / 'records.sqlite3')
            controller.record_repository = RecordingRepository([
                report('2020', '2020-01-01', 9), report('2022', '2022-01-01', 4), report('2024', '2024-01-01', 3)])
            controller.record_repository.database_path = controller.database_path
            client = TestClient(create_app(controller))
            try:
                user = client.app.state.auth_store.register('year-test', 'synthetic-password-123', user_id='year-test')
                token, _ = client.app.state.auth_store.create_session(user)
                client.cookies.set('hv_session', token)
                response = client.get('/api/profiles/year-test/comparisons?start_year=2022&end_year=2024')
                self.assertEqual(200, response.status_code, response.text)
                self.assertEqual(-1, response.json()['comparisons'][0]['absolute_change'])
                self.assertEqual('latest_in_each_year', response.json()['year_selection']['method'])
                self.assertEqual(422, client.get('/api/profiles/year-test/comparisons?start_year=2022').status_code)
                self.assertEqual(404, client.get('/api/profiles/another-profile/comparisons?start_year=2022&end_year=2024').status_code)
            finally:
                client.close()
