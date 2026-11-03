import unittest

from healthvault.application.compare_health_indicators import compare_health_indicators
from healthvault.application.query_health_timeline import query_health_timeline
from healthvault.domain.reports import INDICATOR_CATALOG
from healthvault.tools.directory import TOOL_SPECS
from tests.healthvault.test_compare_health_indicators import RecordingRepository, make_indicator, make_report


class ToolInputTests(unittest.TestCase):
    def repository(self, codes):
        return RecordingRepository([
            make_report('first', '2021-04-12', [make_indicator(code, 4.2) for code in codes]),
            make_report('last', '2025-08-21', [make_indicator(code, 4.8) for code in codes]),
        ])

    def test_unambiguous_aliases_return_the_same_records_and_comparison(self):
        aliases = {'ldl_c': 'ldl_cholesterol', 'low_density_lipoprotein': 'ldl_cholesterol',
                   'hdl_c': 'hdl_cholesterol', 'high_density_lipoprotein': 'hdl_cholesterol',
                   'triglyceride': 'triglycerides', 'fasting_blood_glucose': 'fasting_glucose'}
        for alias, code in aliases.items():
            with self.subTest(alias=alias):
                repo = self.repository(set(aliases.values()))
                params = {'profile_id': 'test', 'indicator_code': alias}
                timeline = query_health_timeline(params, repo)
                comparison = compare_health_indicators(params, repo)
                self.assertEqual(('ok', code, 2), (timeline.status, timeline.indicator_code, timeline.indicator_count))
                self.assertEqual('ok', comparison.status)
                self.assertEqual(code, comparison.comparisons[0].code)
                self.assertEqual(0.6, comparison.to_dict()['comparisons'][0]['absolute_change'])

    def test_ambiguous_glucose_requires_lookup_instead_of_assuming_fasting(self):
        for name in ('glucose', '血糖'):
            repo = self.repository(('fasting_glucose', 'random_glucose'))
            params = {'profile_id': 'test', 'indicator_code': name}
            timeline = query_health_timeline(params, repo)
            comparison = compare_health_indicators(params, repo)
            self.assertEqual('invalid_request', timeline.status)
            self.assertEqual('invalid_request', comparison.status)
            self.assertEqual(name, comparison.issues[0].indicator_code)
            self.assertIn('without indicator_code', comparison.issues[0].message)
            self.assertEqual([], repo.queries)
            all_records = query_health_timeline({'profile_id': 'test'}, repo)
            self.assertEqual({'fasting_glucose', 'random_glucose'},
                             {item.code for report in all_records.timeline for item in report.indicators})

    def test_every_advertised_code_is_accepted_by_both_tools(self):
        for spec in TOOL_SPECS:
            if spec.name not in {'query_health_timeline', 'compare_health_indicators'}:
                continue
            schema = spec.definition()['input_schema']['properties']['indicator_code']
            self.assertIn('enum', schema)
            self.assertNotIn('glucose', schema['enum'])
            self.assertNotIn('indicator_code', spec.input_schema['required'])
            for code in schema['enum']:
                self.assertIn(code, INDICATOR_CATALOG)
                params = {'profile_id': 'test', 'indicator_code': code}
                self.assertNotEqual('invalid_request', query_health_timeline(params, RecordingRepository([])).status)
                self.assertNotEqual('invalid_request', compare_health_indicators(params, RecordingRepository([])).status)


if __name__ == '__main__':
    unittest.main()
