import json
import unittest
from pathlib import Path

from healthvault.web.extraction import build_natural_preview


class NaturalReportCoverageTests(unittest.TestCase):
    def test_synthetic_templates_and_safe_refusals(self):
        pack = json.loads((Path(__file__).parents[1]/'fixtures/health_reports/natural-coverage-v1.json').read_text(encoding='utf-8'))
        for case in pack['cases']:
            with self.subTest(case=case['id']):
                preview = build_natural_preview(case['id'],case['text'],{})
                self.assertEqual(case['date'],preview['report']['report_date'])
                self.assertEqual(case['can_confirm'],preview['can_confirm'])
                self.assertEqual(case['indicators'],[[item['code'],item['value']] for item in preview['indicators']])
                self.assertEqual(case['findings'],[[item['finding_type'],item['body_site'],item['laterality'],item['dimensions_mm'],item['original_unit']] for item in preview['imaging_findings']])
                for item in [*preview['indicators'],*preview['imaging_findings']]:
                    evidence = item['evidence']
                    self.assertEqual(evidence['quote'],case['text'][evidence['span_start']:evidence['span_end']])
                if not case['can_confirm']:
                    self.assertTrue(preview['other_findings'])

if __name__ == '__main__': unittest.main()
