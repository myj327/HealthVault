import unittest

from healthvault.evaluation.ocr.runner import make_gold, score_case, summarize


class OCREvaluationTests(unittest.TestCase):
    def test_gold_is_deterministic_and_patient_disjoint(self):
        first = make_gold(46)
        self.assertEqual(first, make_gold(46))
        self.assertEqual(20, len(first))
        self.assertEqual(100, sum(len(report["fields"]) for report in first))
        self.assertEqual(10, len({report["profile_id"] for report in first}))
        dev = {report["profile_id"] for report in first if report["split"] == "dev"}
        test = {report["profile_id"] for report in first if report["split"] == "test"}
        self.assertFalse(dev & test)
        self.assertEqual({"clean", "compact", "rotated", "faint", "jpeg"},
                         {report["template"] for report in first})

    def test_perfect_case_uses_all_gold_denominators(self):
        report = make_gold(46)[0]
        counts, badcases = score_case(report, report["fields"], [])
        self.assertEqual([], badcases)
        self.assertEqual(5, counts["tp"])
        summary = summarize([{"counts": counts}])
        self.assertEqual({"tp": 5, "fp": 0, "fn": 0, "precision": 1.0,
                          "recall": 1.0, "f1": 1.0}, summary["field_micro"])
        self.assertEqual({"correct": 3, "gold_denominator": 3, "rate": 1.0},
                         summary["value_exact"])
        self.assertEqual(1, summary["date_exact"]["gold_denominator"])
        self.assertEqual(1.0, summary["indicator_field_micro"]["f1"])
        self.assertIsNone(summary["low_confidence_recall"]["rate"])

    def test_missing_and_wrong_values_count_against_gold(self):
        report = make_gold(46)[0]
        extracted = dict(report["fields"])
        del extracted["report_date"]
        del extracted["hba1c"]
        extracted["fasting_glucose"] = {"value": "999", "unit": "mg/dL"}
        blocks = [{"text": "result|fasting_glucose|999|mg/dL|0|1", "confidence": 0.4}]
        counts, badcases = score_case(report, extracted, blocks)
        summary = summarize([{"counts": counts}])
        self.assertEqual(3, summary["field_micro"]["fn"])
        self.assertEqual(1, summary["field_micro"]["fp"])
        self.assertEqual(2, summary["indicator_field_micro"]["fn"])
        self.assertEqual(2, summary["field_key_micro"]["fn"])
        self.assertEqual({"correct": 1, "gold_denominator": 3, "rate": 0.333333},
                         summary["value_exact"])
        self.assertEqual(0, summary["date_exact"]["correct"])
        self.assertEqual(3, summary["correction_burden"]["fields"])
        self.assertEqual(1, summary["low_confidence_recall"]["flagged_corrections"])
        self.assertEqual(3, len(badcases))

    def test_low_confidence_uses_line_position_when_ocr_code_is_corrupt(self):
        report = make_gold(46)[0]
        report["template"] = "clean"
        extracted = dict(report["fields"])
        del extracted["ldl_cholesterol"]
        block = {"text": "result|ldl_cholestero1|2.1|mmol/L|0.0|3.4",
                 "confidence": 0.4,
                 "polygon": [[55, 333], [700, 333], [700, 355], [55, 355]]}
        counts, badcases = score_case(report, extracted, [block])
        self.assertEqual(1, counts["low_confidence_corrections"])
        self.assertEqual(block["text"], badcases[0]["ocr_block"])

    def test_false_positive_is_counted_without_changing_gold_denominator(self):
        report = make_gold(46)[0]
        extracted = {**report["fields"], "extra": "hallucinated"}
        counts, badcases = score_case(report, extracted, [])
        summary = summarize([{"counts": counts}])
        self.assertEqual(1, summary["field_micro"]["fp"])
        self.assertEqual(5, summary["gold_field_count"])
        self.assertEqual("extra", badcases[0]["field"])


if __name__ == "__main__":
    unittest.main()
