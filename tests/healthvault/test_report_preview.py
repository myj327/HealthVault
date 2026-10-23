import hashlib
import unittest
from pathlib import Path
from unittest.mock import patch

from healthvault.application.report_preview import build_report_preview


FIXTURE_DIR = Path(__file__).parents[1] / "fixtures" / "health_reports"


def load_fixture(name):
    return (FIXTURE_DIR / name).read_text(encoding="utf-8")


class ReportPreviewTests(unittest.TestCase):
    def test_two_year_fixtures_are_ready_with_hash_flags_and_evidence(self):
        cases = (
            (
                "synthetic_checkup_2024.txt",
                "synthetic-2024",
                "HV-SYN-2024-001",
                "2024-05-20",
                {
                    "fasting_glucose": (5.4, "mmol/L", 3.9, 6.1, "normal"),
                    "hba1c": (5.5, "%", 4.0, 6.0, "normal"),
                    "ldl_cholesterol": (3.1, "mmol/L", 0.0, 3.4, "normal"),
                },
            ),
            (
                "synthetic_checkup_2025.txt",
                "synthetic-2025",
                "HV-SYN-2025-001",
                "2025-06-18",
                {
                    "fasting_glucose": (6.5, "mmol/L", 3.9, 6.1, "high"),
                    "hba1c": (5.9, "%", 4.0, 6.0, "normal"),
                    "ldl_cholesterol": (2.8, "mmol/L", 0.0, 3.4, "normal"),
                },
            ),
        )

        for filename, source_id, report_id, report_date, expected in cases:
            with self.subTest(filename=filename):
                report_text = load_fixture(filename)
                raw_lines = report_text.splitlines()

                with patch(
                    "builtins.open",
                    side_effect=AssertionError("preview must not perform file I/O"),
                ):
                    result = build_report_preview(source_id, report_text).to_dict()

                self.assertEqual("ready", result["status"])
                self.assertTrue(result["can_confirm"])
                self.assertEqual([], result["issues"])
                self.assertEqual(
                    hashlib.sha256(report_text.encode("utf-8")).hexdigest(),
                    result["source"]["sha256"],
                )
                self.assertEqual(source_id, result["source"]["source_id"])
                self.assertEqual(report_id, result["report"]["report_id"])
                self.assertEqual(report_date, result["report"]["report_date"])
                self.assertEqual(
                    "Synthetic Community Clinic",
                    result["report"]["facility"],
                )
                self.assertEqual(3, len(result["indicators"]))

                by_code = {
                    indicator["code"]: indicator
                    for indicator in result["indicators"]
                }
                self.assertEqual(
                    {"fasting_glucose", "hba1c", "ldl_cholesterol"},
                    set(by_code),
                )
                self.assertEqual("空腹血糖", by_code["fasting_glucose"]["display_name"])

                for code, (value, unit, lower, upper, flag) in expected.items():
                    indicator = by_code[code]
                    self.assertEqual(value, indicator["value"])
                    self.assertEqual(unit, indicator["unit"])
                    self.assertEqual(
                        {"lower": lower, "upper": upper},
                        indicator["reference_range"],
                    )
                    self.assertEqual(flag, indicator["flag"])

                for line_number, code in enumerate(by_code, start=5):
                    evidence = by_code[code]["evidence"]
                    self.assertEqual(source_id, evidence["source_id"])
                    self.assertEqual(line_number, evidence["line_start"])
                    self.assertEqual(line_number, evidence["line_end"])
                    self.assertEqual(raw_lines[line_number - 1], evidence["quote"])

    def test_boundary_values_are_normal_and_lower_values_are_low(self):
        report_text = "\n".join(
            (
                "# Synthetic test data only; no real person",
                "report_id: HV-SYN-BOUNDARY-001",
                "report_date: 2025-01-01",
                "facility: Synthetic Boundary Clinic",
                "result|fasting_glucose|3.9|mmol/L|3.9|6.1",
                "result|hba1c|6.0|%|4.0|6.0",
                "result|ldl_cholesterol|-0.1|mmol/L|0.0|3.4",
            )
        )

        result = build_report_preview("synthetic-boundary", report_text).to_dict()
        flags = {item["code"]: item["flag"] for item in result["indicators"]}

        self.assertEqual("ready", result["status"])
        self.assertEqual("normal", flags["fasting_glucose"])
        self.assertEqual("normal", flags["hba1c"])
        self.assertEqual("low", flags["ldl_cholesterol"])

    def test_unit_mismatch_is_invalid_and_is_not_converted(self):
        report_text = load_fixture("synthetic_invalid_unit.txt")

        result = build_report_preview("synthetic-unit", report_text).to_dict()

        self.assertEqual("invalid", result["status"])
        self.assertFalse(result["can_confirm"])
        issue = next(issue for issue in result["issues"] if issue["code"] == "unit_mismatch")
        self.assertEqual("fasting_glucose", issue["indicator_code"])
        self.assertEqual("mmol/L", issue["expected_unit"])
        self.assertEqual("mg/dL", issue["received_unit"])
        self.assertEqual(5, issue["line"])
        self.assertNotIn(
            "fasting_glucose",
            {indicator["code"] for indicator in result["indicators"]},
        )
        self.assertEqual(2, len(result["indicators"]))

    def test_malformed_fixture_returns_date_and_result_issues(self):
        result = build_report_preview(
            "synthetic-malformed",
            load_fixture("synthetic_malformed.txt"),
        ).to_dict()
        issues = {issue["code"]: issue for issue in result["issues"]}

        self.assertEqual("invalid", result["status"])
        self.assertFalse(result["can_confirm"])
        self.assertEqual(3, issues["invalid_report_date"]["line"])
        self.assertEqual(5, issues["malformed_result"]["line"])

    def test_missing_empty_and_non_string_inputs_are_invalid(self):
        cases = (
            (None, "valid text", "invalid_source_id"),
            ("  ", "valid text", "invalid_source_id"),
            ("source", None, "invalid_report_text"),
            ("source", 42, "invalid_report_text"),
            ("source", "  ", "invalid_report_text"),
        )

        for source_id, report_text, expected_code in cases:
            with self.subTest(source_id=source_id, report_text=report_text):
                result = build_report_preview(source_id, report_text).to_dict()
                self.assertEqual("invalid", result["status"])
                self.assertFalse(result["can_confirm"])
                self.assertIn(expected_code, {issue["code"] for issue in result["issues"]})

    def test_header_indicator_number_duplicate_and_unknown_validation(self):
        report_text = "\n".join(
            (
                "# Synthetic test data only; no real person",
                "report_id: HV-SYN-ERRORS-001",
                "report_date: 2025-01-01",
                "result|fasting_glucose|NaN|mmol/L|3.9|6.1",
                "result|hba1c|5.5|%|6.0|4.0",
                "result|unknown_indicator|1.0|mmol/L|0.0|2.0",
                "result|ldl_cholesterol|2.0|mmol/L|0.0|3.4",
                "result|ldl_cholesterol|2.1|mmol/L|0.0|3.4",
                "unexpected content",
            )
        )

        result = build_report_preview("synthetic-errors", report_text).to_dict()
        codes = {issue["code"] for issue in result["issues"]}

        self.assertEqual("invalid", result["status"])
        self.assertFalse(result["can_confirm"])
        self.assertTrue(
            {
                "missing_header",
                "invalid_number",
                "invalid_reference_range",
                "duplicate_indicator",
                "unrecognized_line",
            }.issubset(codes)
        )
        self.assertEqual("unsupported_indicator", result["warnings"][0]["code"])
        self.assertEqual(6, result["warnings"][0]["line"])
        self.assertEqual("pending_mapping", result["warnings"][0]["status"])

    def test_common_checkup_panels_keep_canonical_units_and_source_evidence(self):
        rows = (
            ("SBP", 124, "mmHg", "systolic_bp"),
            ("body_temperature", 36.6, "℃", "body_temperature"),
            ("rbc", 4.6, "×10^12/L", "red_blood_cells"),
            ("neutrophil_percentage", 52, "%", "neutrophil_percentage"),
            ("alp", 72, "U/L", "alkaline_phosphatase"),
            ("tbil", 12, "μmol/L", "total_bilirubin"),
            ("urea", 5, "mmol/L", "urea"),
            ("sodium", 141, "mmol/L", "sodium"),
            ("ft4", 15, "pmol/L", "ft4"),
            ("crp", 7, "mg/L", "crp"),
        )
        report_text = "\n".join((
            "report_id: HV-SYN-CHECKUP-001",
            "report_date: 2026-06-01",
            "facility: Synthetic Lab",
            *(f"result|{code}|{value}|{unit}|0|{5 if code == 'crp' else 300}" for code, value, unit, _ in rows),
        ))

        result = build_report_preview("synthetic-checkup", report_text).to_dict()

        self.assertTrue(result["can_confirm"])
        self.assertEqual([], result["warnings"])
        self.assertEqual([item[3] for item in rows], [item["code"] for item in result["indicators"]])
        self.assertEqual("°C", result["indicators"][1]["unit"])
        self.assertEqual("10^12/L", result["indicators"][2]["unit"])
        self.assertEqual("high", result["indicators"][-1]["flag"])
        self.assertEqual(
            report_text.splitlines()[3:],
            [item["evidence"]["quote"] for item in result["indicators"]],
        )

    def test_new_indicator_rejects_unverified_unit(self):
        report_text = "\n".join((
            "report_id: HV-SYN-UNIT-002",
            "report_date: 2026-06-01",
            "facility: Synthetic Lab",
            "result|sodium|140|mg/dL|130|150",
        ))

        result = build_report_preview("synthetic-bad-unit", report_text).to_dict()

        self.assertFalse(result["can_confirm"])
        self.assertEqual("unit_mismatch", result["issues"][0]["code"])
        self.assertEqual("mmol/L", result["issues"][0]["expected_unit"])

    def test_expanded_indicators_keep_report_ranges_and_normalize_aliases(self):
        entries = (
            ("tc", "4.5", "mmol/L", "total_cholesterol"),
            ("hdl", "1.3", "mmol/L", "hdl_cholesterol"),
            ("tg", "1.1", "mmol/L", "triglycerides"),
            ("wbc", "6.2", "10^9/L", "white_blood_cells"),
            ("hb", "135", "g/L", "hemoglobin"),
            ("plt", "220", "10^9/L", "platelets"),
            ("scr", "78", "umol/L", "creatinine"),
            ("egfr", "90", "mL/min/1.73m2", "egfr"),
            ("alt", "25", "U/L", "alt"),
            ("ast", "23", "U/L", "ast"),
            ("tsh", "2.1", "mIU/L", "tsh"),
            ("ferritin", "80", "ng/mL", "ferritin"),
        )
        report_text = "\n".join(
            (
                "report_id: HV-SYN-EXPANDED-001",
                "report_date: 2026-01-02",
                "facility: Synthetic Lab",
                *(
                    f"result|{code}|{value}|{unit}|0|300"
                    for code, value, unit, _canonical in entries
                ),
            )
        )

        result = build_report_preview("synthetic-expanded", report_text).to_dict()

        self.assertEqual("ready", result["status"])
        self.assertEqual([], result["issues"])
        self.assertEqual([], result["warnings"])
        self.assertEqual(
            [canonical for _code, _value, _unit, canonical in entries],
            [item["code"] for item in result["indicators"]],
        )
        for raw, item in zip(report_text.splitlines()[3:], result["indicators"]):
            self.assertEqual(raw, item["evidence"]["quote"])

    def test_alias_and_canonical_code_are_detected_as_duplicate(self):
        report_text = "\n".join(
            (
                "report_id: HV-SYN-DUPLICATE-001",
                "report_date: 2026-01-02",
                "facility: Synthetic Lab",
                "result|hdl|1.3|mmol/L|0|3",
                "result|hdl_cholesterol|1.4|mmol/L|0|3",
            )
        )

        result = build_report_preview("synthetic-duplicate", report_text).to_dict()

        self.assertEqual("invalid", result["status"])
        self.assertIn("duplicate_indicator", {issue["code"] for issue in result["issues"]})

    def test_equivalent_unit_spelling_is_canonicalized_without_numeric_conversion(self):
        report_text = "\n".join(
            (
                "report_id: HV-SYN-UNIT-ALIAS-001",
                "report_date: 2026-01-02",
                "facility: Synthetic Lab",
                "result|SCr|78|μmol/L|50|100",
            )
        )

        result = build_report_preview("synthetic-unit-alias", report_text).to_dict()

        self.assertEqual("ready", result["status"])
        self.assertEqual(78, result["indicators"][0]["value"])
        self.assertEqual("umol/L", result["indicators"][0]["unit"])
        self.assertIn("μmol/L", result["indicators"][0]["evidence"]["quote"])


if __name__ == "__main__":
    unittest.main()
