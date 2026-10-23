"""Deterministic parsing for the documented synthetic report format."""

from __future__ import annotations

import hashlib
import math
import re
from datetime import date
from typing import Any

from healthvault.application.extract_clinical_items import (
    extract_explicit_clinical_items,
    match_explicit_clinical_line,
)
from healthvault.domain.reports import (
    INDICATOR_ALIASES,
    INDICATOR_CATALOG,
    UNIT_ALIASES,
    Evidence,
    PreviewWarning,
    ReferenceRange,
    ReportIndicator,
    ReportMetadata,
    ReportPreview,
    ReportSource,
    ValidationIssue,
)


HEADER_NAMES = ("report_id", "report_date", "facility")
ISO_DATE_PATTERN = re.compile(r"\d{4}-\d{2}-\d{2}")
OCR_INDICATOR_ALIASES = {"Idl_cholesterol": "ldl_cholesterol"}


def _normalize_ocr_result(line: str) -> str:
    """Repair observed field separators only; never replace characters in values."""
    if line.startswith("result}fasting_glucose|"):
        line = "result|" + line[len("result}"):]
    elif line.startswith("resultIdl_cholesterol|"):
        line = "result|" + line[len("result"):]
    parts = line.split("|")
    if len(parts) == 5 and parts[:2] == ["result", "fasting_glucose"]:
        # One observed row lost the delimiter between its decimal and known unit.
        merged = re.fullmatch(r"([+-]?\d+(?:\.\d+)?)mmol/L", parts[2].strip())
        if merged:
            line = "|".join(parts[:2] + [merged[1], "mmol/L"] + parts[3:])
    return line


def _issue(
    code: str,
    message: str,
    *,
    line: int | None = None,
    indicator_code: str | None = None,
    expected_unit: str | None = None,
    received_unit: str | None = None,
) -> ValidationIssue:
    return ValidationIssue(
        code=code,
        message=message,
        line=line,
        indicator_code=indicator_code,
        expected_unit=expected_unit,
        received_unit=received_unit,
    )


def _parse_finite_number(value: str) -> float | None:
    try:
        parsed = float(value)
    except ValueError:
        return None
    return parsed if math.isfinite(parsed) else None


def _indicator_flag(value: float, lower: float, upper: float) -> str:
    if value < lower:
        return "low"
    if value > upper:
        return "high"
    return "normal"


def build_report_preview(
    source_id: Any,
    report_text: Any,
    *,
    clinical_context: dict[str, Any] | None = None,
) -> ReportPreview:
    """Parse and validate one in-memory synthetic report without persistence."""
    issues: list[ValidationIssue] = []
    valid_source_id = isinstance(source_id, str) and bool(source_id.strip())
    valid_report_text = isinstance(report_text, str) and bool(report_text.strip())

    if not valid_source_id:
        issues.append(
            _issue(
                "invalid_source_id",
                "source_id must be a non-empty string.",
            )
        )
    if not valid_report_text:
        issues.append(
            _issue(
                "invalid_report_text",
                "report_text must be a non-empty string.",
            )
        )

    source_value = source_id if isinstance(source_id, str) else None
    source_hash = (
        hashlib.sha256(report_text.encode("utf-8")).hexdigest()
        if isinstance(report_text, str)
        else None
    )
    empty_metadata = ReportMetadata(None, None, None)
    if not valid_report_text:
        return ReportPreview(
            source=ReportSource(source_value, source_hash),
            report=empty_metadata,
            indicators=[],
            clinical_items=[],
            issues=issues,
        )

    headers: dict[str, str | None] = {name: None for name in HEADER_NAMES}
    header_lines: dict[str, int] = {}
    seen_headers: set[str] = set()
    seen_indicators: set[str] = set()
    indicators: list[ReportIndicator] = []
    warnings: list[PreviewWarning] = []

    for line_number, raw_line in enumerate(report_text.splitlines(), start=1):
        stripped_line = raw_line.strip()
        if not stripped_line or stripped_line.startswith("#"):
            continue

        header_name, separator, header_value = stripped_line.partition(":")
        header_name = header_name.strip()
        if separator and header_name in headers:
            if header_name in seen_headers:
                issues.append(
                    _issue(
                        "duplicate_header",
                        f"Header '{header_name}' must appear exactly once.",
                        line=line_number,
                    )
                )
                continue
            seen_headers.add(header_name)
            header_lines[header_name] = line_number
            value = header_value.strip()
            if not value:
                issues.append(
                    _issue(
                        "empty_header",
                        f"Header '{header_name}' must not be empty.",
                        line=line_number,
                    )
                )
                continue
            headers[header_name] = value
            continue

        if match_explicit_clinical_line(raw_line) is not None:
            continue

        normalized_line = _normalize_ocr_result(stripped_line)
        if not normalized_line.startswith("result|"):
            issues.append(
                _issue(
                    "unrecognized_line",
                    "Line does not match a supported report header or result.",
                    line=line_number,
                )
            )
            continue

        parts = [part.strip() for part in normalized_line.split("|")]
        if len(parts) != 6:
            issues.append(
                _issue(
                    "malformed_result",
                    "Result line must contain exactly six pipe-delimited fields.",
                    line=line_number,
                )
            )
            continue

        _prefix, input_code, value_text, input_unit, lower_text, upper_text = parts
        normalized_code = input_code.casefold()
        normalized_code = OCR_INDICATOR_ALIASES.get(input_code, normalized_code)
        code = INDICATOR_ALIASES.get(normalized_code, normalized_code)
        unit = UNIT_ALIASES.get(input_unit, input_unit)
        if code in seen_indicators:
            issues.append(
                _issue(
                    "duplicate_indicator",
                    f"Indicator '{code}' must appear at most once.",
                    line=line_number,
                    indicator_code=code,
                )
            )
            continue
        seen_indicators.add(code)

        spec = INDICATOR_CATALOG.get(code)
        if spec is None:
            warnings.append(
                PreviewWarning(
                    "unsupported_indicator",
                    f"Indicator '{input_code}' needs manual mapping and is not saved as structured data.",
                    line_number,
                    input_code,
                    source_value,
                    raw_line,
                )
            )
            continue

        line_has_error = False
        if unit != spec.unit:
            issues.append(
                _issue(
                    "unit_mismatch",
                    f"Indicator '{code}' requires unit '{spec.unit}', received '{unit}'.",
                    line=line_number,
                    indicator_code=code,
                    expected_unit=spec.unit,
                    received_unit=input_unit,
                )
            )
            line_has_error = True

        value = _parse_finite_number(value_text)
        lower = _parse_finite_number(lower_text)
        upper = _parse_finite_number(upper_text)
        if value is None or lower is None or upper is None:
            issues.append(
                _issue(
                    "invalid_number",
                    "Result value and reference bounds must be finite numbers.",
                    line=line_number,
                    indicator_code=code,
                )
            )
            line_has_error = True

        if lower is not None and upper is not None and lower > upper:
            issues.append(
                _issue(
                    "invalid_reference_range",
                    "Reference lower bound must not exceed the upper bound.",
                    line=line_number,
                    indicator_code=code,
                )
            )
            line_has_error = True

        if line_has_error:
            continue
        assert value is not None and lower is not None and upper is not None
        indicators.append(
            ReportIndicator(
                code=code,
                display_name=spec.display_name,
                value=value,
                unit=unit,
                reference_range=ReferenceRange(lower, upper),
                flag=_indicator_flag(value, lower, upper),
                evidence=Evidence(
                    source_id=source_value,
                    line_start=line_number,
                    line_end=line_number,
                    quote=raw_line,
                ),
            )
        )

    for header_name in HEADER_NAMES:
        if header_name not in seen_headers:
            issues.append(
                _issue(
                    "missing_header",
                    f"Required header '{header_name}' is missing.",
                )
            )

    report_date = headers["report_date"]
    valid_base_date = report_date
    if report_date is not None:
        try:
            if not ISO_DATE_PATTERN.fullmatch(report_date):
                raise ValueError
            date.fromisoformat(report_date)
        except ValueError:
            valid_base_date = None
            issues.append(
                _issue(
                    "invalid_report_date",
                    "report_date must be a valid ISO YYYY-MM-DD date.",
                    line=header_lines["report_date"],
                )
            )

    return ReportPreview(
        source=ReportSource(source_value, source_hash),
        report=ReportMetadata(
            report_id=headers["report_id"],
            report_date=report_date,
            facility=headers["facility"],
        ),
        indicators=indicators,
        clinical_items=extract_explicit_clinical_items(
            source_value,
            source_hash,
            report_text,
            report_date=valid_base_date,
            clinical_context=clinical_context,
        ),
        issues=issues,
        warnings=warnings,
    )
