"""Extract only explicitly labelled clinical statements from report text."""

from __future__ import annotations

import calendar
import hashlib
import re
from dataclasses import replace
from datetime import date, timedelta
from typing import Any

from healthvault.domain.reports import ClinicalEvidence, ClinicalItemCandidate


LABEL_PATTERNS = (
    ("diagnosis", re.compile(r"^\s*(?:诊断|临床诊断|出院诊断|诊断意见|diagnosis)\s*[:：]\s*(?P<value>\S(?:.*\S)?)\s*$", re.IGNORECASE)),
    ("provider_advice", re.compile(r"^\s*(?:医生建议|医嘱|建议|检查建议|处理意见|doctor(?:'s)?\s+advice|provider\s+advice|medical\s+advice)\s*[:：]\s*(?P<value>\S(?:.*\S)?)\s*$", re.IGNORECASE)),
    ("visit_type", re.compile(r"^\s*(?:就诊类型|visit\s+type)\s*[:：]\s*(?P<value>\S(?:.*\S)?)\s*$", re.IGNORECASE)),
    ("follow_up_plan", re.compile(r"^\s*(?:复诊计划|复查计划|随访计划|复诊|复查|随访|follow[\s-]*up\s+plan)\s*[:：]\s*(?P<value>\S(?:.*\S)?)\s*$", re.IGNORECASE)),
)
ISO_DATE = re.compile(r"(?<!\d)(\d{4})-(\d{1,2})-(\d{1,2})(?!\d)")
CHINESE_DATE = re.compile(r"(?<!\d)(\d{4})年(\d{1,2})月(\d{1,2})日")
CHINESE_RELATIVE = re.compile(r"(\d+)\s*个?\s*(天|周|月)后")
ENGLISH_RELATIVE = re.compile(r"\bin\s+(\d+)\s+(day|week|month)s?\b", re.IGNORECASE)
DIAGNOSIS_CODE = re.compile(
    r"[（(\[](?P<code>[A-Z][0-9][0-9A-Z](?:\.[0-9A-Z]+)?)[）)\]]\s*$",
    re.IGNORECASE,
)
VISIT_TYPE_VALUES = {
    "初诊": "initial_visit",
    "复诊": "follow_up_visit",
    "门诊": "outpatient",
    "住院": "inpatient",
    "initial": "initial_visit",
    "initial visit": "initial_visit",
    "follow-up": "follow_up_visit",
    "follow up": "follow_up_visit",
    "follow-up visit": "follow_up_visit",
    "follow up visit": "follow_up_visit",
    "outpatient": "outpatient",
    "inpatient": "inpatient",
}


def match_explicit_clinical_line(line: str) -> tuple[str, re.Match[str]] | None:
    """Return a supported kind only when a complete line has an explicit label."""
    for kind, pattern in LABEL_PATTERNS:
        matched = pattern.fullmatch(line)
        if matched is not None:
            return kind, matched
    return None


def _add_months(base: date, months: int) -> date:
    month_index = base.month - 1 + months
    year = base.year + month_index // 12
    month = month_index % 12 + 1
    day = min(base.day, calendar.monthrange(year, month)[1])
    return date(year, month, day)


def _follow_up_date(
    value: str, base_date: date | None
) -> tuple[str, str | None, str]:
    recognized_expressions = {
        matched.group(0).lower()
        for pattern in (
            ISO_DATE,
            CHINESE_DATE,
            CHINESE_RELATIVE,
            ENGLISH_RELATIVE,
        )
        for matched in pattern.finditer(value)
    }
    if len(recognized_expressions) > 1:
        return "needs_date_confirmation", None, value

    for pattern in (ISO_DATE, CHINESE_DATE):
        matched = pattern.search(value)
        if matched is not None:
            original = matched.group(0)
            try:
                normalized = date(*(int(part) for part in matched.groups())).isoformat()
            except ValueError:
                return "needs_date_confirmation", None, original
            return "normalized", normalized, original

    matched = CHINESE_RELATIVE.search(value)
    if matched is not None:
        amount = int(matched.group(1))
        unit = matched.group(2)
        original = matched.group(0)
        if base_date is None:
            return "needs_date_confirmation", None, original
        if unit == "天":
            normalized = base_date + timedelta(days=amount)
        elif unit == "周":
            normalized = base_date + timedelta(weeks=amount)
        else:
            normalized = _add_months(base_date, amount)
        return "normalized", normalized.isoformat(), original

    matched = ENGLISH_RELATIVE.search(value)
    if matched is not None:
        amount = int(matched.group(1))
        unit = matched.group(2).lower()
        original = matched.group(0)
        if base_date is None:
            return "needs_date_confirmation", None, original
        if unit == "day":
            normalized = base_date + timedelta(days=amount)
        elif unit == "week":
            normalized = base_date + timedelta(weeks=amount)
        else:
            normalized = _add_months(base_date, amount)
        return "normalized", normalized.isoformat(), original

    return "needs_date_confirmation", None, value


def _ocr_block_for_line(
    raw_line: str,
    line_number: int,
    clinical_context: dict[str, Any] | None,
) -> dict[str, Any] | None:
    if not clinical_context or clinical_context.get("source_type") != "ocr":
        return None
    blocks = clinical_context.get("ocr_blocks")
    if not isinstance(blocks, list) or line_number > len(blocks):
        return None
    block = blocks[line_number - 1]
    if not isinstance(block, dict) or block.get("text") != raw_line.strip():
        return None
    return block


def extract_explicit_clinical_items(
    source_id: str | None,
    source_sha256: str | None,
    report_text: str,
    *,
    report_date: str | None,
    clinical_context: dict[str, Any] | None = None,
) -> list[ClinicalItemCandidate]:
    """Extract explicit labels without deriving facts from indicators or prose."""
    try:
        base_date = date.fromisoformat(report_date) if report_date else None
    except ValueError:
        base_date = None
    source_type = (
        clinical_context.get("source_type", "text")
        if isinstance(clinical_context, dict)
        else "text"
    )
    items: list[ClinicalItemCandidate] = []
    offset = 0
    for line_number, line_with_ending in enumerate(
        report_text.splitlines(keepends=True), start=1
    ):
        raw_line = line_with_ending.rstrip("\r\n")
        if raw_line.strip().startswith("#"):
            offset += len(line_with_ending)
            continue
        explicit = match_explicit_clinical_line(raw_line)
        if explicit is None:
            offset += len(line_with_ending)
            continue
        kind, matched = explicit
        original_text = matched.group("value")
        span_start = offset + matched.start("value")
        span_end = offset + matched.end("value")
        diagnosis_code: str | None = None
        normalized_value: str | None = None
        date_status: str | None = None
        normalized_date: str | None = None
        date_original: str | None = None
        if kind == "diagnosis":
            code_match = DIAGNOSIS_CODE.search(original_text)
            if code_match is not None:
                diagnosis_code = code_match.group("code").upper()
                normalized_value = original_text[: code_match.start()].strip()
        elif kind == "visit_type":
            normalized_value = VISIT_TYPE_VALUES.get(original_text.strip().lower())
        elif kind == "follow_up_plan":
            try:
                date_status, normalized_date, date_original = _follow_up_date(original_text, base_date)
            except (ValueError, OverflowError):
                date_status, normalized_date, date_original = "needs_date_confirmation", None, original_text
            normalized_value = normalized_date

        block = _ocr_block_for_line(raw_line, line_number, clinical_context)
        confidence = float(block["confidence"]) if block is not None else None
        identity = "\0".join(
            (
                source_id or "",
                source_sha256 or "",
                kind,
                str(line_number),
                str(span_start),
                original_text,
            )
        )
        items.append(
            ClinicalItemCandidate(
                clinical_item_id=hashlib.sha256(identity.encode("utf-8")).hexdigest(),
                kind=kind,
                original_text=original_text,
                diagnosis_code=diagnosis_code,
                normalized_value=normalized_value,
                evidence=ClinicalEvidence(
                    source_id=source_id,
                    line_start=line_number,
                    line_end=line_number,
                    span_start=span_start,
                    span_end=span_end,
                    quote=raw_line,
                    page=block.get("page") if block is not None else None,
                    polygon=block.get("polygon") if block is not None else None,
                ),
                confidence=confidence,
                source_type=source_type,
                recorded_by_provider=True,
                confirmation_status="pending_confirmation",
                date_status=date_status,
                normalized_date=normalized_date,
                date_original=date_original,
            )
        )
        if kind == "provider_advice" and re.search(r"复查|复诊|随访|follow[\s-]*up", original_text, re.I):
            # The report explicitly asks for follow-up; retain the same exact evidence.
            try:
                status, normalized, expression = _follow_up_date(original_text, base_date)
            except (ValueError, OverflowError):
                status, normalized, expression = "needs_date_confirmation", None, original_text
            items.append(replace(items[-1],
                clinical_item_id=hashlib.sha256((identity + "\0follow_up_plan").encode("utf-8")).hexdigest(),
                kind="follow_up_plan", normalized_value=normalized, date_status=status,
                normalized_date=normalized, date_original=expression,
            ))
        offset += len(line_with_ending)
    return items
