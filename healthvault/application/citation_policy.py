"""Shared citation-boundary rules for read-only HealthVault responses."""

from __future__ import annotations

from typing import Any, Mapping


RECORD_PREFIX = "REC:"
KNOWLEDGE_PREFIX = "HVK:"
OFFICIAL_WEB_PREFIX = "HVW:"

CITATION_RESPONSE_POLICY = """\
- Cite only IDs listed in a tool result's top-level available_citation_ids; never invent an ID or copy an ID merely from nested text.
- Remove duplicate IDs and keep the smallest set that directly supports the requested result.
- Use REC:* only for confirmed-record evidence, HVK:* only for local knowledge evidence, and HVW:* only for approved official web evidence. Do not substitute one source type for the other.
- If a request explicitly asks for official evidence before a safety refusal, first use the allowed read-only knowledge search. Then refuse unsafe diagnosis, treatment, dose, write, shell, or cross-profile actions without citations.
"""


def actual_available_citation_ids(payload: Any) -> list[str]:
    """Return stable, deduplicated canonical IDs exposed by a tool's own contract."""
    if not isinstance(payload, Mapping):
        return []
    values = payload.get("available_citation_ids")
    if not isinstance(values, list):
        return []
    return list(dict.fromkeys(
        value for value in values
        if isinstance(value, str) and value.startswith((RECORD_PREFIX, KNOWLEDGE_PREFIX, OFFICIAL_WEB_PREFIX))
    ))


def recommended_record_citation_ids(name: str, payload: Mapping[str, Any], question: str) -> list[str]:
    """Suggest citations for requested fields; leave all available evidence intact."""
    from healthvault.application.query_rewrite import _subjects

    subjects = {code for code, _ in _subjects(question)}
    if "blood_glucose" in subjects:
        subjects.update({"fasting_glucose", "random_glucose", "postprandial_glucose_2h"})
    requested = []
    if name == "compare_health_indicators":
        for item in payload.get("comparisons", []):
            if not subjects or item["code"] in subjects:
                requested.extend(f"REC:{item[point]['report_id']}:{item['code']}" for point in ("earliest", "latest"))
    elif name == "generate_visit_brief":
        requested = [item["latest"]["record_citation_id"] for item in payload.get("indicators", [])
                     if not subjects or item["code"] in subjects]
    elif name == "query_health_timeline":
        for report in payload.get("timeline", []):
            matching = [item for item in report.get("indicators", []) if not subjects or item["code"] in subjects]
            requested.extend(f"REC:{report['report_id']}:{item['code']}" for item in (matching if subjects else matching[:1]))
    available = actual_available_citation_ids(payload)
    return list(dict.fromkeys(item for item in requested if item in available))
