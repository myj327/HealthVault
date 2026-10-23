"""Deterministic final-response boundary for the read-only HealthVault Agent."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from healthvault.application.citation_policy import actual_available_citation_ids


INSUFFICIENT_RESULT_STATUSES = frozenset({"empty", "insufficient_data"})
REJECTED_RESULT_STATUSES = frozenset(
    {
        "rejected",
        "invalid_request",
        "confirmation_required",
        "source_conflict",
        "report_conflict",
    }
)
SAFETY_ISSUE_CODES = frozenset(
    {
        "unsupported_diagnosis",
        "unsupported_treatment",
        "unsupported_future_prediction",
        "unsupported_personal_record",
    }
)


def parse_final_response(value: Any) -> Mapping[str, Any] | None:
    """Parse one strict bare or JSON-fenced final object."""
    if not isinstance(value, str) or not value.strip():
        return None
    text = value.strip()
    fenced = re.fullmatch(r"```json\s*(\{.*\})\s*```", text, flags=re.DOTALL)
    if fenced is not None:
        text = fenced.group(1)
    elif text.startswith("```") or text.endswith("```"):
        return None
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        return None
    return parsed if isinstance(parsed, Mapping) else None


def looks_like_structured_final(value: Any) -> bool:
    """Return whether a response attempted the runtime's JSON final contract."""
    if not isinstance(value, str):
        return False
    return value.lstrip().startswith(("{", "```json"))


@dataclass(frozen=True)
class NormalizedFinalResponse:
    final_status: str
    citations: tuple[str, ...]
    refusal: bool


class FinalResponseNormalizer:
    """Normalize only real tool state and canonical top-level citations."""

    def normalize(
        self,
        final: Mapping[str, Any] | None,
        tool_calls: Sequence[Mapping[str, Any]],
    ) -> NormalizedFinalResponse:
        statuses = [str(call.get("result_status", call.get("status", ""))) for call in tool_calls]
        issue_codes = {
            str(code)
            for call in tool_calls
            for code in call.get("issue_codes", [])
            if isinstance(code, str)
        }
        available: list[str] = []
        recommended: list[str] = []
        for call in tool_calls:
            payload = {"available_citation_ids": call.get("available_citation_ids", [])}
            available.extend(actual_available_citation_ids(payload))
            recommended.extend(actual_available_citation_ids({"available_citation_ids": call.get("recommended_citation_ids", [])}))
        available = list(dict.fromkeys(available))
        last_status = statuses[-1] if statuses else ""
        if last_status in REJECTED_RESULT_STATUSES or issue_codes & SAFETY_ISSUE_CODES:
            return NormalizedFinalResponse("rejected", (), True)
        if (
            isinstance(final, Mapping)
            and final.get("final_status") == "rejected"
            and final.get("refusal") is True
        ):
            # A refusal remains safe even when the permitted lookup found no
            # evidence.  This does not inspect the request or infer medicine.
            return NormalizedFinalResponse("rejected", (), True)
        if last_status in INSUFFICIENT_RESULT_STATUSES:
            return NormalizedFinalResponse("insufficient_evidence", (), True)
        if "partial" in statuses and not available:
            return NormalizedFinalResponse("insufficient_evidence", (), True)
        if not isinstance(final, Mapping) or final.get("final_status") != "ok":
            return NormalizedFinalResponse("rejected", (), True)
        requested = final.get("citations", [])
        if not isinstance(requested, list):
            requested = []
        direct = [item for item in requested if item in available]
        if available and not direct:
            return NormalizedFinalResponse("rejected", (), True)
        limit = len(list(dict.fromkeys(recommended))) or (2 if any(call.get("name") == "compare_health_indicators" for call in tool_calls) else 1)
        return NormalizedFinalResponse("ok", tuple(dict.fromkeys(direct))[:limit], False)
