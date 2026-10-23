"""Deterministic publication gate for the target HealthVault Skill."""

from __future__ import annotations

import re
from dataclasses import dataclass

from agents.frontmatter import parse_frontmatter


REQUIRED_SECTIONS = (
    "When to Use",
    "Prerequisites",
    "Steps",
    "Failure Handling",
    "Verification",
    "Safety Boundaries",
)
DANGEROUS_PATTERN = re.compile(
    r"(?i)(rm\s+-rf|del\s+/[sq]|format\s+[a-z]:|git\s+reset\s+--hard|git\s+clean\s+-[a-z]*f)"
)
SENSITIVE_PATTERN = re.compile(
    r"(?i)(api[_ -]?key\s*[:=]?\s*\S+|(?<![a-z0-9-])sk-[a-z0-9_-]{6,}|患者.{0,24}(?:血压|血糖|诊断|用药)|\b(?:BP|glucose).{0,12}\d+)"
)
UNCONDITIONAL_MULTI_TOOL_PATTERN = re.compile(
    r"(?is)(?:always|every\s+(?:task|request)|regardless\s+of\s+intent|unconditionally|"
    r"所有任务|无条件).{0,400}query_health_timeline.{0,240}"
    r"compare_health_indicators.{0,240}generate_visit_brief"
)
TASK024_ROUTE_MARKERS = (
    "timeline",
    "comparison",
    "knowledge",
    "visit brief",
    "comprehensive summary",
)
TASK024_ROUTE_TOOLS = {
    "timeline": ("query_health_timeline",),
    "comparison": ("compare_health_indicators",),
    "knowledge": ("search_health_knowledge",),
    "visit brief": ("generate_visit_brief",),
    "comprehensive summary": (
        "query_health_timeline",
        "compare_health_indicators",
        "generate_visit_brief",
    ),
}
TASK024_TOOL_NAMES = tuple(dict.fromkeys(
    tool for tools in TASK024_ROUTE_TOOLS.values() for tool in tools
))


@dataclass(frozen=True)
class VerificationResult:
    status: str
    reasons: tuple[str, ...]
    content_sha256: str | None = None


class CandidateVerifier:
    def __init__(self, target_skill_name: str) -> None:
        self._target = target_skill_name

    def verify_content(
        self,
        content: str,
        *,
        action: str,
        active_content: str | None = None,
        policy: str | None = None,
    ) -> VerificationResult:
        import hashlib

        reasons: list[str] = []
        try:
            parsed = parse_frontmatter(content)
        except Exception:
            parsed = None
            reasons.append("invalid_frontmatter")
        if parsed is not None:
            if parsed.meta.get("name") != self._target:
                reasons.append("target_skill_mismatch")
            if not str(parsed.meta.get("description", "")).strip():
                reasons.append("missing_description")
            body = parsed.body
            for section in REQUIRED_SECTIONS:
                if not re.search(rf"(?m)^##\s+{re.escape(section)}\s*$", body):
                    reasons.append(f"missing_section:{section}")
        if DANGEROUS_PATTERN.search(content):
            reasons.append("dangerous_command")
        if SENSITIVE_PATTERN.search(content):
            reasons.append("sensitive_or_patient_fact")
        if UNCONDITIONAL_MULTI_TOOL_PATTERN.search(content):
            reasons.append("unconditional_multi_tool_chain")
        if policy == "task024_v2":
            if re.search(r"(?m)^##\s+When Not to Use\s*$", content) is None:
                reasons.append("missing_section:When Not to Use")
            lowered = content.lower()
            for marker in TASK024_ROUTE_MARKERS:
                if marker not in lowered:
                    reasons.append(f"missing_route:{marker}")
            for route, expected_tools in TASK024_ROUTE_TOOLS.items():
                matches = re.findall(
                    rf"(?im)^\s*(?:\d+[.)]|[-*])\s*{re.escape(route)}\s*:\s*(.+)$",
                    content,
                )
                if len(matches) != 1:
                    reasons.append(f"route_declaration_count:{route}")
                    continue
                declared = tuple(
                    tool for tool in TASK024_TOOL_NAMES if tool in matches[0]
                )
                if set(declared) != set(expected_tools):
                    reasons.append(f"route_tool_mismatch:{route}")
                elif route == "comprehensive summary" and any(
                    matches[0].find(first) >= matches[0].find(second)
                    for first, second in zip(expected_tools, expected_tools[1:])
                ):
                    reasons.append("route_tool_order:comprehensive summary")
            if "retryable=true" not in lowered or "retry once" not in lowered:
                reasons.append("missing_bounded_retry_rule")
            if not all(item in lowered for item in ("insufficient_evidence", "rejected", "stop")):
                reasons.append("missing_terminal_stop_rule")
            if (
                "only for comprehensive summary" not in lowered
                or not all(item in lowered for item in (
                    "query_health_timeline",
                    "compare_health_indicators",
                    "generate_visit_brief",
                ))
            ):
                reasons.append("missing_scoped_comprehensive_chain")
        if action not in {"create", "evolve"}:
            reasons.append("invalid_action")
        if action == "create" and active_content is not None:
            reasons.append("duplicate_ability")
        if action == "evolve" and active_content is not None and content == active_content:
            reasons.append("empty_patch")
        digest = hashlib.sha256(content.encode("utf-8")).hexdigest() if content else None
        return VerificationResult(
            status="rejected" if reasons else "verified",
            reasons=tuple(reasons),
            content_sha256=digest,
        )
