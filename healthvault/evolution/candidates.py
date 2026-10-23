"""Eligibility batching and strict side-query candidate generation."""

from __future__ import annotations

import hashlib
import inspect
import json
import os
import re
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Awaitable, Callable


TARGET_SKILL = "HealthVault-Task-Execution"
ELIGIBLE_SIGNALS = frozenset({
    "explicit_correction",
    "deterministic_failure",
    "verified_reusable_workflow",
})
FORBIDDEN_EVIDENCE_PATTERN = re.compile(
    r"(?i)(api[_ -]?key|(?<![a-z0-9-])sk-[a-z0-9_-]{6,}|患者.{0,24}(?:血压|血糖|诊断|用药)|\b(?:BP|glucose).{0,12}\d+)"
)

CANDIDATE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": ["action", "skill_name", "content", "rationale", "trace_ids"],
    "properties": {
        "action": {"type": "string", "enum": ["create", "evolve"]},
        "skill_name": {"type": "string", "const": TARGET_SKILL},
        "content": {"type": "string", "minLength": 1},
        "rationale": {"type": "string", "minLength": 1},
        "trace_ids": {"type": "array", "minItems": 1, "uniqueItems": True, "items": {"type": "string", "minLength": 1}},
    },
}


@dataclass(frozen=True)
class EvolutionSignal:
    trace_id: str
    signal_type: str
    lesson: str
    reusable: bool
    contains_patient_or_medical_fact: bool = False
    has_increment: bool = True
    source_trace_ids: tuple[str, ...] = ()


@dataclass(frozen=True)
class SkillCandidate:
    candidate_id: str
    action: str
    skill_name: str
    content: str
    rationale: str
    trace_ids: tuple[str, ...]
    status: str = "pending_verification"


SideQuery = Callable[[str, dict[str, Any]], Awaitable[dict[str, Any] | str]]


class CandidateGenerator:
    def __init__(self, evolution_root: str | Path, side_query: SideQuery) -> None:
        self._root = Path(evolution_root)
        self._side_query = side_query
        self.call_count = 0

    async def generate(
        self, signals: list[EvolutionSignal], action: str
    ) -> SkillCandidate:
        if action not in {"create", "evolve"} or not signals:
            raise ValueError("Candidate batch and action are required.")
        trace_ids = list(dict.fromkeys(
            trace_id
            for signal in signals
            for trace_id in (signal.source_trace_ids or (signal.trace_id,))
        ))
        prompt = json.dumps({
            "target_skill": TARGET_SKILL,
            "action": action,
            "evidence": [
                {
                    "trace_ids": list(signal.source_trace_ids or (signal.trace_id,)),
                    "signal_type": signal.signal_type,
                    "lesson": signal.lesson,
                }
                for signal in signals
            ],
            "rules": "Return one reusable workflow Skill only; exclude patient and medical facts.",
        }, ensure_ascii=False, sort_keys=True)
        self.call_count += 1
        response = self._side_query(prompt, CANDIDATE_SCHEMA)
        if inspect.isawaitable(response):
            response = await response
        if isinstance(response, str):
            try:
                response = json.loads(response)
            except json.JSONDecodeError as exc:
                raise ValueError("Candidate response is not valid JSON.") from exc
        self._validate_response(response, action, trace_ids)
        canonical = json.dumps(response, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        candidate = SkillCandidate(
            candidate_id=hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:20],
            action=response["action"],
            skill_name=response["skill_name"],
            content=response["content"],
            rationale=response["rationale"],
            trace_ids=tuple(response["trace_ids"]),
        )
        candidate_dir = self._root / "candidates"
        candidate_dir.mkdir(parents=True, exist_ok=True)
        target = candidate_dir / f"{candidate.candidate_id}.json"
        temporary = candidate_dir / f".{candidate.candidate_id}.{uuid.uuid4().hex}.tmp"
        try:
            temporary.write_text(
                json.dumps(asdict(candidate), ensure_ascii=False, sort_keys=True, indent=2) + "\n",
                encoding="utf-8",
            )
            os.replace(temporary, target)
        finally:
            temporary.unlink(missing_ok=True)
        return candidate

    @staticmethod
    def _validate_response(response: object, action: str, trace_ids: list[str]) -> None:
        if not isinstance(response, dict) or set(response) != set(CANDIDATE_SCHEMA["required"]):
            raise ValueError("Candidate response does not match the exact schema.")
        if response["action"] != action or response["skill_name"] != TARGET_SKILL:
            raise ValueError("Candidate action or target Skill is invalid.")
        if not isinstance(response["content"], str) or not response["content"].strip():
            raise ValueError("Candidate content is empty.")
        if not isinstance(response["rationale"], str) or not response["rationale"].strip():
            raise ValueError("Candidate rationale is empty.")
        ids = response["trace_ids"]
        if (
            not isinstance(ids, list)
            or not ids
            or any(not isinstance(item, str) or not item for item in ids)
            or len(ids) != len(set(ids))
            or not set(ids).issubset(trace_ids)
        ):
            raise ValueError("Candidate trace_ids must cite eligible batch evidence.")


class TaskEndProcessor:
    """Queue only strong reusable signals and invoke at most once per batch."""

    def __init__(self, generator: CandidateGenerator) -> None:
        self._generator = generator
        self._pending: list[EvolutionSignal] = []
        self._seen: set[tuple[str, str]] = set()

    def record(self, signal: EvolutionSignal) -> str:
        dedupe_key = (
            signal.trace_id,
            hashlib.sha256(signal.lesson.strip().encode("utf-8")).hexdigest(),
        )
        if (
            signal.signal_type not in ELIGIBLE_SIGNALS
            or not signal.reusable
            or signal.contains_patient_or_medical_fact
            or FORBIDDEN_EVIDENCE_PATTERN.search(signal.lesson) is not None
            or not signal.has_increment
            or not signal.trace_id.strip()
            or not signal.lesson.strip()
            or (
                signal.source_trace_ids
                and (
                    any(not trace_id.strip() for trace_id in signal.source_trace_ids)
                    or len(signal.source_trace_ids) != len(set(signal.source_trace_ids))
                )
            )
            or dedupe_key in self._seen
        ):
            return "ignored"
        self._seen.add(dedupe_key)
        self._pending.append(signal)
        return "eligible"

    async def process(self, action: str) -> SkillCandidate | None:
        if not self._pending:
            return None
        batch = self._pending
        self._pending = []
        return await self._generator.generate(batch, action)
