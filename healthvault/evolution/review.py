"""Persistent, human-gated review workflow for shared Skill patch candidates."""

from __future__ import annotations

import hashlib
import asyncio
import json
import os
import re
import threading
import uuid
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping

from agents.frontmatter import parse_frontmatter
from healthvault.evolution.publisher import SkillPublisher
from healthvault.evolution.signal_extractor import (
    ExperienceCard,
    SignalExtractor,
    eligible_cards,
    observation_from_persisted_trace,
)
from healthvault.evolution.skill_patch import (
    PatchGenerator,
    PatchPreview,
    build_patch_preview,
)
from healthvault.evolution.verifier import CandidateVerifier


TARGET_SKILL = "HealthVault-Task-Execution"
REVIEW_STATUSES = frozenset({
    "pending_review", "watch", "rejected", "published", "stale", "gate_rejected",
})
_SAFE_ID = re.compile(r"[0-9a-f]{20}\Z")


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _sha(content: str | bytes) -> str:
    raw = content.encode("utf-8") if isinstance(content, str) else content
    return hashlib.sha256(raw).hexdigest()


def _atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        temporary.write_text(
            json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _case(case_id: str, passed: bool, detail: str) -> dict[str, object]:
    return {"case_id": case_id, "passed": passed, "detail": detail}


def evaluate_candidate(active: str, candidate: str) -> dict[str, Any]:
    """Static contract checks only; no claim of task accuracy or held-out evaluation."""
    verifier = CandidateVerifier(TARGET_SKILL)

    def probe(content: str, *, baseline: bool) -> list[dict[str, object]]:
        verification = verifier.verify_content(
            content,
            action="create" if baseline else "evolve",
            active_content=None if baseline else active,
        )
        try:
            parsed = parse_frontmatter(content)
            frontmatter_ok = parsed.meta.get("name") == TARGET_SKILL
        except Exception:
            frontmatter_ok = False
        safety = content.split("## Safety Boundaries", 1)
        active_safety = active.split("## Safety Boundaries", 1)
        safety_unchanged = (
            baseline
            or len(safety) == 2
            and len(active_safety) == 2
            and safety[1].replace("\r\n", "\n") == active_safety[1].replace("\r\n", "\n")
        )
        tool_mentions = set(re.findall(r"\b[a-z][a-z0-9_]+\b", content))
        forbidden_write = "confirm_health_records" in tool_mentions
        web_index = content.find("search_official_health_web")
        local_index = content.find("search_health_knowledge")
        local_before_web = web_index < 0 or (local_index >= 0 and local_index < web_index)
        return [
            _case("legacy.frontmatter", frontmatter_ok, "Skill identity remains valid."),
            _case("legacy.verifier", verification.status == "verified", "Static Skill verifier."),
            _case("legacy.safety_boundary", safety_unchanged, "Safety section is unchanged after newline normalization."),
            _case("contract.no_write_tool", not forbidden_write, "Read-only workflow cannot select confirmation."),
            _case("contract.local_before_web", local_before_web, "Official web fallback remains local-first."),
        ]

    baseline_cases = probe(active, baseline=True)
    candidate_cases = probe(candidate, baseline=False)
    baseline_passed = sum(bool(item["passed"]) for item in baseline_cases)
    candidate_passed = sum(bool(item["passed"]) for item in candidate_cases)
    count = len(candidate_cases)
    return {
        "protocol": "skill-review-static-v2",
        "legacy_regression": candidate_cases[:3],
        "contract_probes": candidate_cases[3:],
        "held_out": {"status": "not_run", "reason": "Requires a separately frozen offline task benchmark."},
        "baseline": {
            "passed": baseline_passed,
            "total": len(baseline_cases),
            "pass_rate": baseline_passed / len(baseline_cases),
        },
        "candidate": {
            "passed": candidate_passed,
            "total": count,
            "pass_rate": candidate_passed / count,
        },
        "delta": candidate_passed / count - baseline_passed / len(baseline_cases),
        "passed": candidate_passed == count,
    }


def _evaluation_regression(active: str):
    def check(candidate: str) -> tuple[bool, tuple[str, ...]]:
        evaluation = evaluate_candidate(active, candidate)
        failures = tuple(
            str(case["case_id"])
            for group in (evaluation["legacy_regression"], evaluation["contract_probes"])
            for case in group
            if not case["passed"]
        )
        return not failures, failures

    return check


def _risk(preview: PatchPreview, evaluation: Mapping[str, Any] | None) -> dict[str, Any]:
    reasons: list[str] = []
    if any(patch.operation == "delete" for patch in preview.patches):
        reasons.append("deletes_existing_workflow_text")
    if any("search_official_health_web" in patch.tool_path for patch in preview.patches):
        reasons.append("touches_network_fallback_workflow")
    if preview.status != "pending_manual_review":
        reasons.append("automatic_gate_rejected")
    if evaluation is not None and not evaluation.get("passed", False):
        reasons.append("static_regression_failed")
    reasons.append("formal_heldout_not_run")
    level = "high" if "static_regression_failed" in reasons else "medium"
    return {"level": level, "reasons": reasons or ["bounded_workflow_only_patch"]}


class SkillReviewStore:
    """Atomic file store plus publish/reject/watch/rollback state transitions."""

    def __init__(self, skill_root: str | Path, evolution_root: str | Path, *, scope: str = "global_shared_skill") -> None:
        self.skill_root = Path(skill_root)
        self.root = Path(evolution_root)
        self.review_root = self.root / "reviews"
        self.publisher = SkillPublisher(self.skill_root, self.root)
        self.scope = scope
        self._lock = threading.RLock()

    @property
    def active_path(self) -> Path:
        return self.skill_root / TARGET_SKILL / "SKILL.md"

    def _path(self, candidate_id: str) -> Path:
        if not _SAFE_ID.fullmatch(candidate_id):
            raise ValueError("Invalid Skill candidate ID.")
        return self.review_root / f"{candidate_id}.json"

    def get(self, candidate_id: str) -> dict[str, Any] | None:
        path = self._path(candidate_id)
        if not path.is_file():
            return None
        value = json.loads(path.read_text(encoding="utf-8"))
        return value if isinstance(value, dict) else None

    def list(self) -> list[dict[str, Any]]:
        if not self.review_root.is_dir():
            return []
        values = [
            value for path in self.review_root.glob("*.json")
            if _SAFE_ID.fullmatch(path.stem)
            and (value := self.get(path.stem)) is not None
        ]
        return sorted(values, key=lambda item: item["created_at"], reverse=True)

    def by_evidence_fingerprint(self, fingerprint: str) -> dict[str, Any] | None:
        return next(
            (item for item in self.list() if item.get("evidence_fingerprint") == fingerprint),
            None,
        )

    def create(
        self,
        preview: PatchPreview,
        cards: Iterable[ExperienceCard],
        *,
        evidence_fingerprint: str,
        evaluation: Mapping[str, Any] | None,
    ) -> dict[str, Any]:
        card_list = tuple(cards)
        identity = preview.candidate_skill_sha256 or hashlib.sha256(
            (evidence_fingerprint + "\0" + "\n".join(preview.reasons)).encode("utf-8")
        ).hexdigest()
        candidate_id = hashlib.sha256(
            f"{preview.active_skill_sha256}\0{identity}\0{evidence_fingerprint}".encode("utf-8")
        ).hexdigest()[:20]
        existing = self.get(candidate_id)
        if existing is not None:
            return existing
        source_trace_ids = sorted({trace for card in card_list for trace in card.trace_ids})
        record = {
            "schema_version": "healthvault-skill-review-v1",
            "candidate_id": candidate_id,
            "scope": self.scope,
            "skill_name": TARGET_SKILL,
            "status": "pending_review" if preview.status == "pending_manual_review" else "gate_rejected",
            "created_at": _now(),
            "updated_at": _now(),
            "active_skill_sha256": preview.active_skill_sha256,
            "candidate_skill_sha256": preview.candidate_skill_sha256,
            "candidate_content": preview.candidate_content,
            "unified_diff": preview.unified_diff,
            "patches": [asdict(patch) for patch in preview.patches],
            "source_trace_ids": source_trace_ids,
            "experience_cards": [
                {**asdict(card), "lesson": (
                    "Explicitly confirmed workflow correction."
                    if card.verification_source == "user_confirmed" else card.lesson
                )}
                for card in card_list
            ],
            "applicability_conditions": sorted({
                item for patch in preview.patches for item in patch.applicability_conditions
            }),
            "when_not_to_use": sorted({
                item for patch in preview.patches for item in patch.when_not_to_use
            }),
            "verification": {
                "status": preview.status,
                "reasons": list(preview.reasons),
                "generator_calls": preview.generator_calls,
            },
            "evaluation": dict(evaluation or {}),
            "risk": _risk(preview, evaluation),
            "evidence_fingerprint": evidence_fingerprint,
            "decisions": [],
        }
        with self._lock:
            _atomic_json(self._path(candidate_id), record)
        return record

    def decide(self, candidate_id: str, decision: str, actor: str) -> dict[str, Any]:
        if decision not in {"approve", "reject", "watch"}:
            raise ValueError("Unknown Skill review decision.")
        with self._lock:
            record = self.get(candidate_id)
            if record is None:
                raise FileNotFoundError("Skill candidate does not exist.")
            if record["status"] in {"published", "rejected", "stale", "gate_rejected"}:
                raise ValueError("Skill candidate is not reviewable.")
            outcome = decision
            if decision == "approve":
                if self.scope == 'personal_skill':
                    from healthvault.evolution.execution_probe import valid_execution_gate
                    probe = record.get('evaluation',{}).get('workflow_probe',{})
                    if (not probe.get('passed') or probe.get('active_sha256') != record['active_skill_sha256']
                            or probe.get('candidate_sha256') != record['candidate_skill_sha256']
                            or not valid_execution_gate(probe, record['active_skill_sha256'], record['candidate_skill_sha256'])):
                        raise ValueError('Personal workflow regression and native execution probes must pass before publication.')
                if not self.active_path.is_file() or record.get("candidate_content") is None:
                    raise ValueError("Skill candidate content is unavailable.")
                active = self.active_path.read_bytes().decode("utf-8")
                if _sha(active) != record["active_skill_sha256"]:
                    record["status"] = "stale"
                    outcome = "stale"
                else:
                    candidate = str(record["candidate_content"])
                    evaluation = evaluate_candidate(active, candidate)
                    verification = CandidateVerifier(TARGET_SKILL).verify_content(
                        candidate, action="evolve", active_content=active,
                    )
                    if verification.status != "verified" or not evaluation["passed"]:
                        raise ValueError("Skill candidate no longer passes publication gates.")
                    release = self.publisher.publish_content(
                        candidate,
                        action="evolve",
                        verification=verification,
                        approved=True,
                    )
                    if release.status != "published":
                        raise RuntimeError(f"Skill publication failed: {release.status}")
                    record["status"] = "published"
                    record["release"] = {
                        "active_sha256": release.active_sha256,
                        "previous_sha256": release.previous_sha256,
                    }
            elif decision == "reject":
                record["status"] = "rejected"
            else:
                record["status"] = "watch"
            record["updated_at"] = _now()
            record["decisions"].append({"decision": outcome, "actor": actor, "at": _now()})
            _atomic_json(self._path(candidate_id), record)
            return record

    def rollback(self, actor: str) -> dict[str, Any]:
        with self._lock:
            result = self.publisher.rollback()
            audit = {
                "status": result.status,
                "active_sha256": result.active_sha256,
                "previous_sha256": result.previous_sha256,
                "actor": actor,
                "at": _now(),
            }
            _atomic_json(self.root / "reviews" / "rollback-last.json", audit)
            return audit


class SkillEvolutionService:
    """Turn persisted redacted traces into at most one gated local Patch review."""

    def __init__(
        self,
        skill_root: str | Path,
        evolution_root: str | Path,
        patch_side_query: Any,
        *,
        scope: str = "global_shared_skill",
    ) -> None:
        self.store = SkillReviewStore(skill_root, evolution_root, scope=scope)
        self.root = Path(evolution_root)
        self.patch_side_query = patch_side_query
        self._generation_lock = asyncio.Lock()

    def trace_paths(self) -> tuple[Path, ...]:
        if not self.store.active_path.is_file():
            return ()
        active_sha = _sha(self.store.active_path.read_bytes())
        paths: list[Path] = []
        for path in sorted((self.root / "traces").glob("*.json"))[-200:]:
            try:
                value = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            if (
                isinstance(value, dict)
                and value.get("schema_version") == "healthvault-task-trace-v1"
                and isinstance(value.get("skill"), dict)
                and value["skill"].get("sha256") == active_sha
                and isinstance(value.get("signal_context"), dict)
            ):
                paths.append(path)
        return tuple(paths)

    async def consider(
        self,
        *,
        feedback_by_trace_id: Mapping[str, str] | None = None,
        user_confirmed_trace_ids: Iterable[str] = (),
        only_trace_ids: Iterable[str] | None = None,
    ) -> dict[str, Any] | None:
        async with self._generation_lock:
            return await self._consider(
                feedback_by_trace_id=feedback_by_trace_id,
                user_confirmed_trace_ids=user_confirmed_trace_ids,
                only_trace_ids=only_trace_ids,
            )

    async def _consider(
        self,
        *,
        feedback_by_trace_id: Mapping[str, str] | None,
        user_confirmed_trace_ids: Iterable[str],
        only_trace_ids: Iterable[str] | None,
    ) -> dict[str, Any] | None:
        if not self.store.active_path.is_file():
            return None
        selected = set(only_trace_ids or ())
        observations = []
        feedback = dict(feedback_by_trace_id or {})
        confirmed = set(user_confirmed_trace_ids)
        for path in self.trace_paths():
            raw = json.loads(path.read_text(encoding="utf-8"))
            trace_id = str(raw["trace_id"])
            if selected and trace_id not in selected:
                continue
            observations.append(observation_from_persisted_trace(
                path,
                feedback=feedback.get(trace_id),
                user_confirmed=trace_id in confirmed,
            ))
        if not observations:
            return None
        report = await SignalExtractor().extract(observations)
        cards = eligible_cards(report.cards)
        if not cards:
            return None
        active = self.store.active_path.read_bytes().decode("utf-8")
        fingerprint = hashlib.sha256(json.dumps({
            "active": _sha(active),
            "cards": sorted(card.card_id for card in cards),
        }, sort_keys=True).encode("utf-8")).hexdigest()
        existing = self.store.by_evidence_fingerprint(fingerprint)
        if existing is not None:
            return existing
        open_review = next((
            item for item in self.store.list()
            if item["active_skill_sha256"] == _sha(active)
            and item["status"] in {"pending_review", "watch"}
        ), None)
        if open_review is not None:
            return open_review
        generator = PatchGenerator(self.patch_side_query)
        preview = await build_patch_preview(
            cards,
            active_content=active,
            generator=generator,
            verifier=CandidateVerifier(TARGET_SKILL),
            regression_check=_evaluation_regression(active),
        )
        evaluation = (
            evaluate_candidate(active, preview.candidate_content)
            if preview.candidate_content is not None
            else None
        )
        return self.store.create(
            preview,
            cards,
            evidence_fingerprint=fingerprint,
            evaluation=evaluation,
        )

    def latest_trace_id_for_task(self, trace_task_id: str) -> str | None:
        latest: tuple[str, str] | None = None
        for path in self.trace_paths():
            value = json.loads(path.read_text(encoding="utf-8"))
            if value.get("task_id") != trace_task_id:
                continue
            marker = str(value.get("started_at", ""))
            if latest is None or marker > latest[0]:
                latest = (marker, str(value["trace_id"]))
        return latest[1] if latest else None

