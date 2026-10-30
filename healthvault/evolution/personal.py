"""Authenticated personal workflow releases, isolated from project Skills."""
from __future__ import annotations

import hashlib
import json
import re
import threading
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from healthvault.evolution.review import SkillEvolutionService, TARGET_SKILL, _atomic_json, _now
from healthvault.evolution.usage_lifecycle import SkillLifecycleManager, SkillUsageEvaluator, SkillUsageEvent
from healthvault.evolution.failures import classify_trace_failure, usage_failure_rank


class PersonalSkillManager:
    def __init__(self, project_root: Path, evolution_root: Path, side_query: Any, *, evaluator: Any = None):
        self.project_root = project_root
        self.root = evolution_root / "personal"
        self.proposals = evolution_root / "shared-proposals"
        self.side_query = side_query
        self.evaluator = evaluator
        self._services: dict[str, SkillEvolutionService] = {}
        self._lock = threading.RLock()
        self._evaluation_locks: dict[str, Any] = {}

    @staticmethod
    def owner_ref(profile_id: str) -> str:
        return hashlib.sha256(profile_id.encode("utf-8")).hexdigest()[:24]

    def service(self, profile_id: str) -> SkillEvolutionService:
        owner = self.owner_ref(profile_id)
        with self._lock:
            if owner not in self._services:
                directory = self.root / owner
                skills = directory / ".bear" / "skills"
                active = skills / TARGET_SKILL / "SKILL.md"
                project = self.project_root / TARGET_SKILL / "SKILL.md"
                if project.is_file() and not active.is_file():
                    active.parent.mkdir(parents=True, exist_ok=True)
                    active.write_bytes(project.read_bytes())
                    _atomic_json(directory / "origin.json", {"project_sha256": hashlib.sha256(project.read_bytes()).hexdigest()})
                self._services[owner] = SkillEvolutionService(
                    skills, directory / "evolution", self.side_query, scope="personal_skill",
                )
            return self._services[owner]

    def settings(self, profile_id: str) -> dict[str, Any]:
        path = self.root / self.owner_ref(profile_id) / "settings.json"
        return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {"manual_review": False}

    def set_manual_review(self, profile_id: str, enabled: bool) -> dict[str, Any]:
        settings = {"manual_review": enabled, "updated_at": _now()}
        with self._lock:
            _atomic_json(self.root / self.owner_ref(profile_id) / "settings.json", settings)
        return settings

    def lifecycle(self, profile_id: str) -> SkillLifecycleManager:
        service = self.service(profile_id)
        return SkillLifecycleManager(service.store.skill_root, service.root)

    def injectable_root(self, profile_id: str) -> Path | None:
        service = self.service(profile_id)
        active = service.store.active_path
        if not active.is_file() or not self.lifecycle(profile_id).is_injectable(hashlib.sha256(active.read_bytes()).hexdigest()):
            return None
        return service.store.skill_root

    def usage(self, profile_id: str) -> dict[str, Any]:
        from healthvault.web.skill_descriptions import usage_explanation
        service = self.service(profile_id)
        state_path = service.root / "web-usage" / "summary.json"
        summary = json.loads(state_path.read_text(encoding="utf-8")) if state_path.is_file() else {"evaluations": [], "last_action": None}
        active = service.store.active_path
        digest = hashlib.sha256(active.read_bytes()).hexdigest() if active.is_file() else None
        project = self.project_root / TARGET_SKILL / 'SKILL.md'
        has_update = bool(digest and project.is_file() and digest != hashlib.sha256(project.read_bytes()).hexdigest())
        return {**summary, 'explanation_zh': usage_explanation(summary), "active_sha256": digest,
                "has_personal_update": has_update, "injectable": bool(digest and self.lifecycle(profile_id).is_injectable(digest)),
                "window_size": 10, "metric_scope": "workflow_contract", "independent_tasks_only": True}

    def disable(self, profile_id: str) -> dict[str, Any]:
        with self._lock:
            result = self.lifecycle(profile_id).disable_active()
            summary = self.usage(profile_id)
            _atomic_json(self.service(profile_id).root / "web-usage" / "summary.json", {
                "evaluations": summary["evaluations"], "last_action": {**result, "actions": ["disable"], 'reason': 'user_requested'}, "updated_at": _now(),
            })
            return result

    def rollback(self, profile_id: str, actor: str) -> dict[str, Any]:
        with self._lock:
            result = self.service(profile_id).store.rollback(actor)
            if result['status'] in {'rolled_back', 'already_rolled_back'}:
                summary = self.usage(profile_id)
                _atomic_json(self.service(profile_id).root / 'web-usage' / 'summary.json', {
                    'evaluations': summary['evaluations'],
                    'last_action': {**result, 'actions': ['rollback'], 'reason': 'user_requested'}, 'updated_at': _now(),
                })
            return result

    def record_usage(self, profile_id: str, trace_path: Path, *, injected: bool) -> dict[str, Any]:
        """Observe redacted runtime results; never generate a candidate in this path."""
        service = self.service(profile_id)
        if trace_path.resolve().parent != (service.root / "traces").resolve():
            raise ValueError("Usage trace must belong to the authenticated user.")
        trace = json.loads(trace_path.read_text(encoding="utf-8"))
        context = trace.get("signal_context") or {}
        failure = trace.get("failure_code")
        calls = trace.get("tool_calls") or []
        safety = next((str(call["failure_code"]) for call in calls if str(call.get("failure_code") or "").startswith("safety_")), None)
        if safety:
            failure = safety
        category = classify_trace_failure(trace)
        external = category == 'external_service'
        digest = trace["skill"]["sha256"]
        task_type = context.get("intent", "general_read_only_workflow")
        event_id = hashlib.sha256(f"{trace['task_id']}\0{digest}\0{task_type}".encode("utf-8")).hexdigest()[:20]
        event = SkillUsageEvent(
            run_id=event_id, task_id=trace["task_id"], task_type=task_type,
            skill_name=TARGET_SKILL, skill_sha256=digest, retrieved=True, injected=injected, used=injected and bool(calls),
            task_status="succeeded" if trace.get("result_status") == "succeeded" else "failed",
            validator_status='not_applicable' if external else "passed" if context.get("validator_status") == "passed" else "failed",
            hard_failure=bool(failure and str(failure).startswith("safety_")),
            failure_code="safety_contract_violation" if category == 'safety' else category + '_failure' if category != 'none' else None,
            tool_count=len(calls), step_count=len(calls), retry_count=0,
            input_tokens=trace.get("tokens", {}).get("input", 0), output_tokens=trace.get("tokens", {}).get("output", 0),
            latency_ms=trace.get("elapsed_ms", 0), cost_usd=trace.get("cost_usd", 0),
            failure_category=category, excluded_from_skill_rate=external,
        )
        with self._lock:
            directory = service.root / "web-usage"
            path = directory / "events" / f"{event_id}.json"
            legacy_id = hashlib.sha256(f"{trace['task_id']}\0{digest}".encode("utf-8")).hexdigest()[:20]
            legacy_path = directory / 'events' / f'{legacy_id}.json'
            if legacy_path.is_file() and json.loads(legacy_path.read_text(encoding='utf-8')).get('task_type') == task_type:
                # Reuse a matching old identity: upgrading must not count a task twice.
                path = legacy_path
                event = replace(event, run_id=legacy_id)
            if path.is_file():
                previous = json.loads(path.read_text(encoding="utf-8"))
                if usage_failure_rank(event.to_dict()) <= usage_failure_rank(previous):
                    return {"status": "already_recorded"}
            _atomic_json(path, {**event.to_dict(), "observed_at": datetime.now(timezone.utc).isoformat()})
            events = [json.loads(item.read_text(encoding="utf-8")) for item in (directory / "events").glob("*.json")]
            events = sorted((item for item in events if item["skill_sha256"] == digest and item["task_type"] == event.task_type), key=lambda item: item["observed_at"])
            attributable = [index for index, item in enumerate(events) if not item.get('excluded_from_skill_rate')]
            start = attributable[-10] if len(attributable) >= 10 else 0 if attributable else max(0, len(events) - 10)
            events = events[start:]
            evaluation = SkillUsageEvaluator(service.root).evaluate(
                [SkillUsageEvent(**{key: value for key, value in item.items() if key != "observed_at"}) for item in events], TARGET_SKILL, digest, event.task_type,
            )
            # Baseline project guidance remains governed by the developer's releases.
            action = self.lifecycle(profile_id).apply(evaluation).to_dict() if injected else None
            summary = self.usage(profile_id)
            evaluations = [item for item in summary["evaluations"] if (item["skill_sha256"], item["task_type"]) != (digest, event.task_type)]
            _atomic_json(directory / "summary.json", {"evaluations": [*evaluations, evaluation.to_dict()], "last_action": action, "updated_at": _now()})
            return {"status": "excluded_runtime_failure" if external else "recorded",
                    "failure_category": category, "excluded_from_skill_rate": external,
                    "evaluation": evaluation.to_dict(), "lifecycle": action}

    async def consider(self, profile_id: str, **kwargs: Any) -> dict[str, Any] | None:
        import asyncio
        from healthvault.evolution.execution_probe import valid_execution_gate
        service = self.service(profile_id)
        if self.injectable_root(profile_id) is None:
            return None
        record = await service.consider(**kwargs)
        if record and record['status']=='pending_review':
            lock = self._evaluation_locks.setdefault(self.owner_ref(profile_id),asyncio.Lock())
            async with lock:
                record = service.store.get(record['candidate_id'])
                if (record and record['status']=='pending_review' and not valid_execution_gate(
                        record['evaluation'].get('workflow_probe',{}), record['active_skill_sha256'], record['candidate_skill_sha256'])):
                    try:
                        if self.evaluator is None:
                            raise ValueError('No workflow evaluator configured.')
                        result = await self.evaluator(service.store.active_path.read_bytes().decode('utf-8'),record['candidate_content'])
                        record['evaluation']['workflow_probe'] = result
                        if (not result.get('passed') or not valid_execution_gate(result, record['active_skill_sha256'], record['candidate_skill_sha256'])):
                            record['status'] = 'gate_rejected'
                    except Exception as exc:
                        record['evaluation']['workflow_probe'] = {'passed':False,'status':'unavailable','error':type(exc).__name__}
                    record['updated_at'] = _now()
                    _atomic_json(service.store._path(record['candidate_id']),record)
        with self._lock:
            record = service.store.get(record["candidate_id"]) if record else None
            if (record and record["status"] == "pending_review"
                    and valid_execution_gate(record['evaluation'].get('workflow_probe',{}),
                                             record['active_skill_sha256'], record['candidate_skill_sha256'])
                    and not self.settings(profile_id)["manual_review"]):
                # Publication repeats verifier, immutable-safety and version checks.
                record = service.store.decide(record["candidate_id"], "approve", "automatic_personal_gate")
        return record

    def candidates(self) -> list[dict[str, Any]]:
        records = []
        if self.root.is_dir():
            for directory in self.root.iterdir():
                if not re.fullmatch(r"[0-9a-f]{24}", directory.name):
                    continue
                from healthvault.evolution.review import SkillReviewStore
                store = SkillReviewStore(directory / ".bear" / "skills", directory / "evolution", scope="personal_skill")
                records.extend({**item, "owner_ref": directory.name} for item in store.list())
        return sorted(records, key=lambda item: item["created_at"], reverse=True)

    def share(self, profile_id: str, candidate_id: str) -> dict[str, Any]:
        record = self.service(profile_id).store.get(candidate_id)
        if not record or record["status"] not in {"pending_review", "published", "watch"}:
            raise ValueError("Only a verified workflow candidate can be shared.")
        owner = self.owner_ref(profile_id)
        proposal_id = hashlib.sha256(f"{owner}:{candidate_id}".encode()).hexdigest()[:20]
        path = self.proposals / f"{proposal_id}.json"
        with self._lock:
            if path.is_file():
                return json.loads(path.read_text(encoding="utf-8"))
            proposal = {key: value for key, value in record.items() if key not in {"candidate_content", "evidence_fingerprint", "decisions"}}
            proposal.update(proposal_id=proposal_id, owner_ref=owner, scope="shared_workflow_proposal",
                            status="pending_review", created_at=_now(), decisions=[])
            _atomic_json(path, proposal)
            return proposal

    def list_proposals(self) -> list[dict[str, Any]]:
        return [json.loads(path.read_text(encoding="utf-8")) for path in sorted(self.proposals.glob("*.json"))]

    def decide_proposal(self, proposal_id: str, decision: str, actor: str) -> dict[str, Any]:
        if not re.fullmatch(r"[0-9a-f]{20}", proposal_id) or decision not in {"approve", "reject", "watch"}:
            raise ValueError("Invalid shared proposal decision.")
        path = self.proposals / f"{proposal_id}.json"
        with self._lock:
            if not path.is_file():
                raise FileNotFoundError("Proposal does not exist.")
            record = json.loads(path.read_text(encoding="utf-8"))
            if record["status"] not in {"pending_review", "watch"}:
                raise ValueError("Proposal is already resolved.")
            record["status"] = {"approve": "approved_for_developer", "reject": "rejected", "watch": "watch"}[decision]
            record["decisions"].append({"decision": decision, "actor": actor, "at": _now()})
            _atomic_json(path, record)
            return record
