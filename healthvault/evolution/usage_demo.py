"""Fixed synthetic active-Skill usage and lifecycle demo with zero API calls."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
import uuid
from pathlib import Path

from agents.skills import load_project_skill_resources, project_skill_manifest
from healthvault.evolution.synthetic_demo import SKILL_V1, SKILL_V2
from healthvault.evolution.usage_lifecycle import (
    SkillLifecycleManager,
    SkillUsageEvaluator,
    SkillUsageEvent,
)


SKILL_NAME = "HealthVault-Task-Execution"
V1_SHA = hashlib.sha256(SKILL_V1.encode("utf-8")).hexdigest()
V2_SHA = hashlib.sha256(SKILL_V2.encode("utf-8")).hexdigest()


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def prepare_synthetic_v1(workspace_root: str | Path) -> None:
    root = Path(workspace_root)
    active = root / ".bear" / "skills" / SKILL_NAME / "SKILL.md"
    version = root / ".bear" / "skill-evolution" / "healthvault" / "versions" / V1_SHA / "SKILL.md"
    active.parent.mkdir(parents=True, exist_ok=True)
    version.parent.mkdir(parents=True, exist_ok=True)
    active.write_bytes(SKILL_V1.encode("utf-8"))
    version.write_bytes(SKILL_V1.encode("utf-8"))
    _write_json(
        root / ".bear" / "skill-evolution" / "healthvault" / "release-state.json",
        {"history": [V1_SHA], "rolled_back_from": None},
    )


def activate_synthetic_v2(workspace_root: str | Path) -> None:
    root = Path(workspace_root)
    active = root / ".bear" / "skills" / SKILL_NAME / "SKILL.md"
    version = root / ".bear" / "skill-evolution" / "healthvault" / "versions" / V2_SHA / "SKILL.md"
    if not active.is_file():
        prepare_synthetic_v1(root)
    version.parent.mkdir(parents=True, exist_ok=True)
    active.write_bytes(SKILL_V2.encode("utf-8"))
    version.write_bytes(SKILL_V2.encode("utf-8"))
    _write_json(
        root / ".bear" / "skill-evolution" / "healthvault" / "release-state.json",
        {"history": [V1_SHA, V2_SHA], "rolled_back_from": None},
    )


def run_usage_demo(
    workspace_root: str | Path,
    *,
    fixture_path: str | Path,
    summary_path: str | Path | None = None,
) -> dict[str, object]:
    workspace = Path(workspace_root)
    fixture = json.loads(Path(fixture_path).read_text(encoding="utf-8"))
    events = [SkillUsageEvent(**item) for item in fixture["events"]]
    evolution_root = workspace / ".bear" / "skill-evolution" / "healthvault"
    skill_root = workspace / ".bear" / "skills"
    evaluator = SkillUsageEvaluator(evolution_root)
    lifecycle = SkillLifecycleManager(skill_root, evolution_root)

    prepare_synthetic_v1(workspace)
    keep = evaluator.evaluate(events, SKILL_NAME, V1_SHA, "timeline")
    keep_lifecycle = lifecycle.apply(keep)
    watch = evaluator.evaluate(events, SKILL_NAME, V1_SHA, "knowledge")
    watch_lifecycle = lifecycle.apply(watch)

    activate_synthetic_v2(workspace)
    safety = evaluator.evaluate(events, SKILL_NAME, V2_SHA, "visit_brief")
    safety_lifecycle = lifecycle.apply(safety)
    manifest = project_skill_manifest(load_project_skill_resources(skill_root, (SKILL_NAME,)))
    archive_evidence = json.loads(
        (evolution_root / "archive" / f"{V2_SHA}.json").read_text(encoding="utf-8")
    )
    summary: dict[str, object] = {
        "schema_version": "healthvault-skill-usage-summary-v1",
        "fixture_version": fixture["fixture_version"],
        "data_classification": "synthetic",
        "skill_name": SKILL_NAME,
        "v1_keep": keep.to_dict(),
        "v1_watch": watch.to_dict(),
        "v2_safety": safety.to_dict(),
        "keep_lifecycle": keep_lifecycle.to_dict(),
        "watch_lifecycle": watch_lifecycle.to_dict(),
        "lifecycle": safety_lifecycle.to_dict(),
        "archive_evidence": archive_evidence,
        "final_whitelist_manifest": manifest[0],
        "archived_version_sha256": V2_SHA,
        "candidate_generation_calls": 0,
        "api_calls": 0,
        "tokens": keep.total_tokens + watch.total_tokens + safety.total_tokens,
        "cost_usd": round(
            keep.total_cost_usd + watch.total_cost_usd + safety.total_cost_usd, 8
        ),
        "claims_real_online_benefit": False,
    }
    if summary_path is not None:
        target = Path(summary_path)
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_name(f".{target.name}.{uuid.uuid4().hex}.tmp")
        try:
            _write_json(temporary, summary)
            os.replace(temporary, target)
        finally:
            temporary.unlink(missing_ok=True)
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--fixture",
        default=str(
            Path(__file__).parents[1]
            / "evaluation"
            / "skill"
            / "task-020-synthetic-fixture.json"
        ),
    )
    parser.add_argument("--summary")
    arguments = parser.parse_args()
    with tempfile.TemporaryDirectory(prefix="healthvault-task-020-") as workspace:
        summary = run_usage_demo(
            workspace,
            fixture_path=arguments.fixture,
            summary_path=arguments.summary,
        )
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
