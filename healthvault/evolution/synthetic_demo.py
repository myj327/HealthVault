"""Runnable zero-API create -> evolve -> rollback TASK-019 demo."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import tempfile
import uuid
from pathlib import Path

from healthvault.evolution.candidates import (
    CandidateGenerator,
    EvolutionSignal,
    TaskEndProcessor,
)
from healthvault.evolution.publisher import SkillPublisher
from healthvault.evolution.verifier import CandidateVerifier

SKILL_V1 = """---
name: HealthVault-Task-Execution
description: Execute read-only HealthVault tasks with explicit evidence and validation.
---

## When to Use
Use for HealthVault timeline, indicator comparison, knowledge search, and visit-brief tasks.

## Prerequisites
Use only the tools exposed by the HealthVault runtime and the injected profile context.

## Steps
1. Identify the requested HealthVault outcome before selecting a tool.
2. Run only approved read-only domain tools and preserve returned citations.
3. Return the tool status accurately; never present a rejection as success.

## Failure Handling
If evidence is insufficient or a tool rejects input, report that result without guessing.

## Verification
Check the final status, requested fields, and every citation before answering.

## Safety Boundaries
Never infer patient facts or medical advice, expose secrets, or invoke file, shell, MCP, or write tools.
"""

SKILL_V2 = SKILL_V1.replace(
    "If evidence is insufficient or a tool rejects input, report that result without guessing.",
    "If evidence is insufficient or a tool rejects input, report that result without guessing. Retry only when the tool result explicitly marks the failure as retryable.",
)


async def run_synthetic_demo(
    workspace_root: str | Path,
    *,
    summary_path: str | Path | None = None,
) -> dict[str, object]:
    """Run generated candidates through verification, publication, and rollback."""
    workspace = Path(workspace_root)
    skill_root = workspace / ".bear" / "skills"
    evolution_root = workspace / ".bear" / "skill-evolution" / "healthvault"
    responses = {
        "create": {
            "action": "create",
            "skill_name": "HealthVault-Task-Execution",
            "content": SKILL_V1,
            "rationale": "Preserve evidence and validate the final result.",
            "trace_ids": ["trace-create"],
        },
        "evolve": {
            "action": "evolve",
            "skill_name": "HealthVault-Task-Execution",
            "content": SKILL_V2,
            "rationale": "Retry only failures explicitly marked retryable.",
            "trace_ids": ["trace-evolve"],
        },
    }

    async def fake_side_query(prompt: str, _schema: dict[str, object]) -> dict[str, object]:
        action = json.loads(prompt)["action"]
        return dict(responses[action])

    generator = CandidateGenerator(evolution_root, fake_side_query)
    publisher = SkillPublisher(skill_root, evolution_root)
    verifier = CandidateVerifier("HealthVault-Task-Execution")

    create_batch = TaskEndProcessor(generator)
    create_batch.record(EvolutionSignal(
        "trace-create",
        "explicit_correction",
        "Validate citations before returning a result.",
        True,
    ))
    create_candidate = await create_batch.process("create")
    if create_candidate is None:
        raise RuntimeError("Synthetic create batch did not generate a candidate.")
    create_verification = verifier.verify_content(
        create_candidate.content,
        action=create_candidate.action,
    )
    create_release = publisher.publish_candidate(
        create_candidate,
        verification=create_verification,
        approved=True,
    )

    evolve_batch = TaskEndProcessor(generator)
    evolve_batch.record(EvolutionSignal(
        "trace-evolve",
        "deterministic_failure",
        "Retry only after a retryable tool status.",
        True,
    ))
    evolve_candidate = await evolve_batch.process("evolve")
    if evolve_candidate is None:
        raise RuntimeError("Synthetic evolve batch did not generate a candidate.")
    active_v1 = (skill_root / "HealthVault-Task-Execution" / "SKILL.md").read_text(encoding="utf-8")
    evolve_verification = verifier.verify_content(
        evolve_candidate.content,
        action=evolve_candidate.action,
        active_content=active_v1,
    )
    evolve_release = publisher.publish_candidate(
        evolve_candidate,
        verification=evolve_verification,
        approved=True,
    )
    rollback_release = publisher.rollback()
    if not rollback_release.manifest:
        raise RuntimeError("Synthetic rollback did not reload the Skill manifest.")

    summary: dict[str, object] = {
        "schema_version": "healthvault-skill-release-summary-v1",
        "scenario": "fixed-synthetic-create-evolve-rollback",
        "skill_name": "HealthVault-Task-Execution",
        "create": {
            "candidate_id": create_candidate.candidate_id,
            "trace_ids": list(create_candidate.trace_ids),
            "verification": create_verification.status,
            "status": create_release.status,
            "sha256": create_release.active_sha256,
        },
        "evolve": {
            "candidate_id": evolve_candidate.candidate_id,
            "trace_ids": list(evolve_candidate.trace_ids),
            "verification": evolve_verification.status,
            "status": evolve_release.status,
            "sha256": evolve_release.active_sha256,
        },
        "rollback": {
            "status": rollback_release.status,
            "from_sha256": rollback_release.previous_sha256,
            "to_sha256": rollback_release.active_sha256,
        },
        "final_whitelist_manifest": rollback_release.manifest[0],
        "side_query_calls": generator.call_count,
        "api_calls": 0,
        "tokens": 0,
        "cost_usd": 0.0,
    }
    if summary_path is not None:
        target = Path(summary_path)
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_name(f".{target.name}.{uuid.uuid4().hex}.tmp")
        try:
            temporary.write_text(
                json.dumps(summary, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
                encoding="utf-8",
                newline="\n",
            )
            os.replace(temporary, target)
        finally:
            temporary.unlink(missing_ok=True)
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", help="Optional path for the generated JSON summary.")
    arguments = parser.parse_args()
    with tempfile.TemporaryDirectory(prefix="healthvault-task-019-") as workspace:
        summary = asyncio.run(run_synthetic_demo(workspace, summary_path=arguments.summary))
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
