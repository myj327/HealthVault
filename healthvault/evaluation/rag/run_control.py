"""Run identity, attempt ledger, and hard usage ceilings for RAG evaluation."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from healthvault.rag.chunking import estimate_tokens


_ATTEMPT_FIELDS = {
    "schema_version",
    "run_config_sha256",
    "sequence",
    "call_id",
    "call_type",
    "strategy",
    "query_id",
    "attempt_number",
    "provider",
    "model",
    "model_config_sha256",
    "prompt_sha256",
    "estimated_input_tokens",
    "reserved_input_tokens",
    "reserved_output_tokens",
    "reserved_cost_usd",
    "input_tokens",
    "output_tokens",
    "latency_ms",
    "success",
    "failure",
    "cost_usd",
    "cost_source",
    "usage_status",
    "budget_input_tokens",
    "budget_output_tokens",
    "budget_cost_usd",
}


class BudgetExceededError(RuntimeError):
    """Raised before a request that could exceed a configured run ceiling."""


@dataclass(frozen=True)
class RunLimits:
    max_attempts: int
    max_input_tokens: int
    max_output_tokens: int
    max_cost_usd: float

    def __post_init__(self) -> None:
        if (
            self.max_attempts < 0
            or self.max_input_tokens < 0
            or self.max_output_tokens < 0
            or self.max_cost_usd < 0
        ):
            raise ValueError("Run limits cannot be negative")


@dataclass(frozen=True)
class AttemptReservation:
    """Conservative budget charged to one transport attempt."""

    estimated_input_tokens: int
    input_tokens: int
    output_tokens: int
    cost_usd: float


# 80 generation + 80 Judge calls, each with at most one retry.  The output
# ceiling reserves 80 * (256 + 128) * 2 tokens.  Input and cost are deliberately
# conservative global stops; they are rechecked before every network attempt.
DEFAULT_REAL_RUN_LIMITS = RunLimits(
    max_attempts=320,
    max_input_tokens=200_000,
    max_output_tokens=61_440,
    max_cost_usd=0.30,
)


def canonical_json_bytes(value: Any) -> bytes:
    return (
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")


def model_config_dict(config: Any) -> dict[str, Any]:
    return asdict(config)


def _compatibility_payload(manifest: dict[str, Any]) -> dict[str, Any]:
    return {
        key: value
        for key, value in manifest.items()
        if key not in {"schema_version", "created_at", "run_config_sha256"}
    }


def _config_hash(manifest: dict[str, Any]) -> str:
    payload = json.dumps(
        _compatibility_payload(manifest),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def build_run_manifest(
    *,
    eval_version: str,
    dataset_sha256: str,
    frozen_test_sha256: str,
    corpus_sha256: str,
    artifact_sha256: dict[str, str],
    aliases_sha256: str,
    query_ids: tuple[str, ...],
    generation_config: Any,
    judge_config: Any,
    max_retries: int,
    limits: RunLimits,
    implementation_provenance: dict[str, Any],
    created_at: str,
    run_type: str = "offline_mock",
    generation_prompt_version: str = "rag-generation-r1-v1",
    judge_prompt_version: str = "rag-judge-r1-v1",
) -> dict[str, Any]:
    provenance_fields = {
        "schema_version",
        "current_head",
        "worktree_dirty",
        "evaluation_source_sha256",
        "evaluation_source_files",
        "prompt_sha256",
        "validator_sha256",
    }
    if set(implementation_provenance) != provenance_fields:
        raise ValueError("implementation provenance must use the exact schema")
    source_files = implementation_provenance["evaluation_source_files"]
    if (
        implementation_provenance["schema_version"] != 1
        or not re.fullmatch(r"[0-9a-f]{40}", implementation_provenance["current_head"])
        or not isinstance(implementation_provenance["worktree_dirty"], bool)
        or not isinstance(source_files, dict)
        or not source_files
        or any(
            not isinstance(path, str)
            or not path
            or not isinstance(digest, str)
            or not re.fullmatch(r"[0-9a-f]{64}", digest)
            for path, digest in source_files.items()
        )
        or any(
            not isinstance(implementation_provenance[field], str)
            or not re.fullmatch(r"[0-9a-f]{64}", implementation_provenance[field])
            for field in (
                "evaluation_source_sha256",
                "prompt_sha256",
                "validator_sha256",
            )
        )
    ):
        raise ValueError("implementation provenance is invalid")
    manifest = {
        "schema_version": 2,
        "created_at": created_at,
        "run_type": run_type,
        "eval_version": eval_version,
        "dataset_sha256": dataset_sha256,
        "frozen_test_sha256": frozen_test_sha256,
        "corpus_sha256": corpus_sha256,
        "artifact_sha256": dict(artifact_sha256),
        "aliases_sha256": aliases_sha256,
        "query_ids": list(query_ids),
        "strategies": ["natural_section", "hybrid_v1"],
        "prompt_versions": {
            "generation": generation_prompt_version,
            "judge": judge_prompt_version,
        },
        "generation_config": model_config_dict(generation_config),
        "judge_config": model_config_dict(judge_config),
        "max_retries": max_retries,
        "limits": asdict(limits),
        "implementation_provenance": json.loads(
            json.dumps(implementation_provenance, sort_keys=True)
        ),
    }
    manifest["run_config_sha256"] = _config_hash(manifest)
    return manifest


def ensure_run_manifest(path: str | Path, expected: dict[str, Any]) -> dict[str, Any]:
    manifest_path = Path(path)
    if expected.get("run_config_sha256") != _config_hash(expected):
        raise ValueError("incompatible run manifest: invalid expected config hash")
    if manifest_path.exists():
        existing = json.loads(manifest_path.read_text(encoding="utf-8"))
        if (
            existing.get("run_config_sha256") != _config_hash(existing)
            or existing.get("run_config_sha256") != expected["run_config_sha256"]
        ):
            raise ValueError("incompatible run manifest")
        return existing
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_bytes(canonical_json_bytes(expected))
    return expected


class AttemptLedger:
    """Append-only, attempt-level accounting bound to one run config hash."""

    def __init__(
        self,
        path: str | Path,
        limits: RunLimits,
        run_config_sha256: str,
    ) -> None:
        self.path = Path(path)
        self.limits = limits
        self.run_config_sha256 = run_config_sha256
        self.records = self._load()
        if any(
            record.get("run_config_sha256") != run_config_sha256
            for record in self.records
        ):
            raise ValueError("incompatible attempt ledger")
        totals = self.totals()
        if (
            totals["attempts"] > limits.max_attempts
            or totals["budget_input_tokens"] > limits.max_input_tokens
            or totals["budget_output_tokens"] > limits.max_output_tokens
            or totals["budget_cost_usd"] > limits.max_cost_usd
        ):
            raise ValueError("attempt ledger already exceeds run limits")

    def _load(self) -> list[dict[str, Any]]:
        if not self.path.exists():
            return []
        records = [
            json.loads(line)
            for line in self.path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        seen = set()
        for sequence, record in enumerate(records, 1):
            if not isinstance(record, dict) or set(record) != _ATTEMPT_FIELDS:
                raise ValueError("attempt ledger record must use the exact schema")
            numeric = (
                "estimated_input_tokens",
                "reserved_input_tokens",
                "reserved_output_tokens",
                "latency_ms",
                "budget_input_tokens",
                "budget_output_tokens",
            )
            if (
                record["schema_version"] != 2
                or record["sequence"] != sequence
                or not isinstance(record["attempt_number"], int)
                or record["attempt_number"] < 1
                or any(
                    not isinstance(record[field], int) or record[field] < 0
                    for field in numeric
                )
                or any(
                    record[field] is not None
                    and (not isinstance(record[field], int) or record[field] < 0)
                    for field in ("input_tokens", "output_tokens")
                )
                or record["cost_usd"] is not None
                and (
                    not isinstance(record["cost_usd"], (int, float))
                    or isinstance(record["cost_usd"], bool)
                    or record["cost_usd"] < 0
                )
                or any(
                    not isinstance(record[field], (int, float))
                    or isinstance(record[field], bool)
                    or record[field] < 0
                    for field in ("reserved_cost_usd", "budget_cost_usd")
                )
                or not isinstance(record["success"], bool)
                or (record["success"] and record["failure"] is not None)
                or (not record["success"] and not isinstance(record["failure"], str))
                or record["usage_status"] not in {"actual", "conservative_unknown"}
                or (
                    record["usage_status"] == "actual"
                    and any(
                        record[field] is None
                        for field in ("input_tokens", "output_tokens", "cost_usd")
                    )
                )
                or (
                    record["usage_status"] == "conservative_unknown"
                    and any(
                        record[field] is not None
                        for field in ("input_tokens", "output_tokens", "cost_usd")
                    )
                )
                or record["budget_input_tokens"] < record["reserved_input_tokens"]
                or record["budget_output_tokens"] < record["reserved_output_tokens"]
                or record["budget_cost_usd"] < record["reserved_cost_usd"]
                or (
                    record["input_tokens"] is not None
                    and record["budget_input_tokens"] < record["input_tokens"]
                )
                or (
                    record["output_tokens"] is not None
                    and record["budget_output_tokens"] < record["output_tokens"]
                )
                or (
                    record["cost_usd"] is not None
                    and record["budget_cost_usd"] < record["cost_usd"]
                )
            ):
                raise ValueError("attempt ledger record is invalid")
            key = (record["call_id"], record["attempt_number"])
            if key in seen:
                raise ValueError("attempt ledger contains a duplicate attempt")
            seen.add(key)
        return records

    def totals(self) -> dict[str, int | float]:
        return {
            "attempts": len(self.records),
            "input_tokens": sum(record["input_tokens"] or 0 for record in self.records),
            "output_tokens": sum(record["output_tokens"] or 0 for record in self.records),
            "cost_usd": round(
                sum(record["cost_usd"] or 0.0 for record in self.records), 10
            ),
            "unknown_usage_attempts": sum(
                record["usage_status"] == "conservative_unknown"
                for record in self.records
            ),
            "budget_input_tokens": sum(
                record["budget_input_tokens"] for record in self.records
            ),
            "budget_output_tokens": sum(
                record["budget_output_tokens"] for record in self.records
            ),
            "budget_cost_usd": round(
                sum(record["budget_cost_usd"] for record in self.records), 10
            ),
        }

    def preflight(self, prompt: str, config: Any) -> AttemptReservation:
        totals = self.totals()
        estimated_input = estimate_tokens(prompt)
        reserved_input = max(estimated_input * 2, estimated_input + 64)
        reserved_output = config.max_output_tokens
        reserved_cost = (
            reserved_input * config.input_cost_per_million
            + config.max_output_tokens * config.output_cost_per_million
        ) / 1_000_000
        if totals["attempts"] + 1 > self.limits.max_attempts:
            raise BudgetExceededError("maximum attempt count would be exceeded")
        if totals["budget_input_tokens"] + reserved_input > self.limits.max_input_tokens:
            raise BudgetExceededError("maximum input tokens would be exceeded")
        if totals["budget_output_tokens"] + reserved_output > self.limits.max_output_tokens:
            raise BudgetExceededError("maximum output tokens would be exceeded")
        if totals["budget_cost_usd"] + reserved_cost > self.limits.max_cost_usd:
            raise BudgetExceededError("maximum cost would be exceeded")
        return AttemptReservation(
            estimated_input_tokens=estimated_input,
            input_tokens=reserved_input,
            output_tokens=reserved_output,
            cost_usd=round(reserved_cost, 10),
        )

    def append(self, record: dict[str, Any]) -> None:
        if record.get("run_config_sha256") != self.run_config_sha256:
            raise ValueError("attempt record is not bound to this run")
        if set(record) != _ATTEMPT_FIELDS:
            raise ValueError("attempt ledger record must use the exact schema")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8", newline="\n") as handle:
            handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
        self.records.append(record)
        self.records = self._load()
        totals = self.totals()
        exceeded = []
        if totals["attempts"] > self.limits.max_attempts:
            exceeded.append("attempt count")
        if totals["budget_input_tokens"] > self.limits.max_input_tokens:
            exceeded.append("input tokens")
        if totals["budget_output_tokens"] > self.limits.max_output_tokens:
            exceeded.append("output tokens")
        if totals["budget_cost_usd"] > self.limits.max_cost_usd:
            exceeded.append("cost")
        if exceeded:
            raise BudgetExceededError(
                "actual usage recorded; maximum " + ", ".join(exceeded) + " exceeded"
            )
