"""TASK-042 development-only safety evidence and split validation."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from healthvault.evaluation.split_integrity import validate_patient_profile_splits


ROOT = Path(__file__).resolve().parents[3]
E2E_ROOT = ROOT / "healthvault" / "evaluation" / "e2e"
DEV_SPLIT_REGISTRY_PATH = E2E_ROOT / "task-042-dev-patient-profile-splits.json"
SAFETY_MATRIX_PATH = E2E_ROOT / "task-042-safety-matrix.json"
BADCASE_PATH = E2E_ROOT / "task-042-badcases.md"


def load_development_split_registry(
    path: Path = DEV_SPLIT_REGISTRY_PATH,
) -> dict[str, Any]:
    """Load only development identities; no test case, prompt, or Gold is read."""
    value = json.loads(path.read_text(encoding="utf-8"))
    if set(value) != {"schema_version", "purpose", "entries"}:
        raise ValueError("TASK-042 split registry schema is invalid")
    if value["schema_version"] != "healthvault-patient-profile-splits-v1":
        raise ValueError("TASK-042 split registry version is invalid")
    if value["purpose"] != "development_only" or not isinstance(value["entries"], list):
        raise ValueError("TASK-042 split registry is not development-only")
    validate_patient_profile_splits(value["entries"])
    return value
