"""Patient/profile split checks shared by development and held-out registries."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any


REQUIRED_SPLIT_FIELDS = frozenset({"patient_id", "profile_id", "split"})


def find_patient_profile_split_conflicts(
    entries: Iterable[Mapping[str, Any]],
) -> list[dict[str, str]]:
    """Return identity conflicts without inspecting prompts, labels, or answers."""
    patient_splits: dict[str, str] = {}
    profile_splits: dict[str, str] = {}
    profile_patients: dict[str, str] = {}
    conflicts: list[dict[str, str]] = []
    for index, entry in enumerate(entries):
        if not isinstance(entry, Mapping) or not REQUIRED_SPLIT_FIELDS <= set(entry):
            raise ValueError("split entry requires patient_id, profile_id, and split")
        patient_id = entry["patient_id"]
        profile_id = entry["profile_id"]
        split = entry["split"]
        if not all(isinstance(value, str) and value.strip() for value in (patient_id, profile_id, split)):
            raise ValueError("split entry fields must be non-empty text")
        for identity_type, identity, known in (
            ("patient", patient_id, patient_splits),
            ("profile", profile_id, profile_splits),
        ):
            previous = known.setdefault(identity, split)
            if previous != split:
                conflicts.append(
                    {
                        "entry": str(index),
                        "identity_type": identity_type,
                        "identity": identity,
                        "first_split": previous,
                        "second_split": split,
                    }
                )
        previous_patient = profile_patients.setdefault(profile_id, patient_id)
        if previous_patient != patient_id:
            conflicts.append(
                {
                    "entry": str(index),
                    "identity_type": "profile_patient_mapping",
                    "identity": profile_id,
                    "first_split": previous_patient,
                    "second_split": patient_id,
                }
            )
    return conflicts


def validate_patient_profile_splits(
    entries: Iterable[Mapping[str, Any]],
) -> None:
    """Reject any patient or profile that crosses named dataset splits."""
    conflicts = find_patient_profile_split_conflicts(entries)
    if conflicts:
        conflict = conflicts[0]
        raise ValueError(
            "patient/profile split conflict: "
            f"{conflict['identity_type']}={conflict['identity']} "
            f"({conflict['first_split']} vs {conflict['second_split']})"
        )
