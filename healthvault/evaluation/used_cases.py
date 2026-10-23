"""Cross-track held-out case registry and deterministic leakage checks."""

from __future__ import annotations

import re
import hashlib
import unicodedata
from difflib import SequenceMatcher
from typing import Any


_TOP_FIELDS = {
    "schema_version",
    "registry_version",
    "created_on",
    "normalization",
    "tracks",
    "entries",
}
_TRACK_FIELDS = {"track", "case_version", "purpose"}
_ENTRY_FIELDS = {
    "case_id",
    "case_family_id",
    "track",
    "case_version",
    "split",
    "held_out",
    "query",
    "query_sha256",
    "upstream_dataset",
    "upstream_id",
    "status",
    "source",
}


def normalize_case_text(value: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError("case query must be non-empty text")
    normalized = unicodedata.normalize("NFKC", value).casefold()
    return " ".join(re.findall(r"[\w]+", normalized, flags=re.UNICODE))


def normalized_query_sha256(value: str) -> str:
    return hashlib.sha256(normalize_case_text(value).encode("utf-8")).hexdigest()


def _near_synonym(left: str, right: str) -> bool:
    left_normalized = normalize_case_text(left)
    right_normalized = normalize_case_text(right)
    if left_normalized == right_normalized:
        return True
    left_tokens = set(left_normalized.split())
    right_tokens = set(right_normalized.split())
    union = left_tokens | right_tokens
    jaccard = len(left_tokens & right_tokens) / len(union) if union else 1.0
    ratio = SequenceMatcher(None, left_normalized, right_normalized).ratio()
    return jaccard >= 0.72 or ratio >= 0.78


def find_held_out_collisions(entries: list[dict[str, Any]]) -> list[dict[str, str]]:
    held_out = [entry for entry in entries if entry.get("held_out") is True]
    collisions = []
    for index, left in enumerate(held_out):
        for right in held_out[index + 1 :]:
            if (left.get("track"), left.get("case_version")) == (
                right.get("track"),
                right.get("case_version"),
            ):
                continue
            if left.get("case_family_id") == right.get("case_family_id"):
                reason = "case_family_id"
            elif (
                left.get("upstream_dataset") == right.get("upstream_dataset")
                and left.get("upstream_id") == right.get("upstream_id")
            ):
                reason = "upstream_id"
            elif normalize_case_text(left["query"]) == normalize_case_text(right["query"]):
                reason = "exact_normalized_query"
            elif _near_synonym(left["query"], right["query"]):
                reason = "near_synonym"
            else:
                continue
            collisions.append(
                {
                    "left_case_id": left["case_id"],
                    "right_case_id": right["case_id"],
                    "reason": reason,
                }
            )
    return collisions


def validate_used_case_registry(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != _TOP_FIELDS:
        raise ValueError("used-case registry must use the exact schema")
    if value["schema_version"] != 1 or value["normalization"] != "nfkc-casefold-alnum-v1":
        raise ValueError("unsupported used-case registry schema")
    if not isinstance(value["tracks"], list) or not value["tracks"]:
        raise ValueError("used-case registry tracks must be non-empty")
    track_keys = set()
    for track in value["tracks"]:
        if not isinstance(track, dict) or set(track) != _TRACK_FIELDS:
            raise ValueError("used-case track must use the exact schema")
        if not all(isinstance(track[field], str) and track[field].strip() for field in _TRACK_FIELDS):
            raise ValueError("used-case track fields must be non-empty text")
        key = (track["track"], track["case_version"])
        if key in track_keys:
            raise ValueError("duplicate used-case track/version")
        track_keys.add(key)
    if not isinstance(value["entries"], list):
        raise ValueError("used-case entries must be a list")
    case_ids = set()
    for entry in value["entries"]:
        if not isinstance(entry, dict) or set(entry) != _ENTRY_FIELDS:
            raise ValueError("used-case entry must use the exact schema")
        text_fields = _ENTRY_FIELDS - {"held_out"}
        if not all(isinstance(entry[field], str) and entry[field].strip() for field in text_fields):
            raise ValueError("used-case entry fields must be non-empty text")
        if not isinstance(entry["held_out"], bool):
            raise ValueError("held_out must be boolean")
        digest = entry["query_sha256"]
        if (
            len(digest) != 64
            or any(character not in "0123456789abcdef" for character in digest)
            or digest != normalized_query_sha256(entry["query"])
        ):
            raise ValueError("query_sha256 must bind the normalized query")
        if (entry["track"], entry["case_version"]) not in track_keys:
            raise ValueError("used-case entry references an unknown track/version")
        if entry["case_id"] in case_ids:
            raise ValueError("duplicate used-case ID")
        case_ids.add(entry["case_id"])
        normalize_case_text(entry["query"])
    collisions = find_held_out_collisions(value["entries"])
    if collisions:
        first = collisions[0]
        raise ValueError(
            "held-out collision: "
            f"{first['left_case_id']} vs {first['right_case_id']} ({first['reason']})"
        )
    return value
