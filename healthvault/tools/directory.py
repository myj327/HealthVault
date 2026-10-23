"""Single source of truth for HealthVault tool contracts and exposure policy."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from typing import Any, Literal


ToolDependency = Literal["records", "knowledge"]


@dataclass(frozen=True)
class ToolSpec:
    """Immutable metadata used to derive every HealthVault tool surface."""

    name: str
    description: str
    input_schema: dict[str, Any]
    read_only: bool
    dependencies: frozenset[ToolDependency] = frozenset()
    harness: bool = True
    mcp: bool = False
    skill_patch: bool = False
    bind_runtime_profile: bool = False
    network_access: bool = False

    def definition(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "input_schema": deepcopy(self.input_schema),
        }


TOOL_SPECS: tuple[ToolSpec, ...] = (
    ToolSpec(
        name="preview_health_report",
        description=(
            "Parse a synthetic HealthVault text report into a validated, "
            "non-persistent preview with source evidence."
        ),
        input_schema={
            "type": "object",
            "properties": {
                "source_id": {
                    "type": "string",
                    "description": "Stable synthetic source identifier supplied by the caller.",
                },
                "report_text": {
                    "type": "string",
                    "description": (
                        "UTF-8 text content in the documented HealthVault synthetic "
                        "report format."
                    ),
                },
            },
            "required": ["source_id", "report_text"],
        },
        read_only=True,
        mcp=True,
        skill_patch=True,
    ),
    ToolSpec(
        name="confirm_health_records",
        description=(
            "Persist a previously previewed synthetic health report only after "
            "explicit user confirmation and exact SHA verification."
        ),
        input_schema={
            "type": "object",
            "properties": {
                "profile_id": {"type": "string"},
                "source_id": {"type": "string"},
                "report_text": {"type": "string"},
                "expected_sha256": {"type": "string"},
                "user_confirmed": {"type": "boolean"},
            },
            "required": [
                "profile_id", "source_id", "report_text", "expected_sha256",
                "user_confirmed",
            ],
        },
        read_only=False,
        dependencies=frozenset({"records"}),
    ),
    ToolSpec(
        name="query_health_timeline",
        description=(
            "Return the chronological timeline of confirmed health reports and "
            "source evidence for one synthetic profile."
        ),
        input_schema={
            "type": "object",
            "properties": {"profile_id": {"type": "string"}},
            "required": ["profile_id"],
        },
        read_only=True,
        dependencies=frozenset({"records"}),
        mcp=True,
        skill_patch=True,
        bind_runtime_profile=True,
    ),
    ToolSpec(
        name="compare_health_indicators",
        description=(
            "Compare the earliest and latest confirmed values for every health "
            "indicator in one synthetic profile."
        ),
        input_schema={
            "type": "object",
            "properties": {"profile_id": {"type": "string"}},
            "required": ["profile_id"],
        },
        read_only=True,
        dependencies=frozenset({"records"}),
        mcp=True,
        skill_patch=True,
        bind_runtime_profile=True,
    ),
    ToolSpec(
        name="search_health_knowledge",
        description=(
            "Search a local curated health knowledge corpus and return concise "
            "excerpts with official source citations."
        ),
        input_schema={
            "type": "object",
            "properties": {
                "query": {"type": "string"},
                "limit": {"type": "integer", "minimum": 1, "maximum": 5},
            },
            "required": ["query"],
        },
        read_only=True,
        dependencies=frozenset({"knowledge"}),
        mcp=True,
        skill_patch=True,
    ),
    ToolSpec(
        name="search_official_health_web",
        description=(
            "Read-only fallback for general health topics from the approved MedlinePlus/NLM "
            "web service. Use local search_health_knowledge first. The service does not "
            "support current clinical guideline questions or personal medical records."
        ),
        input_schema={
            "type": "object",
            "properties": {
                "question": {"type": "string", "maxLength": 1000},
                "limit": {"type": "integer", "minimum": 1, "maximum": 3},
                "local_answer_insufficient": {
                    "type": "boolean",
                    "description": (
                        "Set true only after a local-evidence answer explicitly reports "
                        "insufficient evidence."
                    ),
                },
            },
            "required": ["question"],
        },
        read_only=True,
        dependencies=frozenset({"knowledge"}),
        mcp=True,
        skill_patch=True,
        network_access=True,
    ),
    ToolSpec(
        name="generate_visit_brief",
        description=(
            "Generate a deterministic visit preparation brief from one confirmed "
            "profile, with record evidence and offline knowledge citations."
        ),
        input_schema={
            "type": "object",
            "properties": {"profile_id": {"type": "string"}},
            "required": ["profile_id"],
        },
        read_only=True,
        dependencies=frozenset({"records", "knowledge"}),
        mcp=True,
        skill_patch=True,
        bind_runtime_profile=True,
    ),
)

_BY_NAME = {spec.name: spec for spec in TOOL_SPECS}
if len(_BY_NAME) != len(TOOL_SPECS):
    raise RuntimeError("HealthVault tool directory contains duplicate names.")


def tool_spec(name: str) -> ToolSpec:
    try:
        return _BY_NAME[name]
    except KeyError as exc:
        raise KeyError(f"Unknown HealthVault tool contract: {name}") from exc


def tool_definition(name: str) -> dict[str, Any]:
    return tool_spec(name).definition()


def tool_names(
    *,
    harness: bool | None = None,
    mcp: bool | None = None,
    skill_patch: bool | None = None,
    read_only: bool | None = None,
) -> tuple[str, ...]:
    specs = TOOL_SPECS
    if harness is not None:
        specs = tuple(spec for spec in specs if spec.harness is harness)
    if mcp is not None:
        specs = tuple(spec for spec in specs if spec.mcp is mcp)
    if skill_patch is not None:
        specs = tuple(spec for spec in specs if spec.skill_patch is skill_patch)
    if read_only is not None:
        specs = tuple(spec for spec in specs if spec.read_only is read_only)
    return tuple(spec.name for spec in specs)


HARNESS_TOOL_NAMES = tool_names(harness=True)
HARNESS_READ_ONLY_TOOL_NAMES = tool_names(harness=True, read_only=True)
HARNESS_LOCAL_READ_ONLY_TOOL_NAMES = tuple(
    spec.name for spec in TOOL_SPECS if spec.harness and spec.read_only and not spec.network_access
)
MCP_TOOL_NAMES = tool_names(mcp=True)
SKILL_PATCH_TOOL_NAMES = tool_names(skill_patch=True)
PROFILE_BOUND_TOOL_NAMES = tuple(
    spec.name for spec in TOOL_SPECS if spec.bind_runtime_profile
)
NETWORK_TOOL_NAMES = tuple(spec.name for spec in TOOL_SPECS if spec.network_access)

