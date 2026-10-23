"""Explicit model protocol and credential mapping for HealthVault chat."""

from __future__ import annotations

import os
from dataclasses import dataclass, field


OPENAI_COMPATIBLE = "openai-compatible"
ANTHROPIC = "anthropic"
SUPPORTED_PROVIDERS = (OPENAI_COMPATIBLE, ANTHROPIC)
_KEY_ENVIRONMENT = {
    OPENAI_COMPATIBLE: "HEALTHVAULT_OPENAI_API_KEY",
    ANTHROPIC: "HEALTHVAULT_ANTHROPIC_API_KEY",
}


class HealthVaultModelConfigError(ValueError):
    """Raised before model construction when explicit chat config is invalid."""


@dataclass(frozen=True)
class HealthVaultModelConfig:
    provider: str
    model: str
    api_base: str
    api_key: str = field(repr=False)
    reasoning_effort: str | None = None

    def __post_init__(self) -> None:
        if self.provider not in SUPPORTED_PROVIDERS:
            raise HealthVaultModelConfigError(
                "HealthVault provider must be openai-compatible or anthropic."
            )
        for field_name in ("model", "api_base", "api_key"):
            value = getattr(self, field_name)
            if not isinstance(value, str) or not value.strip():
                raise HealthVaultModelConfigError(
                    f"HealthVault {field_name} must be non-empty."
                )
        if self.reasoning_effort is not None and (
            self.provider != OPENAI_COMPATIBLE
            or not isinstance(self.reasoning_effort, str)
            or self.reasoning_effort not in {"none", "low", "medium", "high", "xhigh", "max"}
        ):
            raise HealthVaultModelConfigError("HealthVault reasoning_effort is invalid for this provider.")


def load_healthvault_model_config(
    *,
    provider: str,
    model: str,
    api_base: str,
    reasoning_effort: str | None = None,
) -> HealthVaultModelConfig:
    """Read only the provider-specific HealthVault key environment variable."""
    if provider not in SUPPORTED_PROVIDERS:
        raise HealthVaultModelConfigError(
            "HealthVault provider must be openai-compatible or anthropic."
        )
    environment_name = _KEY_ENVIRONMENT[provider]
    api_key = os.environ.get(environment_name, "").strip()
    if not api_key:
        raise HealthVaultModelConfigError(
            f"HealthVault API key is required in {environment_name}."
        )
    return HealthVaultModelConfig(
        provider=provider,
        model=model,
        api_base=api_base,
        api_key=api_key,
        reasoning_effort=reasoning_effort,
    )
