"""Application-owned request context for HealthVault tool execution."""

from dataclasses import dataclass


@dataclass(frozen=True)
class RuntimeContext:
    """Carry synthetic partition and request keys outside model control."""

    profile_id: str
    request_id: str

    def __post_init__(self) -> None:
        if not isinstance(self.profile_id, str) or not self.profile_id.strip():
            raise ValueError("RuntimeContext profile_id must be non-empty.")
        if not isinstance(self.request_id, str) or not self.request_id.strip():
            raise ValueError("RuntimeContext request_id must be non-empty.")
