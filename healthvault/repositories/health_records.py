"""Repository port for confirmed health reports and records."""

from typing import Protocol

from healthvault.domain.records import (
    ConfirmedReportBundle,
    RepositoryConfirmation,
    TimelineReport,
)


class HealthRecordPersistenceError(RuntimeError):
    """Raised when confirmed records cannot be committed atomically."""


class HealthRecordQueryError(RuntimeError):
    """Raised when confirmed records cannot be queried reliably."""


class HealthRecordRepository(Protocol):
    def confirm_report(
        self,
        bundle: ConfirmedReportBundle,
    ) -> RepositoryConfirmation:
        """Persist or classify one server-validated confirmation bundle."""
        ...

    def list_timeline(self, profile_id: str) -> tuple[TimelineReport, ...]:
        """Return the stable chronological timeline for one exact profile."""
        ...

    def get_report_detail(self, profile_id: str, report_id: str) -> dict[str, object] | None:
        """Return one confirmed report and original-file metadata."""
        ...

    def get_report_asset(self, profile_id: str, report_id: str) -> tuple[bytes, str] | None:
        """Return one profile-bound original file without loading it in timelines."""
        ...
