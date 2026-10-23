"""SQLite persistence for explicitly confirmed health records."""

from __future__ import annotations

import json
import hashlib
import sqlite3
from contextlib import closing
from datetime import datetime, timezone
from pathlib import Path

from healthvault.domain.records import (
    ConfirmedClinicalItem,
    ConfirmedImagingFinding,
    ConfirmedIndicator,
    ConfirmedReportBundle,
    RepositoryConfirmation,
    TimelineReport,
)
from healthvault.repositories.health_records import (
    HealthRecordPersistenceError,
    HealthRecordQueryError,
)


SCHEMA = """
CREATE TABLE IF NOT EXISTS confirmed_reports (
    id INTEGER PRIMARY KEY,
    idempotency_key TEXT NOT NULL UNIQUE,
    profile_id TEXT NOT NULL,
    source_id TEXT NOT NULL,
    source_sha256 TEXT NOT NULL,
    report_id TEXT NOT NULL,
    report_date TEXT NOT NULL,
    facility TEXT NOT NULL,
    raw_report_text TEXT NOT NULL,
    confirmed_at TEXT NOT NULL,
    UNIQUE(profile_id, source_id),
    UNIQUE(profile_id, report_id)
);

CREATE TABLE IF NOT EXISTS health_records (
    id INTEGER PRIMARY KEY,
    confirmed_report_id INTEGER NOT NULL,
    indicator_code TEXT NOT NULL,
    display_name TEXT NOT NULL,
    value REAL NOT NULL,
    unit TEXT NOT NULL,
    reference_lower REAL NOT NULL,
    reference_upper REAL NOT NULL,
    flag TEXT NOT NULL,
    evidence_source_id TEXT NOT NULL,
    evidence_line_start INTEGER NOT NULL,
    evidence_line_end INTEGER NOT NULL,
    evidence_quote TEXT NOT NULL,
    FOREIGN KEY(confirmed_report_id) REFERENCES confirmed_reports(id),
    UNIQUE(confirmed_report_id, indicator_code)
);

CREATE TABLE IF NOT EXISTS clinical_items (
    id INTEGER PRIMARY KEY,
    confirmed_report_id INTEGER NOT NULL,
    clinical_item_id TEXT NOT NULL,
    kind TEXT NOT NULL,
    original_text TEXT NOT NULL,
    diagnosis_code TEXT,
    normalized_value TEXT,
    evidence_source_id TEXT NOT NULL,
    evidence_line_start INTEGER NOT NULL,
    evidence_line_end INTEGER NOT NULL,
    evidence_span_start INTEGER NOT NULL,
    evidence_span_end INTEGER NOT NULL,
    evidence_quote TEXT NOT NULL,
    evidence_page INTEGER,
    evidence_polygon_json TEXT,
    confidence REAL,
    source_type TEXT NOT NULL,
    recorded_by_provider INTEGER NOT NULL,
    confirmation_status TEXT NOT NULL,
    date_status TEXT,
    normalized_date TEXT,
    date_original TEXT,
    FOREIGN KEY(confirmed_report_id) REFERENCES confirmed_reports(id),
    UNIQUE(confirmed_report_id, clinical_item_id)
);

CREATE TABLE IF NOT EXISTS imaging_findings (
    id INTEGER PRIMARY KEY,
    confirmed_report_id INTEGER NOT NULL,
    finding_id TEXT NOT NULL,
    finding_type TEXT NOT NULL,
    body_site TEXT NOT NULL,
    laterality TEXT NOT NULL,
    location TEXT NOT NULL,
    dimensions_mm_json TEXT NOT NULL,
    original_dimensions TEXT NOT NULL,
    original_unit TEXT NOT NULL,
    evidence_source_id TEXT NOT NULL,
    evidence_line_start INTEGER NOT NULL,
    evidence_quote TEXT NOT NULL,
    FOREIGN KEY(confirmed_report_id) REFERENCES confirmed_reports(id),
    UNIQUE(confirmed_report_id, finding_id)
);

CREATE TABLE IF NOT EXISTS report_assets (
    confirmed_report_id INTEGER PRIMARY KEY,
    filename TEXT NOT NULL,
    media_type TEXT NOT NULL,
    sha256 TEXT NOT NULL,
    content BLOB NOT NULL,
    FOREIGN KEY(confirmed_report_id) REFERENCES confirmed_reports(id)
);
"""


class SQLiteHealthRecordRepository:
    """Persist confirmed reports in an explicitly supplied SQLite database."""

    def __init__(self, database_path: str | Path) -> None:
        self.database_path = Path(database_path)
        connection = self._connect()
        try:
            connection.executescript(SCHEMA)
            for table in ("health_records", "imaging_findings"):
                if "evidence_locator_json" not in {row["name"] for row in connection.execute(f"PRAGMA table_info({table})")}:
                    connection.execute(f"ALTER TABLE {table} ADD COLUMN evidence_locator_json TEXT")
            connection.commit()
        finally:
            connection.close()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(str(self.database_path))
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        return connection
    @staticmethod
    def _utc_timestamp() -> str:
        return (
            datetime.now(timezone.utc)
            .isoformat(timespec="seconds")
            .replace("+00:00", "Z")
        )

    @staticmethod
    def _stored_outcome(
        connection: sqlite3.Connection,
        row: sqlite3.Row,
    ) -> RepositoryConfirmation:
        count_row = connection.execute(
            """
            SELECT COUNT(*) AS indicator_count
            FROM health_records
            WHERE confirmed_report_id = ?
            """,
            (row["id"],),
        ).fetchone()
        clinical_count_row = connection.execute(
            """
            SELECT COUNT(*) AS clinical_item_count
            FROM clinical_items
            WHERE confirmed_report_id = ?
            """,
            (row["id"],),
        ).fetchone()
        return RepositoryConfirmation(
            status="already_confirmed",
            write_performed=False,
            profile_id=row["profile_id"],
            source_id=row["source_id"],
            source_sha256=row["source_sha256"],
            report_id=row["report_id"],
            report_date=row["report_date"],
            indicator_count=count_row["indicator_count"],
            clinical_item_count=clinical_count_row["clinical_item_count"],
        )

    @staticmethod
    def _conflict_outcome(
        bundle: ConfirmedReportBundle,
        status: str,
    ) -> RepositoryConfirmation:
        return RepositoryConfirmation(
            status=status,
            write_performed=False,
            profile_id=bundle.profile_id,
            source_id=bundle.source_id,
            source_sha256=bundle.source_sha256,
            report_id=bundle.report_id,
            report_date=bundle.report_date,
            indicator_count=len(bundle.indicators),
            clinical_item_count=len(bundle.clinical_items),
        )

    def confirm_report(
        self,
        bundle: ConfirmedReportBundle,
    ) -> RepositoryConfirmation:
        asset = bundle.original_asset
        if asset is not None and (
            asset.media_type not in {"image/png", "image/jpeg", "application/pdf"}
            or not asset.content
            or len(asset.content) > 8 * 1024 * 1024
            or hashlib.sha256(asset.content).hexdigest() != asset.sha256
        ):
            raise HealthRecordPersistenceError("Invalid original report asset.")
        connection = self._connect()
        try:
            connection.execute("BEGIN IMMEDIATE")

            idempotent_row = connection.execute(
                "SELECT * FROM confirmed_reports WHERE idempotency_key = ?",
                (bundle.idempotency_key,),
            ).fetchone()
            if idempotent_row is not None:
                exact_match = (
                    idempotent_row["profile_id"] == bundle.profile_id
                    and idempotent_row["source_id"] == bundle.source_id
                    and idempotent_row["source_sha256"] == bundle.source_sha256
                    and idempotent_row["report_id"] == bundle.report_id
                )
                if exact_match:
                    outcome = self._stored_outcome(connection, idempotent_row)
                elif (
                    idempotent_row["profile_id"] == bundle.profile_id
                    and idempotent_row["source_id"] == bundle.source_id
                ):
                    outcome = self._conflict_outcome(bundle, "source_conflict")
                else:
                    raise HealthRecordPersistenceError(
                        "Idempotency key collision does not match stored identity."
                    )
                connection.commit()
                return outcome

            source_row = connection.execute(
                """
                SELECT * FROM confirmed_reports
                WHERE profile_id = ? AND source_id = ?
                """,
                (bundle.profile_id, bundle.source_id),
            ).fetchone()
            if source_row is not None:
                if (
                    source_row["source_sha256"] == bundle.source_sha256
                    and source_row["report_id"] == bundle.report_id
                ):
                    outcome = self._stored_outcome(connection, source_row)
                else:
                    outcome = self._conflict_outcome(bundle, "source_conflict")
                connection.commit()
                return outcome

            report_row = connection.execute(
                """
                SELECT id FROM confirmed_reports
                WHERE profile_id = ? AND report_id = ?
                """,
                (bundle.profile_id, bundle.report_id),
            ).fetchone()
            if report_row is not None:
                outcome = self._conflict_outcome(bundle, "report_conflict")
                connection.commit()
                return outcome

            report_cursor = connection.execute(
                """
                INSERT INTO confirmed_reports (
                    idempotency_key,
                    profile_id,
                    source_id,
                    source_sha256,
                    report_id,
                    report_date,
                    facility,
                    raw_report_text,
                    confirmed_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    bundle.idempotency_key,
                    bundle.profile_id,
                    bundle.source_id,
                    bundle.source_sha256,
                    bundle.report_id,
                    bundle.report_date,
                    bundle.facility,
                    bundle.raw_report_text,
                    self._utc_timestamp(),
                ),
            )
            confirmed_report_id = report_cursor.lastrowid
            if asset is not None:
                connection.execute(
                    "INSERT INTO report_assets (confirmed_report_id, filename, media_type, sha256, content) "
                    "VALUES (?, ?, ?, ?, ?)",
                    (confirmed_report_id, Path(asset.filename).name, asset.media_type, asset.sha256, asset.content),
                )
            for indicator in bundle.indicators:
                connection.execute(
                    """
                    INSERT INTO health_records (
                        confirmed_report_id,
                        indicator_code,
                        display_name,
                        value,
                        unit,
                        reference_lower,
                        reference_upper,
                        flag,
                        evidence_source_id,
                        evidence_line_start,
                        evidence_line_end,
                        evidence_quote,
                        evidence_locator_json
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        confirmed_report_id,
                        indicator.code,
                        indicator.display_name,
                        indicator.value,
                        indicator.unit,
                        indicator.reference_lower,
                        indicator.reference_upper,
                        indicator.flag,
                        indicator.evidence_source_id,
                        indicator.evidence_line_start,
                        indicator.evidence_line_end,
                        indicator.evidence_quote,
                        json.dumps(indicator.evidence_locator, ensure_ascii=False) if indicator.evidence_locator else None,
                    ),
                )

            for item in bundle.clinical_items:
                connection.execute(
                    """
                    INSERT INTO clinical_items (
                        confirmed_report_id,
                        clinical_item_id,
                        kind,
                        original_text,
                        diagnosis_code,
                        normalized_value,
                        evidence_source_id,
                        evidence_line_start,
                        evidence_line_end,
                        evidence_span_start,
                        evidence_span_end,
                        evidence_quote,
                        evidence_page,
                        evidence_polygon_json,
                        confidence,
                        source_type,
                        recorded_by_provider,
                        confirmation_status,
                        date_status,
                        normalized_date,
                        date_original
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        confirmed_report_id,
                        item.clinical_item_id,
                        item.kind,
                        item.original_text,
                        item.diagnosis_code,
                        item.normalized_value,
                        item.evidence_source_id,
                        item.evidence_line_start,
                        item.evidence_line_end,
                        item.evidence_span_start,
                        item.evidence_span_end,
                        item.evidence_quote,
                        item.evidence_page,
                        json.dumps(item.evidence_polygon, ensure_ascii=False)
                        if item.evidence_polygon is not None
                        else None,
                        item.confidence,
                        item.source_type,
                        int(item.recorded_by_provider),
                        item.confirmation_status,
                        item.date_status,
                        item.normalized_date,
                        item.date_original,
                    ),
                )

            for finding in bundle.imaging_findings:
                connection.execute(
                    """INSERT INTO imaging_findings (
                        confirmed_report_id, finding_id, finding_type, body_site,
                        laterality, location, dimensions_mm_json,
                        original_dimensions, original_unit, evidence_source_id,
                        evidence_line_start, evidence_quote, evidence_locator_json
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    (
                        confirmed_report_id, finding.finding_id, finding.finding_type,
                        finding.body_site, finding.laterality, finding.location,
                        json.dumps(finding.dimensions_mm), finding.original_dimensions,
                        finding.original_unit, finding.evidence_source_id,
                        finding.evidence_line_start, finding.evidence_quote,
                        json.dumps(finding.evidence_locator, ensure_ascii=False) if finding.evidence_locator else None,
                    ),
                )

            connection.commit()
            return RepositoryConfirmation(
                status="confirmed",
                write_performed=True,
                profile_id=bundle.profile_id,
                source_id=bundle.source_id,
                source_sha256=bundle.source_sha256,
                report_id=bundle.report_id,
                report_date=bundle.report_date,
                indicator_count=len(bundle.indicators),
                clinical_item_count=len(bundle.clinical_items),
            )
        except sqlite3.Error as error:
            connection.rollback()
            raise HealthRecordPersistenceError(
                "Failed to persist confirmed health records atomically."
            ) from error
        finally:
            connection.close()

    def get_report_detail(self, profile_id: str, report_id: str) -> dict[str, object] | None:
        with closing(self._connect()) as connection:
            row = connection.execute(
                """SELECT reports.report_id, reports.report_date, reports.facility,
                          reports.raw_report_text, reports.source_id,
                          assets.filename, assets.media_type, assets.sha256
                   FROM confirmed_reports AS reports
                   LEFT JOIN report_assets AS assets ON assets.confirmed_report_id = reports.id
                   WHERE reports.profile_id = ? AND reports.report_id = ?""",
                (profile_id, report_id),
            ).fetchone()
        if row is None:
            return None
        return {
            "report_id": row["report_id"],
            "report_date": row["report_date"],
            "facility": row["facility"],
            "source_id": row["source_id"],
            "report_text": row["raw_report_text"],
            "original_asset": (
                {"filename": row["filename"], "media_type": row["media_type"], "sha256": row["sha256"]}
                if row["media_type"] else None
            ),
        }

    def get_report_asset(self, profile_id: str, report_id: str) -> tuple[bytes, str] | None:
        with closing(self._connect()) as connection:
            row = connection.execute(
                """SELECT assets.content, assets.media_type
                   FROM report_assets AS assets
                   JOIN confirmed_reports AS reports ON reports.id = assets.confirmed_report_id
                   WHERE reports.profile_id = ? AND reports.report_id = ?""",
                (profile_id, report_id),
            ).fetchone()
        return (bytes(row["content"]), row["media_type"]) if row else None

    def list_timeline(self, profile_id: str) -> tuple[TimelineReport, ...]:
        """Read one profile with a single parameterized, ordered JOIN query."""
        connection: sqlite3.Connection | None = None
        try:
            connection = self._connect()
            rows = connection.execute(
                """
                SELECT
                    reports.id AS report_database_id,
                    reports.report_id,
                    reports.report_date,
                    reports.facility,
                    reports.source_id,
                    reports.source_sha256,
                    records.indicator_code,
                    records.display_name,
                    records.value,
                    records.unit,
                    records.reference_lower,
                    records.reference_upper,
                    records.flag,
                    records.evidence_source_id,
                    records.evidence_line_start,
                    records.evidence_line_end,
                    records.evidence_quote, records.*
                FROM confirmed_reports AS reports
                LEFT JOIN health_records AS records
                    ON records.confirmed_report_id = reports.id
                WHERE reports.profile_id = ?
                ORDER BY
                    reports.report_date ASC,
                    reports.report_id ASC,
                    reports.source_id ASC,
                    records.indicator_code ASC
                """,
                (profile_id,),
            ).fetchall()

            clinical_rows = connection.execute(
                """
                SELECT *
                FROM clinical_items
                WHERE confirmed_report_id IN (
                    SELECT id FROM confirmed_reports WHERE profile_id = ?
                )
                ORDER BY confirmed_report_id ASC, evidence_line_start ASC, id ASC
                """,
                (profile_id,),
            ).fetchall()
            imaging_rows = connection.execute(
                """SELECT * FROM imaging_findings
                WHERE confirmed_report_id IN (
                    SELECT id FROM confirmed_reports WHERE profile_id = ?
                ) ORDER BY confirmed_report_id ASC, evidence_line_start ASC, id ASC""",
                (profile_id,),
            ).fetchall()
            imaging_by_report: dict[int, list[ConfirmedImagingFinding]] = {}
            for row in imaging_rows:
                imaging_by_report.setdefault(row["confirmed_report_id"], []).append(
                    ConfirmedImagingFinding(
                        finding_id=row["finding_id"],
                        finding_type=row["finding_type"],
                        body_site=row["body_site"],
                        laterality=row["laterality"],
                        location=row["location"],
                        dimensions_mm=tuple(json.loads(row["dimensions_mm_json"])),
                        original_dimensions=row["original_dimensions"],
                        original_unit=row["original_unit"],
                        evidence_source_id=row["evidence_source_id"],
                        evidence_line_start=row["evidence_line_start"],
                        evidence_quote=row["evidence_quote"],
                        evidence_locator=json.loads(row["evidence_locator_json"]) if "evidence_locator_json" in row.keys() and row["evidence_locator_json"] else None,
                    )
                )
            clinical_by_report: dict[int, list[ConfirmedClinicalItem]] = {}
            for row in clinical_rows:
                polygon = (
                    json.loads(row["evidence_polygon_json"])
                    if row["evidence_polygon_json"] is not None
                    else None
                )
                clinical_by_report.setdefault(row["confirmed_report_id"], []).append(
                    ConfirmedClinicalItem(
                        clinical_item_id=row["clinical_item_id"],
                        kind=row["kind"],
                        original_text=row["original_text"],
                        diagnosis_code=row["diagnosis_code"],
                        normalized_value=row["normalized_value"],
                        evidence_source_id=row["evidence_source_id"],
                        evidence_line_start=row["evidence_line_start"],
                        evidence_line_end=row["evidence_line_end"],
                        evidence_span_start=row["evidence_span_start"],
                        evidence_span_end=row["evidence_span_end"],
                        evidence_quote=row["evidence_quote"],
                        evidence_page=row["evidence_page"],
                        evidence_polygon=polygon,
                        confidence=row["confidence"],
                        source_type=row["source_type"],
                        recorded_by_provider=bool(row["recorded_by_provider"]),
                        confirmation_status=row["confirmation_status"],
                        date_status=row["date_status"],
                        normalized_date=row["normalized_date"],
                        date_original=row["date_original"],
                    )
                )

            reports: list[TimelineReport] = []
            current_database_id: int | None = None
            current_row: sqlite3.Row | None = None
            current_indicators: list[ConfirmedIndicator] = []

            def append_current_report() -> None:
                if current_row is None:
                    return
                reports.append(
                    TimelineReport(
                        report_id=current_row["report_id"],
                        report_date=current_row["report_date"],
                        facility=current_row["facility"],
                        source_id=current_row["source_id"],
                        source_sha256=current_row["source_sha256"],
                        indicators=tuple(current_indicators),
                        clinical_items=tuple(
                            clinical_by_report.get(current_row["report_database_id"], [])
                        ),
                        imaging_findings=tuple(
                            imaging_by_report.get(current_row["report_database_id"], [])
                        ),
                    )
                )

            for row in rows:
                database_id = row["report_database_id"]
                if database_id != current_database_id:
                    append_current_report()
                    current_database_id = database_id
                    current_row = row
                    current_indicators = []
                if row["indicator_code"] is not None:
                    current_indicators.append(
                        ConfirmedIndicator(
                            code=row["indicator_code"],
                            display_name=row["display_name"],
                            value=row["value"],
                            unit=row["unit"],
                            reference_lower=row["reference_lower"],
                            reference_upper=row["reference_upper"],
                            flag=row["flag"],
                            evidence_source_id=row["evidence_source_id"],
                            evidence_line_start=row["evidence_line_start"],
                            evidence_line_end=row["evidence_line_end"],
                            evidence_quote=row["evidence_quote"],
                            evidence_locator=json.loads(row["evidence_locator_json"]) if "evidence_locator_json" in row.keys() and row["evidence_locator_json"] else None,
                        )
                    )
            append_current_report()
            return tuple(reports)
        except sqlite3.Error as error:
            raise HealthRecordQueryError(
                "Failed to query the confirmed health timeline."
            ) from error
        finally:
            if connection is not None:
                connection.close()


class ReadOnlySQLiteHealthRecordRepository(SQLiteHealthRecordRepository):
    """Expose record queries through SQLite's enforced read-only URI mode."""

    def __init__(self, database_path: str | Path) -> None:
        self.database_path = Path(database_path)
        if not self.database_path.is_file():
            raise FileNotFoundError("Health record database must already exist.")
        connection = self._connect()
        try:
            connection.execute("SELECT 1").fetchone()
        finally:
            connection.close()

    def _connect(self) -> sqlite3.Connection:
        uri = self.database_path.resolve().as_uri() + "?mode=ro"
        connection = sqlite3.connect(uri, uri=True)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        connection.execute("PRAGMA query_only = ON")
        return connection
