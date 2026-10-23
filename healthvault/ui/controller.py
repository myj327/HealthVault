"""Small UI controller that delegates all domain work to existing services."""

from __future__ import annotations

import hashlib
import os
from pathlib import Path
from typing import Any

from healthvault.application.compare_health_indicators import (
    compare_health_indicators,
)
from healthvault.application.confirm_health_records import confirm_health_records
from healthvault.domain.records import ConfirmedSourceAsset
from healthvault.application.generate_visit_brief import generate_visit_brief
from healthvault.application.health_calendar import HealthCalendarService
from healthvault.application.query_health_timeline import query_health_timeline
from healthvault.application.report_preview import build_report_preview
from healthvault.application.search_health_knowledge import search_health_knowledge
from healthvault.infrastructure.sqlite_fts_health_knowledge import (
    SQLiteFTSHealthKnowledgeRepository,
)
from healthvault.infrastructure.hybrid_health_knowledge import (
    HybridHealthKnowledgeRepository,
)
from healthvault.infrastructure.sqlite_health_records import (
    SQLiteHealthRecordRepository,
)
from healthvault.infrastructure.sqlite_health_calendar import (
    SQLiteHealthCalendarRepository,
)
from healthvault.ocr.pipeline import OCRPipeline
from healthvault.rag.releases import active_corpus_root, configured_release_root


DEFAULT_DATABASE_PATH = Path(".task_outputs") / "healthvault-ui" / "healthvault.sqlite3"
DATABASE_ENVIRONMENT_VARIABLE = "HEALTHVAULT_UI_DB_PATH"
KNOWLEDGE_MODE_ENVIRONMENT_VARIABLE = "HEALTHVAULT_KNOWLEDGE_MODE"
HYBRID_INDEX_ENVIRONMENT_VARIABLE = "HEALTHVAULT_HYBRID_INDEX_PATH"
HYBRID_MODEL_ENVIRONMENT_VARIABLE = "HEALTHVAULT_HYBRID_MODEL_PATH"


def configured_database_path() -> Path:
    configured = os.environ.get(DATABASE_ENVIRONMENT_VARIABLE)
    return Path(configured) if configured else DEFAULT_DATABASE_PATH


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


class HealthVaultUIController:
    """Expose presentation-friendly dicts without duplicating application logic."""

    def __init__(
        self,
        database_path: str | Path,
        *,
        ocr_pipeline: OCRPipeline | None = None,
        knowledge_release_root: str | Path | None = None,
    ) -> None:
        self.database_path = Path(database_path)
        self.database_path.parent.mkdir(parents=True, exist_ok=True)
        self.record_repository = SQLiteHealthRecordRepository(self.database_path)
        self.calendar_service = HealthCalendarService(
            SQLiteHealthCalendarRepository(self.database_path)
        )
        self.knowledge_release_root = Path(knowledge_release_root) if knowledge_release_root is not None else configured_release_root()
        self.reload_knowledge()
        self.ocr_pipeline = ocr_pipeline or OCRPipeline()

    def reload_knowledge(self) -> None:
        corpus = active_corpus_root(self.knowledge_release_root)
        self.knowledge_path = corpus / "chunks" / "hybrid_v1.json"
        from healthvault.rag.retrieval_config import selection
        settings = selection(corpus, self.knowledge_release_root)
        self.knowledge_mode = settings['knowledge_mode']
        self.hybrid_index_path = settings['hybrid_index_path']
        self.hybrid_model_path = settings['hybrid_model_path']
        lexical_repository = SQLiteFTSHealthKnowledgeRepository(self.knowledge_path, corpus / "aliases" / "aliases.json")
        if self.knowledge_mode == "hybrid":
            self.knowledge_repository = HybridHealthKnowledgeRepository.from_local_artifacts(
                lexical_repository,
                artifact_path=self.knowledge_path,
                index_directory=self.hybrid_index_path,
                model_path=self.hybrid_model_path,
            )
        else:
            self.knowledge_repository = lexical_repository

    def process_image(self, image_bytes: bytes, filename: str) -> dict[str, Any]:
        return self.ocr_pipeline.process(image_bytes, filename)

    def preview_report(self, source_id: str, report_text: str) -> dict[str, Any]:
        preview = build_report_preview(source_id, report_text).to_dict()
        return {
            "input_kind": "text",
            "source_id": source_id,
            "report_text": report_text,
            "sha256": _sha256(report_text),
            "preview": preview,
        }

    def preview_ocr_report(
        self,
        source_id: str,
        report_text: str,
        image_sha256: str,
        ocr_text_sha256: str,
        *,
        ocr_blocks: list[dict[str, Any]] | None = None,
        reviewed: bool,
    ) -> dict[str, Any]:
        source_type = "ocr" if _sha256(report_text) == ocr_text_sha256 else "ocr_edited"
        preview = build_report_preview(
            source_id,
            report_text,
            clinical_context={
                "source_type": source_type,
                "ocr_blocks": ocr_blocks or [],
            },
        ).to_dict()
        return {
            "input_kind": "image",
            "source_id": source_id,
            "report_text": report_text,
            "sha256": _sha256(report_text),
            "image_sha256": image_sha256,
            "ocr_text_sha256": ocr_text_sha256,
            "ocr_reviewed": reviewed,
            "preview": preview,
        }

    @staticmethod
    def preview_matches(
        preview_state: Any,
        current_source_id: str,
        current_report_text: str,
        *,
        current_image_sha256: str | None = None,
        ocr_reviewed: bool = False,
    ) -> bool:
        common_matches = (
            isinstance(preview_state, dict)
            and preview_state.get("source_id") == current_source_id
            and preview_state.get("report_text") == current_report_text
            and preview_state.get("sha256") == _sha256(current_report_text)
            and isinstance(preview_state.get("preview"), dict)
        )
        if not common_matches:
            return False
        if preview_state.get("input_kind") != "image":
            return True
        return (
            preview_state.get("image_sha256") == current_image_sha256
            and preview_state.get("ocr_reviewed") is True
            and ocr_reviewed is True
        )

    def confirm_preview(
        self,
        *,
        profile_id: str,
        preview_state: Any,
        current_source_id: str,
        current_report_text: str,
        current_image_sha256: str | None = None,
        ocr_reviewed: bool = False,
        original_asset: ConfirmedSourceAsset | None = None,
    ) -> dict[str, Any]:
        if not isinstance(preview_state, dict):
            return {
                "status": "preview_required",
                "confirmed": False,
                "write_performed": False,
                "idempotent_replay": False,
                "issues": [
                    {
                        "code": "preview_required",
                        "message": "Preview the current report before confirmation.",
                    }
                ],
            }
        if preview_state.get("input_kind") == "image" and not ocr_reviewed:
            return {
                "status": "ocr_review_required",
                "confirmed": False,
                "write_performed": False,
                "idempotent_replay": False,
                "issues": [
                    {
                        "code": "ocr_review_required",
                        "message": "Review the OCR text before confirmation.",
                    }
                ],
            }
        if not self.preview_matches(
            preview_state,
            current_source_id,
            current_report_text,
            current_image_sha256=current_image_sha256,
            ocr_reviewed=ocr_reviewed,
        ):
            return {
                "status": "preview_mismatch",
                "confirmed": False,
                "write_performed": False,
                "idempotent_replay": False,
                "issues": [
                    {
                        "code": "preview_mismatch",
                        "message": "Report source or text changed after preview.",
                    }
                ],
            }
        return confirm_health_records(
            {
                "profile_id": profile_id,
                "source_id": current_source_id,
                "report_text": current_report_text,
                "expected_sha256": preview_state["sha256"],
                "user_confirmed": True,
            },
            self.record_repository,
            original_asset=original_asset,
            preview_clinical_items=preview_state["preview"].get(
                "clinical_items", []
            ),
            evidence_locators={
                item["code"]: {
                    key: item["evidence"][key]
                    for key in ("span_start", "span_end", "regions", "fields")
                    if key in item["evidence"]
                }
                for item in preview_state["preview"].get("indicators", [])
                if "span_start" in item["evidence"]
            },
        ).to_dict()

    def timeline(self, profile_id: str) -> dict[str, Any]:
        return query_health_timeline(
            {"profile_id": profile_id}, self.record_repository
        ).to_dict()

    def comparisons(self, profile_id: str) -> dict[str, Any]:
        return compare_health_indicators(
            {"profile_id": profile_id}, self.record_repository
        ).to_dict()

    def search_knowledge(self, query: str, limit: int = 3) -> dict[str, Any]:
        return search_health_knowledge(
            {"query": query, "limit": limit}, self.knowledge_repository
        ).to_dict()

    def visit_brief(self, profile_id: str) -> dict[str, Any]:
        return generate_visit_brief(
            {"profile_id": profile_id},
            self.record_repository,
            self.knowledge_repository,
        ).to_dict()

    def sync_calendar(self, profile_id: str) -> dict[str, Any]:
        return self.calendar_service.sync(profile_id)

    def calendar(self, profile_id: str, as_of: str) -> dict[str, Any]:
        return self.calendar_service.view(profile_id, as_of)

    def create_calendar_event(
        self,
        profile_id: str,
        idempotency_key: str,
        title: str,
        scheduled_date: str,
        remind_before_days: int,
    ) -> dict[str, Any]:
        return self.calendar_service.create_user_event(
            profile_id,
            idempotency_key,
            title,
            scheduled_date,
            remind_before_days,
        )

    def activate_calendar_event(
        self, profile_id: str, event_id: str, **changes: Any
    ) -> dict[str, Any]:
        return self.calendar_service.activate(profile_id, event_id, **changes)

    def update_calendar_event(
        self, profile_id: str, event_id: str, **changes: Any
    ) -> dict[str, Any]:
        return self.calendar_service.update(profile_id, event_id, **changes)

    def complete_calendar_event(
        self, profile_id: str, event_id: str
    ) -> dict[str, Any]:
        return self.calendar_service.complete(profile_id, event_id)

    def cancel_calendar_event(
        self, profile_id: str, event_id: str
    ) -> dict[str, Any]:
        return self.calendar_service.cancel(profile_id, event_id)

    def acknowledge_calendar_reminder(
        self, profile_id: str, event_id: str, as_of: str
    ) -> dict[str, Any]:
        return self.calendar_service.acknowledge(profile_id, event_id, as_of)
