"""Idempotent synthetic data used by the interview demo."""

from __future__ import annotations

from typing import Any

from healthvault.ui.controller import HealthVaultUIController


DEMO_REPORTS = (
    {
        "source_id": "demo-checkup-2024",
        "report_text": """# Synthetic demo data only; no real person
report_id: HV-DEMO-2024-001
report_date: 2024-05-20
facility: Synthetic Community Clinic
result|fasting_glucose|5.4|mmol/L|3.9|6.1
result|hba1c|5.5|%|4.0|6.0
result|ldl_cholesterol|3.1|mmol/L|0.0|3.4
""",
    },
    {
        "source_id": "demo-checkup-2025",
        "report_text": """# Synthetic demo data only; no real person
report_id: HV-DEMO-2025-001
report_date: 2025-06-18
facility: Synthetic Community Clinic
result|fasting_glucose|6.5|mmol/L|3.9|6.1
result|hba1c|5.9|%|4.0|6.0
result|ldl_cholesterol|2.8|mmol/L|0.0|3.4
""",
    },
    {
        "source_id": "demo-clinical-2026",
        "report_text": """# Synthetic demo data only; no real person
report_id: HV-DEMO-CLINICAL-2026-001
report_date: 2026-06-18
facility: Synthetic Community Clinic
result|fasting_glucose|6.8|mmol/L|3.9|6.1
诊断：2型糖尿病（E11.9）
医生建议：控制饮食并规律运动
就诊类型：复诊
复诊计划：2026年10月16日复查
""",
    },
)


def seed_demo_profile(
    controller: HealthVaultUIController,
    profile_id: str,
) -> dict[str, Any]:
    """Load a repeatable, synthetic longitudinal record without any model call."""

    normalized_profile_id = profile_id.strip()
    if not normalized_profile_id:
        return {
            "status": "invalid_profile",
            "message": "请先填写演示档案 ID。",
            "profile_id": normalized_profile_id,
            "inserted_reports": 0,
            "existing_reports": 0,
        }

    inserted_reports = 0
    existing_reports = 0
    for report in DEMO_REPORTS:
        preview = controller.preview_report(report["source_id"], report["report_text"])
        confirmation = controller.confirm_preview(
            profile_id=normalized_profile_id,
            preview_state=preview,
            current_source_id=report["source_id"],
            current_report_text=report["report_text"],
        )
        if confirmation.get("write_performed"):
            inserted_reports += 1
        elif confirmation.get("confirmed"):
            existing_reports += 1
        else:
            return {
                "status": "seed_failed",
                "message": "演示档案生成失败，请检查报告解析结果。",
                "profile_id": normalized_profile_id,
                "inserted_reports": inserted_reports,
                "existing_reports": existing_reports,
                "confirmation": confirmation,
            }

    calendar_sync = controller.sync_calendar(normalized_profile_id)
    controller.create_calendar_event(
        normalized_profile_id,
        "healthvault-demo-annual-review-v1",
        "年度健康复查（合成演示）",
        "2027-01-15",
        7,
    )
    return {
        "status": "ready",
        "message": "演示档案已就绪，可按导航顺序讲解完整闭环。",
        "profile_id": normalized_profile_id,
        "inserted_reports": inserted_reports,
        "existing_reports": existing_reports,
        "calendar_status": calendar_sync.get("status"),
    }
