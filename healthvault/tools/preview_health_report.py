"""Tool definition and handler for synthetic health report previews."""

from __future__ import annotations

import json
from typing import Any

from healthvault.application.report_preview import build_report_preview
from healthvault.tools.directory import tool_definition


PREVIEW_HEALTH_REPORT_TOOL = tool_definition("preview_health_report")


async def preview_health_report(tool_input: Any) -> str:
    """Return a validated preview JSON string without persistent side effects."""
    if isinstance(tool_input, dict):
        source_id = tool_input.get("source_id")
        report_text = tool_input.get("report_text")
    else:
        source_id = None
        report_text = None

    preview = build_report_preview(source_id, report_text)
    return json.dumps(preview.to_dict(), ensure_ascii=False)
