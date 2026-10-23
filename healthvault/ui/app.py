"""Single-process Streamlit demo for the existing offline HealthVault flow."""

from __future__ import annotations

import hashlib
from datetime import date
from html import escape

import streamlit as st

from healthvault.ocr.pipeline import OCRProcessingError
from healthvault.ui.demo_data import seed_demo_profile
from healthvault.ui.controller import (
    HealthVaultUIController,
    configured_database_path,
)
from healthvault.ui.theme import inject_theme


st.set_page_config(
    page_title="HealthVault · 纵向健康档案",
    page_icon=":material/health_and_safety:",
    layout="wide",
    initial_sidebar_state="auto",
)
inject_theme()


@st.cache_resource
def controller_for(database_path: str) -> HealthVaultUIController:
    return HealthVaultUIController(database_path)


def render_hero() -> None:
    st.markdown(
        """
        <a class="hv-skip-link" href="#healthvault-main">跳到主要内容</a>
        <section class="hv-hero" aria-labelledby="healthvault-title">
          <div class="hv-hero-main">
            <div class="hv-hero-topline">
              <span class="hv-mark" aria-hidden="true">HV</span>
              <span class="hv-eyebrow">Longitudinal Health Record Workspace</span>
            </div>
            <h1 id="healthvault-title">个人纵向健康档案工作台</h1>
            <p class="hv-hero-copy">
              导入并确认健康报告，查看跨期指标，检索可追溯医学知识，并整理就诊准备信息。
            </p>
          </div>
          <div class="hv-hero-status" role="status" aria-label="系统状态">
            <span class="hv-status-pill is-ready"><span class="hv-dot" aria-hidden="true"></span>本地服务就绪</span>
            <span class="hv-status-pill">5 个工作模块</span>
          </div>
        </section>
        <div id="healthvault-main"></div>
        """,
        unsafe_allow_html=True,
    )


def render_section_intro(kicker: str, title: str, description: str) -> None:
    st.markdown(
        f"""
        <header class="hv-section-head">
          <div class="hv-kicker">{escape(kicker)}</div>
          <h2>{escape(title)}</h2>
          <p>{escape(description)}</p>
        </header>
        """,
        unsafe_allow_html=True,
    )


def nested_markdown(markdown: str) -> str:
    """Keep generated section headings below the page and tab headings."""

    return "\n".join(
        f"##{line}" if line.startswith("#") else line
        for line in markdown.splitlines()
    )


def render_overview(controller: HealthVaultUIController, profile_id: str) -> None:
    timeline = controller.timeline(profile_id)
    comparisons = controller.comparisons(profile_id)
    calendar = controller.calendar(profile_id, date.today().isoformat())
    reports = timeline.get("timeline", []) if timeline.get("status") == "ok" else []
    indicator_count = sum(len(report.get("indicators", [])) for report in reports)
    comparison_count = len(comparisons.get("comparisons", []))
    pending_count = len(calendar.get("pending_confirmation", []))
    upcoming_count = len(calendar.get("upcoming_due", []))

    st.markdown(
        f"""
        <section aria-label="当前档案概览">
          <div class="hv-overview-head">
            <strong>当前档案概览</strong>
            <span>仅展示经过确认并写入当前档案的数据</span>
          </div>
          <div class="hv-stat-grid">
            <article class="hv-stat-card">
              <span class="hv-stat-label">已确认报告</span>
              <strong class="hv-stat-value">{len(reports)}</strong>
              <span class="hv-stat-note">只统计通过人工确认的档案</span>
            </article>
            <article class="hv-stat-card">
              <span class="hv-stat-label">结构化指标</span>
              <strong class="hv-stat-value">{indicator_count}</strong>
              <span class="hv-stat-note">保留数值、单位与来源证据</span>
            </article>
            <article class="hv-stat-card">
              <span class="hv-stat-label">可比较趋势</span>
              <strong class="hv-stat-value">{comparison_count}</strong>
              <span class="hv-stat-note">同指标、同单位的跨期变化</span>
            </article>
            <article class="hv-stat-card">
              <span class="hv-stat-label">健康日程</span>
              <strong class="hv-stat-value">{pending_count + upcoming_count}</strong>
              <span class="hv-stat-note">待确认 {pending_count} · 即将到期 {upcoming_count}</span>
            </article>
          </div>
        </section>
        """,
        unsafe_allow_html=True,
    )


def render_issues(issues: list[dict]) -> None:
    for issue in issues:
        context = f"（第 {issue['line']} 行）" if "line" in issue else ""
        st.error(f"{issue['code']}：{issue['message']}{context}")


def render_import_tab(controller: HealthVaultUIController, profile_id: str) -> None:
    render_section_intro(
        "01 · Report Intake",
        "导入、核对，再写入档案",
        "解析和预览都不会改动数据库；只有完成核对并点击确认后，报告才会进入当前用户档案。",
    )
    st.markdown(
        """
        <div class="hv-step-strip" aria-label="导入流程">
          <div class="hv-mini-step"><strong>1 · 上传与解析</strong>支持 UTF-8 文本与本地图片 OCR</div>
          <div class="hv-mini-step"><strong>2 · 核对与预览</strong>原图、文本、字段和来源哈希保持一致</div>
          <div class="hv-mini-step"><strong>3 · 明确确认</strong>确认后幂等写入当前档案身份</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    uploaded = st.file_uploader(
        "上传 UTF-8 .txt 或清晰的 PNG/JPG 合成报告",
        type=["txt", "png", "jpg", "jpeg"],
    )
    st.session_state.setdefault("ui_source_id", "")
    st.session_state.setdefault("ui_report_text", "")
    st.session_state.setdefault("ui_input_kind", "text")

    if uploaded is None and st.session_state.get("ui_upload_signature"):
        st.session_state.pop("ui_upload_signature", None)
        st.session_state.pop("ui_preview", None)
        st.session_state.pop("ui_confirmation", None)
        st.session_state.pop("ui_ocr_result", None)
        st.session_state["ui_input_kind"] = "text"
        st.session_state["ui_source_id"] = ""
        st.session_state["ui_report_text"] = ""
        st.session_state["ui_ocr_reviewed"] = False
    elif uploaded is not None:
        raw = uploaded.getvalue()
        extension = uploaded.name.rsplit(".", 1)[-1].lower()
        input_kind = "text" if extension == "txt" else "image"
        content_hash = hashlib.sha256(raw).hexdigest()
        upload_signature = f"{input_kind}:{uploaded.name}:{content_hash}"
        if st.session_state.get("ui_upload_signature") != upload_signature:
            st.session_state["ui_upload_signature"] = upload_signature
            st.session_state["ui_input_kind"] = input_kind
            st.session_state["ui_upload_error"] = ""
            st.session_state["ui_report_text"] = ""
            st.session_state.pop("ui_preview", None)
            st.session_state.pop("ui_confirmation", None)
            st.session_state.pop("ui_ocr_result", None)
            st.session_state["ui_ocr_reviewed"] = False
            if input_kind == "text":
                try:
                    decoded = raw.decode("utf-8", errors="strict")
                except UnicodeDecodeError:
                    st.session_state["ui_upload_error"] = (
                        "文件不是有效的 UTF-8 文本。"
                    )
                    st.session_state["ui_source_id"] = ""
                else:
                    st.session_state["ui_source_id"] = uploaded.name
                    st.session_state["ui_report_text"] = decoded
            else:
                st.session_state["ui_source_id"] = f"image:{content_hash}"

        if input_kind == "image":
            st.image(raw, caption=uploaded.name, width="stretch")
            if st.button("执行本地 OCR（CPU）"):
                try:
                    result = controller.process_image(raw, uploaded.name)
                except OCRProcessingError as exc:
                    st.session_state["ui_upload_error"] = str(exc)
                    st.session_state.pop("ui_ocr_result", None)
                    st.session_state["ui_report_text"] = ""
                else:
                    st.session_state["ui_upload_error"] = ""
                    st.session_state["ui_ocr_result"] = result
                    st.session_state["ui_report_text"] = result["text"]
                    st.session_state["ui_ocr_reviewed"] = False
                    st.session_state.pop("ui_preview", None)
                    st.session_state.pop("ui_confirmation", None)

    if st.session_state.get("ui_upload_error"):
        st.error(st.session_state["ui_upload_error"])

    input_kind = st.session_state.get("ui_input_kind", "text")
    ocr_result = st.session_state.get("ui_ocr_result")
    if input_kind == "image" and ocr_result is not None:
        confidence = ocr_result["confidence"]
        average_column, minimum_column = st.columns(2)
        average_column.metric("平均置信度", f"{confidence['average']:.1%}")
        minimum_column.metric("最低置信度", f"{confidence['minimum']:.1%}")
        if confidence["low_confidence_blocks"]:
            st.warning(
                "存在低置信度文字块，请对照原图逐项核对后再生成预览。"
            )
        st.caption(
            f"{ocr_result['engine']['name']} {ocr_result['engine']['version']} · "
            f"{ocr_result['engine']['runtime']} "
            f"{ocr_result['engine']['runtime_version']} · "
            f"artifact={ocr_result['artifact']['directory']}"
        )
        with st.expander("逐块 OCR 结果与坐标"):
            st.dataframe(ocr_result["blocks"], width="stretch", hide_index=True)

    source_id = st.text_input("来源 ID", key="ui_source_id")
    text_label = "OCR 文本（可编辑）" if input_kind == "image" else "报告原文"
    report_text = st.text_area(text_label, key="ui_report_text", height=220)
    ocr_reviewed = False
    if input_kind == "image":
        ocr_reviewed = st.checkbox("已核对 OCR 文本", key="ui_ocr_reviewed")

    can_generate = bool(source_id.strip() and report_text.strip())
    if input_kind == "image":
        can_generate = can_generate and ocr_result is not None and ocr_reviewed
    if st.button("生成预览", type="primary", disabled=not can_generate):
        if input_kind == "image":
            st.session_state["ui_preview"] = controller.preview_ocr_report(
                source_id=source_id,
                report_text=report_text,
                image_sha256=ocr_result["image_sha256"],
                ocr_text_sha256=ocr_result["text_sha256"],
                ocr_blocks=ocr_result["blocks"],
                reviewed=ocr_reviewed,
            )
        else:
            st.session_state["ui_preview"] = controller.preview_report(
                source_id, report_text
            )
        st.session_state.pop("ui_confirmation", None)

    preview_state = st.session_state.get("ui_preview")
    if preview_state is None:
        if input_kind == "image" and ocr_result is None:
            st.info("先执行本地 OCR，再核对识别文本；这些步骤不会写入数据库。")
        elif input_kind == "image" and not ocr_reviewed:
            st.info("请对照原图核对 OCR 文本并勾选确认，再生成预览。")
        else:
            st.info("上传报告后先生成预览；此步骤不会写入数据库。")
        return

    current_image_sha256 = (
        ocr_result["image_sha256"]
        if input_kind == "image" and ocr_result is not None
        else None
    )
    matches = controller.preview_matches(
        preview_state,
        source_id,
        report_text,
        current_image_sha256=current_image_sha256,
        ocr_reviewed=ocr_reviewed,
    )
    if not matches:
        st.session_state.pop("ui_preview", None)
        st.session_state.pop("ui_confirmation", None)
        st.warning("图片、来源、OCR 核对状态或报告内容已变化，旧预览已失效。")
        return

    preview = preview_state["preview"]
    st.markdown("### 当前预览")
    st.json(
        {
            "status": preview["status"],
            "source": preview["source"],
            "report": preview["report"],
            "indicators": preview["indicators"],
            "clinical_items": preview["clinical_items"],
            "warnings": preview.get("warnings", []),
        }
    )
    render_issues(preview["issues"])

    can_confirm = bool(profile_id.strip()) and preview["can_confirm"] and matches
    if st.button("确认并保存", disabled=not can_confirm):
        st.session_state["ui_confirmation"] = controller.confirm_preview(
            profile_id=profile_id,
            preview_state=preview_state,
            current_source_id=source_id,
            current_report_text=report_text,
            current_image_sha256=current_image_sha256,
            ocr_reviewed=ocr_reviewed,
        )

    confirmation = st.session_state.get("ui_confirmation")
    if confirmation is not None and confirmation.get("profile_id") != profile_id:
        confirmation = None
    if confirmation is not None:
        if confirmation.get("confirmed"):
            label = "已保存" if confirmation["write_performed"] else "已存在，未重复写入"
            st.success(label)
        else:
            render_issues(confirmation["issues"])


def render_timeline_tab(controller: HealthVaultUIController, profile_id: str) -> None:
    render_section_intro(
        "02 · Longitudinal View",
        "健康时间线与跨期比较",
        "按报告日期还原纵向记录，并仅对同代码、同单位的已确认指标计算变化。",
    )
    timeline = controller.timeline(profile_id)
    if timeline["status"] != "ok":
        st.info("当前档案还没有已确认报告。可从左侧一键载入合成演示档案。")
        return

    series_by_code: dict[str, list[dict]] = {}
    indicator_labels: dict[str, str] = {}
    for report in timeline["timeline"]:
        for item in report["indicators"]:
            code = item["code"]
            indicator_labels[code] = f"{item['display_name']}（{item['unit']}）"
            series_by_code.setdefault(code, []).append(
                {
                    "date": report["report_date"],
                    "value": item["value"],
                    "flag": item["flag"],
                }
            )

    chartable_codes = [
        code for code, points in series_by_code.items() if len(points) >= 2
    ]
    if chartable_codes:
        st.markdown("### 关键指标趋势")
        selected_code = st.selectbox(
            "选择趋势指标",
            chartable_codes,
            format_func=lambda code: indicator_labels[code],
            key="timeline_chart_indicator",
        )
        chart_points = sorted(series_by_code[selected_code], key=lambda item: item["date"])
        latest_point = chart_points[-1]
        previous_point = chart_points[-2]
        delta = latest_point["value"] - previous_point["value"]
        metric_columns = st.columns(3)
        metric_columns[0].metric("最新值", latest_point["value"])
        metric_columns[1].metric("较上次变化", f"{delta:+.2f}")
        metric_columns[2].metric("报告内标记", latest_point["flag"])
        st.vega_lite_chart(
            data=None,
            spec={
                "data": {"values": chart_points},
                "mark": {
                    "type": "line",
                    "color": "#0284c7",
                    "strokeWidth": 3,
                    "point": {"filled": True, "size": 90, "color": "#0f766e"},
                },
                "encoding": {
                    "x": {
                        "field": "date",
                        "type": "temporal",
                        "title": None,
                        "axis": {"format": "%Y-%m", "labelAngle": 0},
                    },
                    "y": {
                        "field": "value",
                        "type": "quantitative",
                        "title": indicator_labels[selected_code],
                        "scale": {"zero": False},
                    },
                    "tooltip": [
                        {"field": "date", "type": "temporal", "title": "报告日期"},
                        {"field": "value", "type": "quantitative", "title": "数值"},
                        {"field": "flag", "type": "nominal", "title": "报告标记"},
                    ],
                },
                "height": 240,
                "config": {
                    "view": {"stroke": None},
                    "axis": {
                        "gridColor": "#e2e8f0",
                        "domainColor": "#94a3b8",
                        "labelColor": "#475569",
                        "titleColor": "#334155",
                    },
                },
            },
            width="stretch",
        )

    st.markdown("### 已确认报告")
    for report in timeline["timeline"]:
        with st.expander(
            f"{report['report_date']} · {report['report_id']} · {report['facility']}",
            expanded=True,
        ):
            st.caption(
                f"source_id={report['source']['source_id']} · "
                f"sha256={report['source']['sha256']}"
            )
            st.dataframe(
                [
                    {
                        "指标": item["display_name"],
                        "代码": item["code"],
                        "数值": item["value"],
                        "单位": item["unit"],
                        "标记": item["flag"],
                        "原始证据": item["evidence"]["quote"],
                    }
                    for item in report["indicators"]
                ],
                width="stretch",
                hide_index=True,
            )
            st.markdown("**病历明确记载**")
            if report["clinical_items"]:
                st.dataframe(
                    [
                        {
                            "类型": item["kind"],
                            "原始措辞": item["original_text"],
                            "规范化值": item["normalized_value"],
                            "日期状态": item["date_status"],
                            "规范化日期": item["normalized_date"],
                            "确认状态": item["confirmation_status"],
                            "来源证据": item["evidence"]["quote"],
                            "置信度": item["confidence"],
                        }
                        for item in report["clinical_items"]
                    ],
                    width="stretch",
                    hide_index=True,
                )
            else:
                st.info("本报告没有明示诊断、医生建议、就诊类型或复诊计划。")
            st.markdown("系统推断：0（本版本不会从指标、症状或知识库推断诊断/医嘱）")

    comparison = controller.comparisons(profile_id)
    st.markdown("### 跨期比较")
    if not comparison["comparisons"]:
        st.info("至少需要两次同指标记录才能比较。")
    else:
        st.dataframe(
            [
                {
                    "指标": item["display_name"],
                    "最早日期": item["earliest"]["report_date"],
                    "最早值": item["earliest"]["value"],
                    "最新日期": item["latest"]["report_date"],
                    "最新值": item["latest"]["value"],
                    "单位": item["unit"],
                    "变化": item["absolute_change"],
                    "方向": item["direction"],
                }
                for item in comparison["comparisons"]
            ],
            width="stretch",
            hide_index=True,
        )


def render_knowledge_tab(controller: HealthVaultUIController) -> None:
    render_section_intro(
        "03 · Evidence Retrieval",
        "离线健康知识检索",
        "检索项目内固定的权威来源；证据不足或超出健康教育范围时，系统会停止作答。",
    )
    st.session_state.setdefault("ui_knowledge_query", "")
    st.caption("快速示例")
    query_presets = (
        ("HbA1c 空腹要求", "Must I fast before an HbA1c blood draw?"),
        ("参考范围", "Why should I check the reference range on my own lab report?"),
        ("越界请求", "Which prescription dose should I use to lower a lab value?"),
    )
    preset_columns = st.columns(3)
    for index, (label, preset) in enumerate(query_presets):
        if preset_columns[index].button(label, key=f"knowledge_preset_{index}"):
            st.session_state["ui_knowledge_query"] = preset

    query = st.text_input(
        "检索问题",
        key="ui_knowledge_query",
        placeholder="输入健康教育问题，例如 HbA1c fasting requirements…",
        help="检索仅使用项目内版本化官方语料，不访问互联网。",
    )
    limit = st.slider("返回证据数量", 1, 5, 3)
    if st.button("检索知识", type="primary", disabled=not query.strip()):
        st.session_state["ui_search_result"] = controller.search_knowledge(query, limit)

    result = st.session_state.get("ui_search_result")
    if result is None:
        st.info("检索仅使用项目内固定官方语料，不访问互联网。")
        return
    render_issues(result["issues"])
    if not result["results"]:
        st.info("没有找到足够相关的离线证据。")
    for hit in result["results"]:
        source = hit["source"]
        with st.container(border=True):
            title_column, rank_column = st.columns([8, 1])
            title_column.markdown(f"### {hit['title']}")
            rank_column.metric("排序", hit["rank"])
            st.markdown(
                f'<span class="hv-citation-pill">{escape(hit["citation_id"])}</span>',
                unsafe_allow_html=True,
            )
            st.markdown(f"来源：[{source['publisher']}]({source['url']})")
            st.caption(" / ".join(hit["title_path"]))
            st.write(hit["excerpt"])


def render_brief_tab(controller: HealthVaultUIController, profile_id: str) -> None:
    render_section_intro(
        "04 · Visit Preparation",
        "有来源的就诊准备摘要",
        "将已确认档案、跨期变化与官方知识证据组织成可带去就诊的问题清单。",
    )
    if st.button("生成离线摘要", type="primary"):
        st.session_state["ui_brief"] = controller.visit_brief(profile_id)
    result = st.session_state.get("ui_brief")
    if result is not None and result.get("profile_id") != profile_id:
        result = None
    if result is None:
        st.info("摘要直接复用已确认档案、跨期比较和离线知识 citation。")
        return
    render_issues(result["issues"])
    if result["brief_markdown"]:
        with st.container(border=True):
            st.markdown(nested_markdown(result["brief_markdown"]))
    else:
        st.info("当前 profile 没有可用于摘要的已确认报告。")


def _calendar_event_table(events: list[dict]) -> None:
    if not events:
        st.caption("暂无")
        return
    st.dataframe(
        [
            {
                "event_id": event["event_id"],
                "来源": event["source"],
                "类型": event["event_type"],
                "标题": event["title"],
                "日期": event["scheduled_date"],
                "状态": event["status"],
                "提醒": event["reminder_status"],
                "原始病历": event["source_original_text"],
            }
            for event in events
        ],
        width="stretch",
        hide_index=True,
    )


def render_calendar_tab(controller: HealthVaultUIController, profile_id: str) -> None:
    render_section_intro(
        "05 · Health Calendar",
        "可确认、可追溯的健康日历",
        "病历中的复诊计划先进入待确认队列，用户核对日期后才会激活提醒。",
    )
    as_of = st.date_input("查询日期 as_of", value=date.today(), key="calendar_as_of")
    if st.button("同步已确认病历", key="calendar_sync"):
        st.session_state["calendar_action"] = controller.sync_calendar(profile_id)

    with st.form("calendar_create_user"):
        st.markdown("### 新建个人日程")
        idempotency_key = st.text_input("幂等键", key="calendar_new_key")
        title = st.text_input("标题", key="calendar_new_title")
        scheduled = st.date_input("日期", value=as_of, key="calendar_new_date")
        reminder = st.number_input(
            "提前提醒天数", min_value=0, max_value=365, value=1, key="calendar_new_reminder"
        )
        if st.form_submit_button("创建日程"):
            st.session_state["calendar_action"] = controller.create_calendar_event(
                profile_id,
                idempotency_key,
                title,
                scheduled.isoformat(),
                int(reminder),
            )

    view = controller.calendar(profile_id, as_of.isoformat())
    action = st.session_state.get("calendar_action")
    if action is not None:
        st.caption(f"最近操作：{action['status']}")

    calendar_columns = st.columns(3)
    calendar_columns[0].metric("待确认", len(view["pending_confirmation"]))
    calendar_columns[1].metric("即将到期", len(view["upcoming_due"]))
    calendar_columns[2].metric("已结束", len(view["ended"]))

    st.markdown("### 待确认事件")
    _calendar_event_table(view["pending_confirmation"])
    for event in view["pending_confirmation"]:
        with st.expander(f"激活 · {event['title']}"):
            with st.form(f"activate_{event['event_id']}"):
                activate_title = st.text_input(
                    "核对标题", value=event["title"], key=f"activate_title_{event['event_id']}"
                )
                activate_date = st.date_input(
                    "核对日期",
                    value=date.fromisoformat(event["scheduled_date"]),
                    key=f"activate_date_{event['event_id']}",
                )
                activate_reminder = st.number_input(
                    "提前提醒天数",
                    min_value=0,
                    max_value=365,
                    value=event["remind_before_days"],
                    key=f"activate_reminder_{event['event_id']}",
                )
                if st.form_submit_button("确认并激活"):
                    st.session_state["calendar_action"] = controller.activate_calendar_event(
                        profile_id,
                        event["event_id"],
                        title=activate_title,
                        scheduled_date=activate_date.isoformat(),
                        remind_before_days=int(activate_reminder),
                    )
                    st.rerun()
                if st.form_submit_button("取消待确认事件"):
                    st.session_state["calendar_action"] = controller.cancel_calendar_event(
                        profile_id, event["event_id"]
                    )
                    st.rerun()

    st.markdown("### 即将到期")
    _calendar_event_table(view["upcoming_due"])
    for event in view["upcoming_due"]:
        with st.expander(f"管理 · {event['title']}"):
            with st.form(f"update_{event['event_id']}"):
                update_title = st.text_input(
                    "标题", value=event["title"], key=f"update_title_{event['event_id']}"
                )
                update_date = st.date_input(
                    "日期",
                    value=date.fromisoformat(event["scheduled_date"]),
                    key=f"update_date_{event['event_id']}",
                )
                update_reminder = st.number_input(
                    "提前提醒天数",
                    min_value=0,
                    max_value=365,
                    value=event["remind_before_days"],
                    key=f"update_reminder_{event['event_id']}",
                )
                if st.form_submit_button("保存修改"):
                    st.session_state["calendar_action"] = controller.update_calendar_event(
                        profile_id,
                        event["event_id"],
                        title=update_title,
                        scheduled_date=update_date.isoformat(),
                        remind_before_days=int(update_reminder),
                    )
                    st.rerun()
            action_columns = st.columns(3)
            if action_columns[0].button("完成", key=f"complete_{event['event_id']}"):
                st.session_state["calendar_action"] = controller.complete_calendar_event(
                    profile_id, event["event_id"]
                )
                st.rerun()
            if action_columns[1].button("取消", key=f"cancel_{event['event_id']}"):
                st.session_state["calendar_action"] = controller.cancel_calendar_event(
                    profile_id, event["event_id"]
                )
                st.rerun()
            if action_columns[2].button(
                "确认提醒",
                key=f"ack_{event['event_id']}",
                disabled=event["reminder_status"] not in {"due", "overdue"},
            ):
                st.session_state["calendar_action"] = controller.acknowledge_calendar_reminder(
                    profile_id, event["event_id"], as_of.isoformat()
                )
                st.rerun()

    st.markdown("### 历史与已结束")
    _calendar_event_table(view["ended"])
    st.markdown("### 待修正计划")
    st.metric("未生成事件的模糊/无效日期计划", view["needs_date_confirmation_count"])


def render_sidebar(
    controller: HealthVaultUIController,
    database_path: str,
) -> str:
    st.sidebar.markdown(
        """
        <div class="hv-sidebar-brand">
          <span class="hv-mark" aria-hidden="true">HV</span>
          <span class="hv-sidebar-title">HealthVault</span>
        </div>
        <p class="hv-sidebar-copy">本地优先的个人纵向健康档案</p>
        <p class="hv-sidebar-label">档案控制</p>
        """,
        unsafe_allow_html=True,
    )
    profile_id = st.sidebar.text_input(
        "档案身份",
        value="demo-profile",
        help="运行时会将身份注入到应用层；模型不能覆盖该值。",
    )

    if st.sidebar.button(
        "一键载入合成演示档案",
        type="primary",
        width="stretch",
        disabled=not profile_id.strip(),
    ):
        st.session_state["ui_demo_seed"] = seed_demo_profile(controller, profile_id)
        st.rerun()

    seed_result = st.session_state.get("ui_demo_seed")
    if seed_result is not None and seed_result.get("profile_id") != profile_id.strip():
        seed_result = None
    if seed_result is not None:
        if seed_result.get("status") == "ready":
            st.sidebar.success(seed_result["message"])
            st.sidebar.caption(
                f"新增 {seed_result['inserted_reports']} 份 · "
                f"已存在 {seed_result['existing_reports']} 份"
            )
        else:
            st.sidebar.error(seed_result.get("message", "演示档案加载失败。"))

    repository_name = type(controller.knowledge_repository).__name__
    retrieval_mode = "Hybrid RRF" if repository_name.startswith("Hybrid") else "SQLite FTS5"
    st.sidebar.markdown(
        f"""
        <p class="hv-sidebar-label">运行状态</p>
        <div class="hv-mode-row"><span>数据边界</span><strong>本地 SQLite</strong></div>
        <div class="hv-mode-row"><span>知识检索</span><strong>{escape(retrieval_mode)}</strong></div>
        <div class="hv-mode-row"><span>模型 API</span><strong>主链无需调用</strong></div>
        <p class="hv-sidebar-label">演示路径</p>
        <div class="hv-demo-route">
          <strong>建议操作顺序</strong>
          1. 一键载入合成档案<br>
          2. 查看跨年趋势与证据<br>
          3. 演示知识检索边界<br>
          4. 生成就诊准备摘要<br>
          5. 核对健康日历事件
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.sidebar.caption(f"数据库：{database_path}")
    return profile_id


database_path = configured_database_path()
controller = controller_for(str(database_path))
profile_id = render_sidebar(controller, str(database_path))
render_hero()
st.warning("当前仅使用合成演示数据。系统仅提供健康教育与就诊准备，不提供诊断、改药或剂量建议。")
render_overview(controller, profile_id)

import_tab, timeline_tab, knowledge_tab, brief_tab, calendar_tab = st.tabs(
    ["导入与确认", "健康时间线", "知识检索", "就诊摘要", "健康日历"]
)
with import_tab:
    render_import_tab(controller, profile_id)
with timeline_tab:
    render_timeline_tab(controller, profile_id)
with knowledge_tab:
    render_knowledge_tab(controller)
with brief_tab:
    render_brief_tab(controller, profile_id)
with calendar_tab:
    render_calendar_tab(controller, profile_id)
