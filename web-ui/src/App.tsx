import { useEffect, useMemo, useRef, useState } from 'react'
import SkillReviewPanel from './SkillReviewPanel'
import PersonalSkillPanel from './PersonalSkillPanel'
import SharedExperiencePanel from './SharedExperiencePanel'
import KnowledgeReviewPanel from './KnowledgeReviewPanel'
import EvidenceLocation, { type Evidence } from './EvidenceLocation'
import CalendarPanel from './CalendarPanel'
import './unsaved.css'

type Issue = { code: string; message: string; line?: number; quote?: string; indicator_code?: string }
type Indicator = {
  code: string; display_name: string; value: number; unit: string; flag?: string
  reference_range?: { lower: number; upper: number }; evidence?: Evidence
}
type ImagingFinding = {
  finding_id: string; finding_type: string; body_site: string; laterality: string; location: string
  dimensions_mm: number[]; original_dimensions: string; original_unit: string
  evidence?: Evidence
}
type Preview = {
  status: string; can_confirm?: boolean
  report?: { report_id?: string; report_date?: string; facility?: string }
  indicators?: Indicator[]; imaging_findings?: ImagingFinding[]; clinical_items?: Array<{ kind: string; original_text: string; normalized_date?: string | null; date_status?: string; evidence?: Evidence }>
  other_findings?: Array<{ evidence: { line_start: number; line_end: number; quote: string } }>
  issues?: Issue[]; warnings?: Issue[]
}
type Draft = {
  token: string; source_id?: string; report_text: string; input_kind?: 'text' | 'image' | 'pdf'; preview?: Preview | null
  preview_sha256?: string; image_sha256?: string; image_url?: string | null
  original_text?: string; ocr_reviewed?: boolean; processing_status?: string; processing_error?: string | null; saved?: boolean
  archived_at?: string | null; duplicate_of?: { image_filename?: string | null; source_id: string } | null
}
type ReportUpload = { token: string; source_id: string; input_kind: string; image_filename: string | null; processing_status: string; updated_at: string; archived_at?: string | null; duplicate_of?: { source_id: string } | null }
type TimelineReport = { report_id: string; report_date: string; facility: string; indicators: Indicator[]; imaging_findings?: ImagingFinding[] }
type Timeline = { status: string; report_count: number; indicator_count: number; timeline: TimelineReport[] }
type ReportDetail = { report_id: string; report_date: string; facility: string; source_id: string; report_text: string; original_text?: string | null; original_asset: { filename: string; media_type: string; sha256: string } | null }
type AuthAccount = { username: string; profile_id: string; csrf_token: string; is_admin: boolean }
type Comparison = { code: string; display_name: string; unit: string }
type Comparisons = { status: string; comparisons: Comparison[]; issues: Issue[] }
type CatalogItem = { code: string; display_name: string; unit: string }
type CatalogResult = { count: number; indicators: CatalogItem[] }
type AnswerSource = { kind: 'knowledge' | 'record' | 'official_web'; citation_id: string; title: string; detail: string; url: string; publisher?: string; retrieved_on?: string; updated_on?: string | null; report_id?: string; evidence_locator?: Evidence; report_links?: Array<{ report_id: string; title: string; evidence: Evidence }> }
type AnswerResult = { status: 'answered' | 'insufficient_evidence'; question: string; resolved_question?: string; answer: string; sources: AnswerSource[] }
type AnswerHistoryItem = AnswerResult & { mode: 'knowledge' | 'visit'; created_at: string }
type VisitBrief = { brief_markdown: string; report_count: number }
type AgentCapabilities = { status: string; memory: boolean; session_compaction: boolean; task_resume: boolean; mcp: boolean; mcp_scope: string; global_skills: boolean }
type AgentTask = { task_id: string; title: string; version: number; dialogue: Array<{ role: 'user' | 'assistant'; text: string }>; fold_count: number; updated_at: string | null; checkpoint: { status: string; completed_steps: string[] } | null; answer?: string; tool_names?: string[]; memory_status?: string }
type AgentTaskSummary = { task_id: string; title: string; version: number; message_count: number; fold_count: number; updated_at: string }
type TrendPoint = { date: string; value: number; reportId: string; evidence?: Evidence }
type TrendSeries = { key: string; code: string; name: string; unit: string; points: TrendPoint[] }
type ImagingGroup = { key: string; label: string; rows: Array<{ reportId: string; date: string; finding: ImagingFinding }> }
type View = 'overview' | 'reports' | 'trends' | 'imaging' | 'knowledge' | 'visit' | 'agent' | 'skills' | 'calendar'

const initialTimeline: Timeline = { status: 'empty', report_count: 0, indicator_count: 0, timeline: [] }
const initialComparisons: Comparisons = { status: 'empty', comparisons: [], issues: [] }
let csrfToken = ''

function newSourceId() {
  return `report-${new Date().toISOString().slice(0, 10)}-${crypto.randomUUID().slice(0, 8)}`
}

function newTaskId() {
  return 'task-' + new Date().toISOString().slice(0, 10) + '-' + crypto.randomUUID().slice(0, 8)
}

async function api<T>(url: string, init?: RequestInit): Promise<T> {
  const headers = new Headers(init?.headers)
  if (init?.method && init.method !== 'GET' && csrfToken) headers.set('X-CSRF-Token', csrfToken)
  const response = await fetch(url, { ...init, headers, credentials: 'same-origin' })
  const body = await response.json().catch(() => ({}))
  if (!response.ok) {
    if (response.status === 401 && !url.startsWith('/api/auth/')) window.dispatchEvent(new Event('healthvault-auth-expired'))
    const detail = typeof body.detail === 'string' ? body.detail : body.detail?.issues?.[0]?.message || body.detail?.status || body.message
    throw new Error(detail || `请求失败（${response.status}）`)
  }
  return body as T
}

function jsonBody(value: unknown): RequestInit {
  return { headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(value) }
}

function formatValue(value: number) {
  return Number.isFinite(value) ? String(value) : '—'
}

function flagLabel(flag?: string) {
  return flag === 'high' ? '高于参考' : flag === 'low' ? '低于参考' : '参考范围内'
}

function Icon({ name, size = 20 }: { name: 'home' | 'file' | 'trend' | 'book' | 'visit' | 'plus' | 'check' | 'alert' | 'arrow' | 'upload' | 'spark' | 'calendar'; size?: number }) {
  const common = { width: size, height: size, viewBox: '0 0 24 24', fill: 'none', stroke: 'currentColor', strokeWidth: 1.8, strokeLinecap: 'round' as const, strokeLinejoin: 'round' as const, 'aria-hidden': true as const }
  const paths = {
    home: <><path d="m3 10 9-7 9 7v10a1 1 0 0 1-1 1H4a1 1 0 0 1-1-1z"/><path d="M9 21v-7h6v7"/></>,
    file: <><path d="M5 3h9l5 5v13H5z"/><path d="M14 3v5h5M8 13h8M8 17h6"/></>,
    trend: <><path d="M3 20h18M5 16l5-5 4 3 5-7"/><path d="M16 7h3v3"/></>,
    book: <><path d="M12 6c-3-2-6-2-9-1v14c3-1 6-1 9 1 3-2 6-2 9-1V5c-3-1-6-1-9 1z"/><path d="M12 6v14"/></>,
    visit: <><path d="M8 3h8v4H8zM6 5H5a2 2 0 0 0-2 2v13h18V7a2 2 0 0 0-2-2h-1M7 12h10M7 16h7"/></>,
    calendar: <><rect x="3" y="5" width="18" height="16" rx="2"/><path d="M7 3v4M17 3v4M3 11h18M7 15h2M13 15h2M7 18h2"/></>,
    plus: <><path d="M12 5v14M5 12h14"/></>,
    check: <><path d="m5 12 4 4L19 6"/></>,
    alert: <><path d="M12 3 2 21h20zM12 9v5M12 18h.01"/></>,
    arrow: <><path d="M5 12h14m-6-6 6 6-6 6"/></>,
    upload: <><path d="M12 16V3m-5 5 5-5 5 5M4 16v4h16v-4"/></>,
    spark: <><path d="m12 2 1.7 6.3L20 10l-6.3 1.7L12 18l-1.7-6.3L4 10l6.3-1.7L12 2ZM19 17l.7 2.3L22 20l-2.3.7L19 23l-.7-2.3L16 20l2.3-.7L19 17Z"/></>,
  }
  return <svg {...common}>{paths[name]}</svg>
}

function buildSeries(timeline: Timeline): TrendSeries[] {
  const series = new Map<string, TrendSeries>()
  const reports = [...(timeline.timeline || [])].sort((a, b) => a.report_date.localeCompare(b.report_date) || a.report_id.localeCompare(b.report_id))
  for (const report of reports) {
    for (const item of report.indicators || []) {
      if (!Number.isFinite(item.value)) continue
      const key = `${item.code}::${item.unit}`
      if (!series.has(key)) series.set(key, { key, code: item.code, name: item.display_name, unit: item.unit, points: [] })
      series.get(key)!.points.push({ date: report.report_date, value: item.value, reportId: report.report_id, evidence: item.evidence })
    }
  }
  return [...series.values()].filter(item => item.points.length >= 2).sort((a, b) => a.name.localeCompare(b.name, 'zh-CN'))
}

function buildImagingGroups(timeline: Timeline): ImagingGroup[] {
  const groups = new Map<string, ImagingGroup>()
  for (const report of timeline.timeline || []) {
    for (const finding of report.imaging_findings || []) {
      const key = [finding.finding_type, finding.body_site, finding.laterality, finding.location].join('::')
      const type = finding.finding_type === 'fibroid' ? '肌瘤' : '结节'
      const side = finding.laterality === 'left' ? '左侧' : finding.laterality === 'right' ? '右侧' : ''
      if (!groups.has(key)) groups.set(key, { key, label: `${finding.body_site}${side}${finding.location} ${type}`, rows: [] })
      groups.get(key)!.rows.push({ reportId: report.report_id, date: report.report_date, finding })
    }
  }
  return [...groups.values()].map(group => ({ ...group, rows: group.rows.sort((a, b) => b.date.localeCompare(a.date)) }))
    .sort((a, b) => a.label.localeCompare(b.label, 'zh-CN'))
}

export default function App() {
  const [view, setView] = useState<View>('overview')
  const [account, setAccount] = useState<AuthAccount | null>(null)
  const [authReady, setAuthReady] = useState(false)
  const [authMode, setAuthMode] = useState<'login' | 'register'>('login')
  const [authUsername, setAuthUsername] = useState('')
  const [authPassword, setAuthPassword] = useState('')
  const [extractionConsent, setExtractionConsent] = useState(false)
  const [answerConsent, setAnswerConsent] = useState(false)
  const [allowWebSearch, setAllowWebSearch] = useState(false)
  const [sourceId, setSourceId] = useState(newSourceId)
  const [reportText, setReportText] = useState('')
  const [sourceOpen, setSourceOpen] = useState(false)
  const [draft, setDraft] = useState<Draft | null>(null)
  const [uploads, setUploads] = useState<ReportUpload[]>([])
  const [uploadFilter, setUploadFilter] = useState('active')
  const [isImage, setIsImage] = useState(false)
  const [ocrReviewed, setOcrReviewed] = useState(false)
  const [timeline, setTimeline] = useState<Timeline>(initialTimeline)
  const [selectedImagingReport, setSelectedImagingReport] = useState('')
  const [reportDetail, setReportDetail] = useState<ReportDetail | null>(null)
  const [selectedEvidence, setSelectedEvidence] = useState<Evidence | null>(null)
  const [comparisons, setComparisons] = useState<Comparisons>(initialComparisons)
  const [catalog, setCatalog] = useState<CatalogItem[]>([])
  const [catalogQuery, setCatalogQuery] = useState('')
  const [selectedTrend, setSelectedTrend] = useState('')
  const [knowledgeQuestion, setKnowledgeQuestion] = useState('')
  const [knowledgeAnswer, setKnowledgeAnswer] = useState<AnswerResult | null>(null)
  const [answerHistory, setAnswerHistory] = useState<AnswerHistoryItem[]>([])
  const [visitQuestion, setVisitQuestion] = useState('')
  const [visitAnswer, setVisitAnswer] = useState<AnswerResult | null>(null)
  const [visitBrief, setVisitBrief] = useState<VisitBrief | null>(null)
  const [agentTaskId, setAgentTaskId] = useState(newTaskId)
  const [agentTask, setAgentTask] = useState<AgentTask | null>(null)
  const [agentTaskList, setAgentTaskList] = useState<AgentTaskSummary[]>([])
  const [agentCapabilities, setAgentCapabilities] = useState<AgentCapabilities | null>(null)
  const [agentPrompt, setAgentPrompt] = useState('')
  const [workflowCorrection, setWorkflowCorrection] = useState('')
  const [workflowConfirmed, setWorkflowConfirmed] = useState(false)
  const [workflowConsent, setWorkflowConsent] = useState(false)
  const [busy, setBusy] = useState(false)
  const [message, setMessage] = useState('')
  const [error, setError] = useState('')
  const [failedRefresh, setFailedRefresh] = useState<{ message: string; run: () => Promise<void> } | null>(null)
  const agentInputRevision = useRef(0)
  const fileRef = useRef<HTMLInputElement>(null)
  const profile = account?.profile_id || ''

  useEffect(() => {
    let active = true
    api<AuthAccount>('/api/auth/me').then(user => {
      if (active) { csrfToken = user.csrf_token; setAccount(user) }
    }).catch(() => { if (active) setAccount(null) })
      .finally(() => { if (active) setAuthReady(true) })
    const expired = () => { csrfToken = ''; setAccount(null); setAuthMode('login'); setError('登录已过期，请重新登录。') }
    window.addEventListener('healthvault-auth-expired', expired)
    return () => { active = false; window.removeEventListener('healthvault-auth-expired', expired) }
  }, [])

  useEffect(() => {
    if (!profile) return
    let active = true
    api<CatalogResult>('/api/indicators')
      .then(result => { if (active) setCatalog(result.indicators) })
      .catch((reason: Error) => { if (active) setError(reason.message) })
    return () => { active = false }
  }, [profile])

  useEffect(() => {
    setDraft(null); setReportText(''); setSourceId(newSourceId()); setSourceOpen(false)
    setUploads([])
    setUploadFilter('active'); setSelectedEvidence(null)
    setFailedRefresh(null)
    setIsImage(false); setOcrReviewed(false); setVisitAnswer(null); setKnowledgeAnswer(null); setAnswerHistory([])
    setVisitBrief(null)
    setKnowledgeQuestion(''); setVisitQuestion(''); setExtractionConsent(false); setAnswerConsent(false); setAllowWebSearch(false)
    setAgentTaskId(newTaskId()); setAgentTask(null); setAgentTaskList([]); setAgentCapabilities(null); setAgentPrompt('')
    setWorkflowCorrection(''); setWorkflowConfirmed(false); setWorkflowConsent(false)
    setTimeline(initialTimeline); setComparisons(initialComparisons); setSelectedImagingReport(''); setReportDetail(null)
    if (!profile) return
    let active = true
    Promise.all([
      api<Timeline>(`/api/profiles/${encodeURIComponent(profile)}/timeline`),
      api<Comparisons>(`/api/profiles/${encodeURIComponent(profile)}/comparisons`),
    ]).then(([history, trends]) => {
      if (active) { setTimeline(history); setComparisons(trends) }
    }).catch((reason: Error) => { if (active) setError(reason.message) })
    Promise.all([
      api<{ tasks: AgentTaskSummary[] }>('/api/profiles/' + encodeURIComponent(profile) + '/agent/tasks'),
      api<AgentCapabilities>('/api/profiles/' + encodeURIComponent(profile) + '/agent/capabilities'),
    ]).then(([tasks, capabilities]) => {
      if (active) { setAgentTaskList(tasks.tasks); setAgentCapabilities(capabilities) }
    }).catch((reason: Error) => { if (active) setError(reason.message) })
    api<{ answers: AnswerHistoryItem[] }>('/api/profiles/' + encodeURIComponent(profile) + '/answers/history')
      .then(result => { if (active) setAnswerHistory(result.answers) })
      .catch((reason: Error) => { if (active) setError(reason.message) })
    api<{ uploads: ReportUpload[] }>('/api/profiles/' + encodeURIComponent(profile) + '/uploads?status=all')
      .then(result => { if (active) setUploads(result.uploads) })
      .catch((reason: Error) => { if (active) setError(reason.message) })
    return () => { active = false }
  }, [profile])

  useEffect(() => {
    if (view !== 'imaging' || !profile) return
    const reportId = selectedImagingReport || [...timeline.timeline].sort((a, b) => b.report_date.localeCompare(a.report_date))[0]?.report_id
    if (!reportId) { setReportDetail(null); return }
    let active = true
    setReportDetail(null)
    api<ReportDetail>('/api/profiles/' + encodeURIComponent(profile) + '/reports/' + encodeURIComponent(reportId))
      .then(detail => { if (active) setReportDetail(detail) })
      .catch((reason: Error) => { if (active) setError(reason.message) })
    return () => { active = false }
  }, [view, profile, selectedImagingReport, timeline])

  useEffect(() => {
    if (view !== 'visit' || !profile) return
    let active = true
    setVisitBrief(null)
    api<VisitBrief>('/api/profiles/' + encodeURIComponent(profile) + '/visit-brief')
      .then(result => { if (active) setVisitBrief(result) })
      .catch((reason: Error) => { if (active) setError(reason.message) })
    return () => { active = false }
  }, [view, profile, timeline])

  const latest = useMemo(() => [...(timeline.timeline || [])].sort((a, b) => b.report_date.localeCompare(a.report_date))[0], [timeline])
  const series = useMemo(() => buildSeries(timeline), [timeline])
  const imagingGroups = useMemo(() => buildImagingGroups(timeline), [timeline])
  const matchingCatalog = useMemo(() => catalog.filter(item =>
    `${item.display_name} ${item.code} ${item.unit}`.toLocaleLowerCase().includes(catalogQuery.trim().toLocaleLowerCase())
  ), [catalog, catalogQuery])
  const activeSeries = series.find(item => item.key === selectedTrend) || series[0]
  const activeImagingReportId = selectedImagingReport || latest?.report_id || ''
  const activeImagingRecord = timeline.timeline.find(item => item.report_id === activeImagingReportId)
  const hasFreshPreview = Boolean(draft?.preview && draft.preview_sha256 && draft.report_text === reportText && draft.source_id === sourceId)
  const canConfirm = Boolean(hasFreshPreview && draft?.preview?.can_confirm && !draft.archived_at && (!isImage || ocrReviewed))
  const filteredUploads = uploads.filter(item => uploadFilter === 'archived' ? Boolean(item.archived_at) : !item.archived_at &&
    (uploadFilter === 'active' || (uploadFilter === 'failed' ? item.processing_status === 'extraction_failed' : item.processing_status !== 'extraction_failed')))
  const [discardAction, setDiscardAction] = useState<(() => void) | null>(null)
  const discardDialog = useRef<HTMLDialogElement>(null)
  useEffect(() => {
    if (discardAction) discardDialog.current?.showModal()
    else discardDialog.current?.close()
  }, [discardAction])
  const hasUnsavedEdits = draft ? draft.report_text !== reportText || draft.source_id !== sourceId : reportText.length > 0
  useEffect(() => {
    if (!hasUnsavedEdits) return
    const warn = (event: BeforeUnloadEvent) => { event.preventDefault(); event.returnValue = '' }
    window.addEventListener('beforeunload', warn)
    return () => window.removeEventListener('beforeunload', warn)
  }, [hasUnsavedEdits])
  function mayDiscardEdits(retry: () => void, approved: boolean) {
    if (!hasUnsavedEdits || approved) return true
    setDiscardAction(() => retry)
    return false
  }

  async function refreshHistory() {
    const [history, trends] = await Promise.all([
      api<Timeline>(`/api/profiles/${encodeURIComponent(profile)}/timeline`),
      api<Comparisons>(`/api/profiles/${encodeURIComponent(profile)}/comparisons`),
    ])
    setTimeline(history); setComparisons(trends); setVisitAnswer(null)
  }

  async function refreshUploads() {
    const result = await api<{ uploads: ReportUpload[] }>('/api/profiles/' + encodeURIComponent(profile) + '/uploads?status=all')
    setUploads(result.uploads)
  }

  async function refreshAfterSuccess(run: () => Promise<void>, failureMessage = '已保存，列表更新失败。') {
    try { await run(); setFailedRefresh(null) }
    catch { setFailedRefresh({ message: failureMessage, run }) }
  }

  async function retryRefresh() {
    if (!failedRefresh) return
    setBusy(true)
    try { await failedRefresh.run(); setFailedRefresh(null); setMessage('列表已更新。') }
    catch { /* Keep the saved result and the refresh-only action. */ }
    finally { setBusy(false) }
  }

  async function refreshAnswers() {
    const saved = await api<{ answers: AnswerHistoryItem[] }>('/api/profiles/' + encodeURIComponent(profile) + '/answers/history')
    setAnswerHistory(saved.answers)
  }

  async function refreshAgentTasks() {
    const tasks = await api<{ tasks: AgentTaskSummary[] }>('/api/profiles/' + encodeURIComponent(profile) + '/agent/tasks')
    setAgentTaskList(tasks.tasks)
  }

  function openReport(reportId: string, evidence?: Evidence) {
    if (reportDetail?.report_id !== reportId) setReportDetail(null)
    setSelectedImagingReport(reportId); setSelectedEvidence(evidence || null); setView('imaging')
  }

  async function organizeUpload(item: ReportUpload, discardApproved = false) {
    if (draft?.token === item.token && !mayDiscardEdits(() => void organizeUpload(item, true), discardApproved)) return
    setBusy(true); setError(''); setMessage('')
    try {
      const result = await api<Draft>('/api/drafts/' + encodeURIComponent(item.token) + (item.archived_at ? '/restore' : '/archive'), { method: 'POST' })
      if (draft?.token === item.token) {
        if (result.archived_at) { setDraft(null); setReportText(''); setSourceId(newSourceId()); setSourceOpen(false); setIsImage(false); setOcrReviewed(false) }
        else setDraft(result)
      }
      setMessage(result.archived_at ? '草稿已归档，原件和文字仍保留，可随时恢复。' : '草稿已恢复，可以继续核对。')
      await refreshAfterSuccess(refreshUploads)
    } catch (reason) { setError((reason as Error).message) }
    finally { setBusy(false) }
  }

  async function openUpload(token: string, discardApproved = false) {
    if (!mayDiscardEdits(() => void openUpload(token, true), discardApproved)) return
    setBusy(true); setError(''); setMessage('')
    try {
      const result = await api<Draft>('/api/drafts/' + encodeURIComponent(token) + '?profile_id=' + encodeURIComponent(profile))
      setDraft(result); setSourceId(result.source_id || ''); setReportText(result.report_text)
      setIsImage(result.input_kind === 'image' || result.input_kind === 'pdf'); setOcrReviewed(Boolean(result.ocr_reviewed))
      setExtractionConsent(false); setSourceOpen(true); setView('reports')
      setMessage('已打开保存的原件和待核对内容。')
    } catch (reason) { setError((reason as Error).message) }
    finally { setBusy(false) }
  }

  async function saveOriginal() {
    if (!reportText.trim() || !sourceId.trim()) return
    setBusy(true); setError(''); setMessage('')
    try {
      const result = draft && draft.source_id === sourceId
        ? await api<Draft>('/api/drafts/' + encodeURIComponent(draft.token), { method: 'PUT', ...jsonBody({ profile_id: profile, report_text: reportText, reviewed: ocrReviewed, model_consent: false }) })
        : await api<Draft>('/api/drafts/text', { method: 'POST', ...jsonBody({ profile_id: profile, source_id: sourceId, report_text: reportText, auto_extract: false }) })
      setDraft(result); setMessage('原文已保存，可稍后继续核对和抽取。')
      await refreshAfterSuccess(refreshUploads)
    } catch (reason) { setError((reason as Error).message) }
    finally { setBusy(false) }
  }

  async function askKnowledge() {
    if (knowledgeQuestion.trim().length < 3) { setError('请输入至少 3 个字的问题。'); return }
    if (!answerConsent) { setError('请先确认本次提问会发送到模型服务。'); return }
    setBusy(true); setError(''); setKnowledgeAnswer(null)
    try {
      setKnowledgeAnswer(await api<AnswerResult>('/api/answers/knowledge', { method: 'POST', ...jsonBody({ question: knowledgeQuestion.trim(), context_question: knowledgeAnswer?.resolved_question || knowledgeAnswer?.question || '', allow_query_rewrite: true, model_consent: answerConsent, allow_web_search: allowWebSearch }) }))
      await refreshAfterSuccess(refreshAnswers, '回答已显示，历史列表更新失败。')
    } catch (reason) { setError((reason as Error).message) }
    finally { setBusy(false) }
  }

  async function askVisit() {
    if (visitQuestion.trim().length < 3) { setError('请输入至少 3 个字的问题。'); return }
    if (!answerConsent) { setError('请先确认本次提问和必要记录证据会发送到模型服务。'); return }
    setBusy(true); setError(''); setVisitAnswer(null)
    try {
      setVisitAnswer(await api<AnswerResult>(`/api/profiles/${encodeURIComponent(profile)}/answers/visit`, { method: 'POST', ...jsonBody({ question: visitQuestion.trim(), context_question: visitAnswer?.resolved_question || visitAnswer?.question || '', model_consent: answerConsent }) }))
      await refreshAfterSuccess(refreshAnswers, '回答已显示，历史列表更新失败。')
    } catch (reason) { setError((reason as Error).message) }
    finally { setBusy(false) }
  }

  async function loadAgentTask(taskId: string) {
    if (!taskId.trim()) return
    setBusy(true); setError('')
    try {
      const task = await api<AgentTask>('/api/profiles/' + encodeURIComponent(profile) + '/agent/tasks/' + encodeURIComponent(taskId.trim()))
      setAgentTaskId(task.task_id); setAgentTask(task)
      setWorkflowCorrection(''); setWorkflowConfirmed(false); setWorkflowConsent(false)
    } catch (reason) { setError((reason as Error).message) }
    finally { setBusy(false) }
  }

  async function sendAgentMessage() {
    if (agentPrompt.trim().length < 3) return
    if (!answerConsent) { setError('请先确认本次对话会发送到模型服务。'); return }
    const submittedPrompt = agentPrompt
    const submittedRevision = agentInputRevision.current
    setBusy(true); setError('')
    try {
      const url = '/api/profiles/' + encodeURIComponent(profile) + '/agent/tasks/' + encodeURIComponent(agentTaskId) + '/messages'
      const task = await api<AgentTask>(url, {
        method: 'POST', ...jsonBody({ prompt: submittedPrompt.trim(), expected_version: agentTask?.version || 0, model_consent: answerConsent, allow_web_search: allowWebSearch }),
      })
      setAgentTask(task)
      if (agentInputRevision.current === submittedRevision) setAgentPrompt(current => current === submittedPrompt ? '' : current)
      await refreshAfterSuccess(refreshAgentTasks, '对话已保存，历史列表更新失败。')
    } catch (reason) { setError((reason as Error).message) }
    finally { setBusy(false) }
  }

  async function compactAgentTask() {
    if (!agentTask) return
    if (!answerConsent) { setError('整理上下文会使用模型服务，请先确认。'); return }
    setBusy(true); setError('')
    try {
      const url = '/api/profiles/' + encodeURIComponent(profile) + '/agent/tasks/' + encodeURIComponent(agentTask.task_id) + '/compact'
      const task = await api<AgentTask>(url, {
        method: 'POST', ...jsonBody({ expected_version: agentTask.version, model_consent: answerConsent }),
      })
      setAgentTask(task); setMessage('已整理较长的对话，可以继续提问。')
    } catch (reason) { setError((reason as Error).message) }
    finally { setBusy(false) }
  }

  async function submitWorkflowCorrection() {
    if (!agentTask || !workflowConfirmed || !workflowConsent) return
    setBusy(true); setError(''); setMessage('')
    try {
      const result = await api<{ status: string; reason?: string; candidate?: { status: string } }>('/api/profiles/' + encodeURIComponent(profile) + '/agent/tasks/' + encodeURIComponent(agentTask.task_id) + '/workflow-feedback', {
        method: 'POST', ...jsonBody({ correction: workflowCorrection.trim(), confirmed: workflowConfirmed, model_consent: workflowConsent }),
      })
      setMessage(result.status === 'candidate_created' ? result.candidate?.status === 'published' ? '个人工作流已通过检查，只对你生效。' : '纠正已保存，可在我的工作流查看验证状态。' : result.reason || '纠正未达到候选门槛。')
      setWorkflowCorrection(''); setWorkflowConfirmed(false); setWorkflowConsent(false)
    } catch (reason) { setError((reason as Error).message) }
    finally { setBusy(false) }
  }

  async function uploadFile(file: File, discardApproved = false) {
    if (!mayDiscardEdits(() => void uploadFile(file, true), discardApproved)) return
    if (!profile) { setError('请先填写档案 ID'); return }
    setBusy(true); setError(''); setMessage('')
    try {
      const form = new FormData()
      form.append('profile_id', profile); form.append('file', file)
      const result = await api<Draft>(file.name.toLowerCase().endsWith('.pdf') ? '/api/drafts/pdf' : '/api/drafts/image', { method: 'POST', body: form })
      setDraft(result); setReportText(result.report_text || '')
      setSourceId(result.source_id || `image:${result.image_sha256 || file.name}`)
      setIsImage(true); setOcrReviewed(false); setSourceOpen(true); setView('reports')
      setMessage(result.processing_error ? '原件已保存，请根据识别提示继续核对或补充原文。' : '原件与识别文字已保存。请对照原件核对文字，再生成预览。')
      await refreshAfterSuccess(refreshUploads)
    } catch (reason) { setError((reason as Error).message) }
    finally { setBusy(false); if (fileRef.current) fileRef.current.value = '' }
  }

  async function previewReport() {
    if (!profile || !reportText.trim() || !sourceId.trim()) { setError('请填写报告文本和来源 ID。'); setSourceOpen(true); return }
    if (isImage && !ocrReviewed) { setError('请先核对 OCR 文字并勾选“已核对”。'); setSourceOpen(true); return }
    const needsModel = !(reportText.includes('result|') && reportText.includes('report_id:'))
    setBusy(true); setError(''); setMessage('')
    try {
      const result = draft && draft.source_id === sourceId
        ? await api<Draft>(`/api/drafts/${encodeURIComponent(draft.token)}`, { method: 'PUT', ...jsonBody({ profile_id: profile, report_text: reportText, reviewed: ocrReviewed, model_consent: extractionConsent }) })
        : await api<Draft>('/api/drafts/text', { method: 'POST', ...jsonBody({ profile_id: profile, source_id: sourceId, report_text: reportText, auto_extract: needsModel && extractionConsent, model_consent: extractionConsent }) })
      setDraft(result); setSourceId(result.source_id || sourceId); setReportText(result.report_text || reportText)
      setSourceOpen(false)
      setMessage(result.processing_error ? '原件和文字已保存，抽取未完成，请稍后重试。' : result.preview?.can_confirm ? '预览已生成，请核对字段和原文证据。' : '原文已保存，请根据提示继续核对。')
      await refreshAfterSuccess(refreshUploads)
    } catch (reason) { setError((reason as Error).message) }
    finally { setBusy(false) }
  }

  async function confirmReport() {
    if (!draft || !canConfirm) return
    setBusy(true); setError(''); setMessage('')
    try {
      const result = await api<{ status: string; confirmed?: boolean }>(`/api/drafts/${encodeURIComponent(draft.token)}/confirm`, {
        method: 'POST', ...jsonBody({ profile_id: profile, report_text: reportText, preview_sha256: draft.preview_sha256, acknowledged: true }),
      })
      if (result.status !== 'confirmed' && !result.confirmed) throw new Error(`确认未完成：${result.status}`)
      setMessage('报告已确认并保存到当前档案。明示复诊计划可在健康日历中核对安排。')
      setDraft(null); setReportText(''); setSourceId(newSourceId()); setSourceOpen(false)
      setOcrReviewed(false); setIsImage(false)
      setView('overview')
      await refreshAfterSuccess(async () => { await Promise.all([refreshHistory(), refreshUploads()]) })
    } catch (reason) { setError((reason as Error).message) }
    finally { setBusy(false) }
  }

  function startNew(discardApproved = false) {
    if (!mayDiscardEdits(() => startNew(true), discardApproved)) return
    setDraft(null); setReportText(''); setSourceId(newSourceId()); setSourceOpen(false)
    setIsImage(false); setOcrReviewed(false); setError(''); setMessage(''); setView('reports')
  }

  async function submitAuth() {
    setBusy(true); setError('')
    try {
      const user = await api<AuthAccount>('/api/auth/' + authMode, {
        method: 'POST', ...jsonBody({ username: authUsername.trim(), password: authPassword }),
      })
      csrfToken = user.csrf_token
      setAccount(user); setAuthPassword(''); setMessage('')
    } catch (reason) { setError((reason as Error).message) }
    finally { setBusy(false) }
  }

  async function logout(discardApproved = false) {
    if (!mayDiscardEdits(() => void logout(true), discardApproved)) return
    setBusy(true); setError('')
    try {
      await api<{ logged_out: boolean }>('/api/auth/logout', { method: 'POST' })
      csrfToken = ''; setAccount(null); setAuthPassword(''); setExtractionConsent(false); setAnswerConsent(false); setView('overview')
    } catch (reason) { setError((reason as Error).message) }
    finally { setBusy(false) }
  }

  if (!authReady) return <div className="auth-shell"><div className="auth-card">正在加载 HealthVault…</div></div>
  if (!account) return <div className="auth-shell"><div className="auth-card"><span className="eyebrow">PERSONAL HEALTH RECORD</span><h1>HealthVault</h1><p>登录后查看自己的报告、原始文件与持续对话。</p>
    {error && <div className="notice error" role="alert">{error}</div>}
    <form onSubmit={event => { event.preventDefault(); void submitAuth() }}>
      <label className="field-label">用户名<input autoComplete="username" value={authUsername} onChange={event => setAuthUsername(event.target.value)} minLength={3} maxLength={40} required/></label>
      <label className="field-label">密码<input type="password" autoComplete={authMode === 'register' ? 'new-password' : 'current-password'} value={authPassword} onChange={event => setAuthPassword(event.target.value)} minLength={authMode === 'register' ? 12 : 1} maxLength={128} required/></label>
      <button type="submit" className="primary-button" disabled={busy}>{busy ? '请稍候…' : authMode === 'login' ? '登录' : '创建账户'}</button>
    </form>
    <button type="button" className="text-button auth-switch" disabled={busy} onClick={() => { setAuthMode(authMode === 'login' ? 'register' : 'login'); setError('') }}>{authMode === 'login' ? '没有账户？创建账户' : '已有账户？登录'}</button>
    <p className="auth-note">账户之间的数据独立保存。创建账户后需要重新导入属于自己的报告。</p>
  </div></div>

  const pageTitle = { overview: '健康概览', reports: '报告核对', trends: '指标趋势', imaging: '报告原件', knowledge: '知识问答', visit: '就诊准备', agent: '持续对话', skills: '工作流与知识管理', calendar: '健康日历' }[view]

  return <div className="app-shell">
    <dialog ref={discardDialog} className="unsaved-dialog" aria-labelledby="unsaved-title" aria-describedby="unsaved-description" onCancel={() => setDiscardAction(null)}>
      <h2 id="unsaved-title">报告编辑尚未保存</h2>
      <p id="unsaved-description">继续会放弃尚未保存的编辑。你也可以继续编辑，先保存原文。</p>
      <div className="review-actions">
        <button type="button" autoFocus className="subtle-button" onClick={() => setDiscardAction(null)}>继续编辑</button>
        <button type="button" className="primary-button" onClick={() => { const pending = discardAction; setDiscardAction(null); pending?.() }}>放弃编辑并继续</button>
      </div>
    </dialog>
    <aside className="sidebar">
      <a className="brand" href="#overview" onClick={event => { event.preventDefault(); setView('overview') }} aria-label="HealthVault 首页">
        <span className="brand-mark"><span /></span><span>Health<span className="brand-accent">Vault</span><small>个人健康档案</small></span>
      </a>
      <div className="sidebar-label">工作台</div>
      <nav aria-label="主导航" className="nav-list">
        {([
          ['overview', 'home', '概览'], ['reports', 'file', '报告核对'], ['trends', 'trend', '指标趋势'],
          ['imaging', 'file', '报告原件'], ['calendar', 'calendar', '健康日历'], ['knowledge', 'book', '知识问答'], ['visit', 'visit', '就诊准备'], ['agent', 'spark', '持续对话'],
        ] as const).map(([id, icon, label]) => <button key={id} type="button" aria-current={view === id ? 'page' : undefined} className={view === id ? 'nav-item active' : 'nav-item'} onClick={() => setView(id)}><Icon name={icon}/>{label}</button>)}
        <button type="button" aria-current={view === 'skills' ? 'page' : undefined} className={view === 'skills' ? 'nav-item active' : 'nav-item'} onClick={() => setView('skills')}><Icon name="check"/>{account.is_admin ? '工作流管理' : '我的工作流'}</button>
      </nav>
      <div className="sidebar-bottom"><div className="privacy-note"><span className="privacy-dot"/>本地档案<br/><small>问答会向已配置模型服务发送问题及必要证据</small></div></div>
    </aside>

    <main className="main-content">
      <header className="topbar">
        <div><span className="eyebrow">PERSONAL HEALTH RECORD</span><h1>{pageTitle}</h1></div>
        <div className="account-control"><span>当前账户</span><strong>{account.username}</strong><button type="button" disabled={busy} onClick={() => void logout()}>退出</button></div>
      </header>
      {error && <div className="notice error" role="alert"><Icon name="alert" size={18}/>{error}<button type="button" onClick={() => setError('')} aria-label="关闭错误">×</button></div>}
      {message && <div className="notice success" role="status"><Icon name="check" size={18}/>{message}<button type="button" onClick={() => setMessage('')} aria-label="关闭提示">×</button></div>}
      {failedRefresh && <div className="notice refresh-warning" role="status"><Icon name="alert" size={18}/>{failedRefresh.message}<button type="button" className="subtle-button" disabled={busy} onClick={() => void retryRefresh()}>刷新列表</button></div>}
      {view === 'skills' && <><PersonalSkillPanel request={api}/>{account.is_admin && <><SharedExperiencePanel request={api}/><SkillReviewPanel request={api}/><KnowledgeReviewPanel request={api}/></>}</>}
      {view === 'calendar' && <CalendarPanel key={profile} profile={profile} request={api} onOpenReport={id => openReport(id)}/>}
      {(view === 'reports' || view === 'imaging') && <section className="panel pending-uploads"><div className="panel-heading"><div><span className="eyebrow">SAVED SOURCES</span><h2>已保存的草稿</h2><p>归档会整理列表并保留原件；清空编辑器不会删除已保存的草稿。</p></div><label className="upload-filter">显示<select value={uploadFilter} onChange={event => setUploadFilter(event.target.value)}><option value="active">全部待处理</option><option value="pending">待核对</option><option value="failed">处理失败</option><option value="archived">已归档</option></select></label></div><div className="pending-upload-list">{filteredUploads.map(item => <div className="upload-row" key={item.token}><button type="button" className="agent-task-choice" disabled={busy} onClick={() => void openUpload(item.token)}><strong>{item.image_filename || item.source_id}</strong><small>{item.archived_at ? '已归档' : item.processing_status === 'extraction_failed' ? '处理失败 · 原件已保存' : '待核对'} · {new Date(item.updated_at).toLocaleString('zh-CN')}{item.duplicate_of ? ' · 文件或文字已上传过' : ''}</small></button><button type="button" className="subtle-button" disabled={busy} onClick={() => void organizeUpload(item)}>{item.archived_at ? '恢复草稿' : '归档草稿'}</button></div>)}</div>{!filteredUploads.length && <p className="muted-empty">这个分类下没有草稿。</p>}</section>}

      {view === 'overview' && <>
        <section className="hero-card">
          <div><span className="hero-kicker">你的健康记录，一目了然</span><h2>把检查结果整理成<br/>可追溯的健康时间线。</h2><p>上传报告，核对识别结果后保存；随后可比较指标并准备就诊问题。</p><button type="button" className="primary-button" onClick={() => startNew()}><Icon name="plus" size={18}/>添加报告</button></div>
          <div className="hero-graphic" aria-hidden="true"><div className="graphic-ring"><span className="graphic-pulse"/></div><span className="graphic-line one"/><span className="graphic-line two"/></div>
        </section>
        <section className="metric-grid" aria-label="档案统计">
          <div className="metric-card"><span>已确认报告</span><strong>{timeline.report_count ?? 0}</strong><small>份已核对并归档</small></div>
          <div className="metric-card"><span>结构化指标</span><strong>{timeline.indicator_count ?? 0}</strong><small>条可追溯结果</small></div>
          <div className="metric-card"><span>可比较指标</span><strong>{series.length}</strong><small>项拥有多次同单位记录</small></div>
        </section>
        <section className="two-column">
          <div className="panel"><div className="panel-heading"><div><span className="eyebrow">RECENT RECORD</span><h2>最近报告</h2></div><button type="button" className="text-button" onClick={() => setView('reports')}>查看详情 <Icon name="arrow" size={16}/></button></div>
            {latest ? <div className="recent-report"><div className="date-tile"><strong>{latest.report_date.slice(8, 10)}</strong><small>{latest.report_date.slice(0, 7)}</small></div><div><h3>{latest.facility || '检查报告'}</h3><p>{latest.report_date} · {latest.indicators.length} 项指标</p></div></div> : <EmptyState title="还没有已确认的报告" detail="添加一份报告，核对后就会显示在这里。" action={() => startNew()}/>}
          </div>
          <div className="panel"><div className="panel-heading"><div><span className="eyebrow">AT A GLANCE</span><h2>最近指标</h2></div><button type="button" className="text-button" onClick={() => setView('trends')}>查看趋势 <Icon name="arrow" size={16}/></button></div>
            {latest?.indicators?.length ? <div className="compact-indicators">{latest.indicators.slice(0, 4).map(item => <div className="compact-row" key={item.code}><span>{item.display_name}</span><strong>{formatValue(item.value)} <small>{item.unit}</small></strong></div>)}</div> : <p className="muted-empty">确认报告后即可查看指标。</p>}
          </div>
        </section>
      </>}

      {view === 'reports' && <div className="report-layout">
        <section className="panel editor-panel"><div className="panel-heading"><div><span className="eyebrow">STEP 01 · SOURCE</span><h2>添加并核对报告</h2><p>上传图片或粘贴文本，确认前先检查提取结果。</p></div><button type="button" className="subtle-button" onClick={() => startNew()}>清空</button></div>
          {draft?.duplicate_of && <p className="review-hint" role="status">这份文件或文字已上传过：{draft.duplicate_of.image_filename || draft.duplicate_of.source_id}。请核对草稿或已确认档案，避免重复操作。</p>}
          {draft?.archived_at && <p className="review-hint">这份草稿已归档。可在上方列表恢复后继续编辑。</p>}
          <input ref={fileRef} type="file" accept="image/png,image/jpeg,application/pdf" hidden onChange={event => { const file = event.target.files?.[0]; if (file) void uploadFile(file) }}/>
          <button type="button" className="upload-zone" onClick={() => fileRef.current?.click()} disabled={busy}><Icon name="upload" size={26}/><strong>上传体检或影像报告</strong><span>PDF、PNG、JPG · 自动识别后请人工核对</span></button>
          <div className="or-divider"><span>或</span></div>
          <button type="button" className="source-toggle" aria-expanded={sourceOpen} onClick={() => setSourceOpen(!sourceOpen)}><span><strong>{isImage ? '核对原图与 OCR 文本' : reportText ? '查看或编辑报告原文' : '粘贴报告文本'}</strong><small>{reportText ? `已读取 ${reportText.length} 个字符 · ${sourceOpen ? '正在查看来源信息' : '来源信息已收起'}` : '来源 ID 会自动生成，可在此修改'}</small></span><span className="toggle-chevron">⌄</span></button>
          {sourceOpen && <div className="source-fields">
            {isImage && draft?.image_url && <div className="source-image-wrap"><span>上传原件</span>{draft.input_kind === 'pdf' ? <a href={draft.image_url} target="_blank" rel="noopener noreferrer">打开 PDF 原件核对</a> : <img className="source-image" src={draft.image_url} alt="待核对的原始报告图片"/>}</div>}
            <label className="field-label">来源 ID<input value={sourceId} onChange={event => setSourceId(event.target.value)} disabled={Boolean(draft?.archived_at) || (isImage && Boolean(draft))}/></label>
            <label className="field-label">报告原文<textarea value={reportText} readOnly={Boolean(draft?.archived_at)} onChange={event => setReportText(event.target.value)} placeholder={'report_id: HV-001\nreport_date: 2026-09-27\nfacility: 体检中心\nresult|fasting_glucose|5.2|mmol/L|3.9|6.1'} rows={12}/></label>
            {isImage && <label className="review-check"><input type="checkbox" checked={ocrReviewed} onChange={event => setOcrReviewed(event.target.checked)}/>我已对照原件核对识别文字、数值、单位和日期</label>}
          </div>}
          <label className="review-check model-consent"><input type="checkbox" checked={extractionConsent} onChange={event => setExtractionConsent(event.target.checked)}/>我同意将本次报告文字发送至已配置模型服务，用于提出待核对的字段候选；图片 OCR 在本机完成。</label>
          <button type="button" className="subtle-button full" onClick={() => void saveOriginal()} disabled={busy || !reportText.trim() || Boolean(draft?.archived_at)}>保存原文，稍后核对</button>
          <button type="button" className="primary-button full" onClick={() => void previewReport()} disabled={busy || !reportText.trim() || Boolean(draft?.archived_at)}>{busy ? '处理中…' : '生成核对预览'}<Icon name="arrow" size={18}/></button>
          <p className="review-hint">常见指标先用本地规则核对；选择上方同意后，模型可补充复杂字段候选。无法结构化的原文保留待核对。</p>
          <details className="catalog-disclosure">
            <summary>支持的体检数值指标 <strong>{catalog.length} 项</strong></summary>
            <div className="catalog-body">
              <label htmlFor="catalog-search">查找指标</label>
              <input id="catalog-search" type="search" value={catalogQuery} onChange={event => setCatalogQuery(event.target.value)} placeholder="输入指标名称、代码或单位" />
              <p>以下项目可从规范化结果行识别；仍需核对原文、单位和报告参考范围。</p>
              <div className="catalog-list">
                {matchingCatalog.map(item => <div className="catalog-item" key={item.code}><span>{item.display_name}<small>{item.code}</small></span><strong>{item.unit}</strong></div>)}
                {!matchingCatalog.length && <p>没有匹配的指标。</p>}
              </div>
            </div>
          </details>
        </section>
        <section className="panel preview-panel"><div className="panel-heading"><div><span className="eyebrow">STEP 02 · REVIEW</span><h2>核对预览</h2></div>{draft?.preview && <span className={draft.preview.can_confirm ? 'status-pill ready' : 'status-pill issue'}>{draft.preview.can_confirm ? '可确认' : '需修正'}</span>}</div>
          {draft?.saved && <p className="review-hint">原件和原文已保存。结构化字段需核对确认后才进入时间线。</p>}
          {draft?.processing_error && <p className="notice error" role="alert">{draft.processing_error}</p>}
          {draft?.original_text && draft.original_text !== draft.report_text && <details className="evidence"><summary>最初识别或上传的原文</summary><pre className="review-code">{draft.original_text}</pre></details>}
          {!draft?.preview ? <EmptyState title={isImage ? '等待核对 OCR 文本' : '等待生成预览'} detail="只有确认后的结构化字段才会写入档案。"/> : <>
            {!hasFreshPreview && <div className="inline-warning">原文或来源已变化，请重新生成预览。</div>}
            <div className="report-meta"><div><span>报告编号</span><strong>{draft.preview.report?.report_id || '—'}</strong></div><div><span>报告日期</span><strong>{draft.preview.report?.report_date || '—'}</strong></div><div><span>机构</span><strong>{draft.preview.report?.facility || '—'}</strong></div></div>
            {(draft.preview.issues || []).map((issue, index) => <div className="issue-card" key={`${issue.code}-${index}`}><Icon name="alert" size={17}/><div><strong>{issue.code}</strong><p>{issue.message}{issue.line ? ` · 第 ${issue.line} 行` : ''}</p></div></div>)}
            {Boolean(draft.preview.other_findings?.length) && <details className="other-findings"><summary>其他发现：待核对原文（{draft.preview.other_findings!.length} 行）</summary><p>这些原文尚未结构化，已随报告保存，请结合原件核对。</p>{draft.preview.other_findings!.slice(0, 50).map(item => <div key={item.evidence.line_start}><small>第 {item.evidence.line_start} 行</small><pre className="review-code">{item.evidence.quote}</pre></div>)}{draft.preview.other_findings!.length > 50 && <p>这里只展开前 50 行，完整内容保存在报告原文中。</p>}</details>}
            {(draft.preview.warnings || []).map((warning, index) => <div className="warning-card" key={`${warning.code}-${index}`}><Icon name="alert" size={17}/><div><strong>需人工映射：{warning.indicator_code || warning.code}</strong><p>{warning.message}{warning.line ? ` · 第 ${warning.line} 行` : ''}</p>{warning.quote && <code>{warning.quote}</code>}</div></div>)}
            <div className="section-heading"><h3>识别的指标</h3><span>{draft.preview.indicators?.length || 0} 项</span></div>
            {(draft.preview.indicators || []).length ? <div className="indicator-table" role="table" aria-label="预览指标"><div className="table-head" role="row"><span>项目</span><span>结果</span><span>参考范围</span><span>状态</span></div>{draft.preview.indicators!.map(item => <div className="table-row" role="row" key={item.code}><span><strong>{item.display_name}</strong><small>{item.code}</small>{item.evidence?.quote && <EvidenceLocation evidence={item.evidence} imageUrl={draft.image_url} pdf={draft.input_kind === 'pdf'}/>}</span><span>{formatValue(item.value)} {item.unit}</span><span>{item.reference_range ? `${item.reference_range.lower}–${item.reference_range.upper}` : '—'}</span><span className={item.flag === 'normal' ? 'flag normal' : 'flag attention'}>{flagLabel(item.flag)}</span></div>)}</div> : <p className="muted-empty">暂无可结构化的指标，请检查原文格式。</p>}
            {(draft.preview.imaging_findings || []).length > 0 && <div className="imaging-preview"><div className="section-heading"><h3>影像尺寸</h3><span>{draft.preview.imaging_findings?.length} 处</span></div>{draft.preview.imaging_findings?.map(item => <div className="imaging-preview-row" key={item.finding_id}><div><strong>{item.body_site} {item.location} · {item.finding_type === 'fibroid' ? '肌瘤' : '结节'}</strong><span>{item.original_dimensions} {item.original_unit} · 换算为 {item.dimensions_mm.join(' × ')} mm</span></div>{item.evidence?.quote && <EvidenceLocation evidence={item.evidence} imageUrl={draft.image_url} pdf={draft.input_kind === 'pdf'}/>}</div>)}</div>}
            {(draft.preview.clinical_items || []).length > 0 && <div className="clinical-list"><div className="section-heading"><h3>报告明示临床项目</h3><span>{draft.preview.clinical_items?.length} 项</span></div>{draft.preview.clinical_items?.map((item, index) => <div className="clinical-row" key={index}><span>{({ diagnosis: '明示诊断', provider_advice: '医生建议', follow_up_plan: '复诊计划', visit_type: '就诊类型' } as Record<string,string>)[item.kind] || item.kind}</span><div><strong>{item.original_text}</strong>{item.kind === 'follow_up_plan' && <p>{item.normalized_date ? `原文日期归一化：${item.normalized_date}` : '日期待核对；保存后可在健康日历补充。'}</p>}{item.evidence && <EvidenceLocation evidence={item.evidence} imageUrl={draft.image_url} pdf={draft.input_kind === 'pdf'}/>}</div></div>)}</div>}
            <p className="review-hint">确认前请核对每个数值、单位、日期和来源原文。未知项目不会写成结构化指标。</p>
            <button type="button" className="primary-button full" disabled={busy || !canConfirm} onClick={() => void confirmReport()}><Icon name="check" size={18}/>{busy ? '保存中…' : '确认并保存到档案'}</button>
          </>}
        </section>
      </div>}

      {view === 'trends' && <section className="panel trends-panel"><div className="panel-heading"><div><span className="eyebrow">LONGITUDINAL VIEW</span><h2>按时间比较指标</h2><p>选择一个指标，查看所有已确认的同单位记录；不自动换算单位。</p></div></div>
        {activeSeries ? <div className="trend-workspace"><div className="trend-selector" aria-label="选择指标">{series.map(item => <button type="button" key={item.key} className={activeSeries.key === item.key ? 'trend-choice active' : 'trend-choice'} aria-pressed={activeSeries.key === item.key} onClick={() => setSelectedTrend(item.key)}><span>{item.name}</span><small>{item.code} · {item.unit} · {item.points.length} 次</small></button>)}</div><TrendChart series={activeSeries} onOpenReport={openReport}/></div> : <EmptyState title="还没有可比较的指标" detail="同一指标至少需要两次同单位的已确认记录。" action={() => startNew()}/>}
        {(comparisons.issues || []).length > 0 && <div className="comparison-note"><strong>未纳入比较的项目</strong><p>{comparisons.issues.length} 项缺少第二次结果或单位不一致，系统不会自动换算单位。</p></div>}
      </section>}

      {view === 'trends' && <section className="panel imaging-history"><div className="panel-heading"><div><span className="eyebrow">IMAGING HISTORY</span><h2>结节与肌瘤的历年记录</h2><p>按部位列出每次报告的原始尺寸及毫米换算值；多个病灶可能并非同一病灶，请结合影像报告核对。</p></div></div>
        {imagingGroups.length ? imagingGroups.map(group => <div className="imaging-group" key={group.key}><h3>{group.label}</h3><div className="imaging-history-list">{group.rows.map((row, index) => <div className="imaging-history-row" key={row.reportId + '-' + row.finding.finding_id + '-' + index}><time dateTime={row.date}>{row.date}</time><div><strong>{row.finding.original_dimensions} {row.finding.original_unit}</strong><small>{row.finding.dimensions_mm.join(' × ')} mm · {row.reportId}</small><details className="evidence"><summary>查看报告原文</summary><code>{row.finding.evidence?.quote || '无原文摘录'}</code></details></div></div>)}</div></div>) : <p className="muted-empty">尚无已确认的结节或肌瘤尺寸记录。</p>}
      </section>}
      {view === 'imaging' && <div className="imaging-library">
        <section className="panel imaging-library-list"><div className="panel-heading"><div><span className="eyebrow">SOURCE LIBRARY</span><h2>已确认的报告</h2><p>选择一份报告，查看原文件、医生文字和提取结果。</p></div></div>
          <div className="imaging-report-list">{[...timeline.timeline].sort((a, b) => b.report_date.localeCompare(a.report_date)).map(item => <button type="button" key={item.report_id} className={activeImagingReportId === item.report_id ? 'imaging-report-choice active' : 'imaging-report-choice'} onClick={() => openReport(item.report_id)}><time dateTime={item.report_date}>{item.report_date}</time><strong>{item.facility || '体检或影像报告'}</strong><small>{item.imaging_findings?.length ? item.imaging_findings.length + ' 条影像尺寸记录' : '查看报告内容'}</small></button>)}</div>
          {!timeline.timeline.length && <EmptyState title="尚无已确认的报告" detail="上传图片或 PDF，核对识别文字后确认保存。" action={() => startNew()}/>}
        </section>
        <section className="panel imaging-library-detail">{reportDetail ? <>
          <div className="panel-heading"><div><span className="eyebrow">REPORT DETAIL</span><h2>{reportDetail.facility || '报告详情'}</h2><p>{reportDetail.report_date} · {reportDetail.original_asset?.filename || '文字报告'}</p></div></div>
          {selectedEvidence && <div className="selected-evidence"><h3>本次引用对应的原文</h3><p>{selectedEvidence.quote}</p><EvidenceLocation evidence={selectedEvidence} imageUrl={reportDetail.original_asset ? '/api/profiles/' + encodeURIComponent(profile) + '/reports/' + encodeURIComponent(reportDetail.report_id) + '/original' : undefined} pdf={reportDetail.original_asset?.media_type === 'application/pdf'}/></div>}
          {reportDetail.original_asset ? <div className="original-asset"><div className="section-heading"><h3>原始文件</h3><span>已保存在当前档案</span></div>{reportDetail.original_asset.media_type.startsWith('image/') ? <img src={'/api/profiles/' + encodeURIComponent(profile) + '/reports/' + encodeURIComponent(reportDetail.report_id) + '/original'} alt={'原始报告图片：' + reportDetail.original_asset.filename} loading="lazy"/> : <a className="subtle-button" href={'/api/profiles/' + encodeURIComponent(profile) + '/reports/' + encodeURIComponent(reportDetail.report_id) + '/original'} target="_blank" rel="noreferrer">打开原始 PDF</a>}</div> : <p className="review-hint">这份记录没有保存原始图片或 PDF；此前仅存文字的报告仍可查看下方原文。</p>}
          <div className="section-heading"><h3>提取出的影像尺寸</h3><span>{activeImagingRecord?.imaging_findings?.length || 0} 条</span></div>
          {activeImagingRecord?.imaging_findings?.length ? <div className="imaging-detail-findings">{activeImagingRecord.imaging_findings.map(item => <div className="imaging-history-row" key={item.finding_id}><div><strong>{item.body_site} {item.finding_type === 'fibroid' ? '肌瘤' : '结节'} · {item.original_dimensions} {item.original_unit}</strong><small>换算为 {item.dimensions_mm.join(' × ')} mm</small><small>原文：{item.evidence?.quote}</small></div></div>)}</div> : <p className="muted-empty">该报告没有通过校验的结节或肌瘤尺寸，可查看下方医生报告文字。</p>}
          <div className="section-heading"><h3>报告文字（含影像描述）</h3><span>识别后经用户确认的版本</span></div><div className="doctor-report-text">{reportDetail.report_text}</div>
          {reportDetail.original_text && reportDetail.original_text !== reportDetail.report_text && <details className="evidence"><summary>最初上传或识别的原文</summary><pre className="review-code">{reportDetail.original_text}</pre></details>}
        </> : <EmptyState title="选择一份报告查看" detail="报告原图和医生文字会在这里显示。"/>}</section>
      </div>}
      {view === 'agent' && <div className="agent-layout">
        <section className="panel agent-sidebar"><div className="panel-heading"><div><span className="eyebrow">YOUR CONVERSATIONS</span><h2>对话记录</h2><p>选择一段旧对话继续提问，或开始新对话。</p></div></div>
          <button type="button" className="primary-button new-conversation" disabled={busy} onClick={() => { setAgentTaskId(newTaskId()); setAgentTask(null); setAgentPrompt('') }}><Icon name="plus" size={18}/>新对话</button>
          <div className="section-heading"><h3>最近对话</h3><span>{agentTaskList.length} 段</span></div>
          <div className="agent-task-list">{agentTaskList.map(item => <button type="button" key={item.task_id} className={agentTaskId === item.task_id ? 'agent-task-choice active' : 'agent-task-choice'} disabled={busy} onClick={() => void loadAgentTask(item.task_id)}><strong>{item.title}</strong><small>{item.updated_at.slice(0, 16).replace('T', ' ')} · {item.message_count} 条消息{item.fold_count ? ' · 已整理上下文' : ''}</small></button>)}</div>
          <details className="agent-technical"><summary>对话如何保存</summary><p>对话、摘要与任务进度保存在本机，并按当前档案隔离。医疗数据在回答时从已确认报告读取。</p><p>模型：{agentCapabilities?.status === 'ready' ? '已配置' : '待配置'} · 工具在网页服务内运行</p></details>
        </section>
        <section className="panel agent-conversation"><div className="panel-heading"><div><span className="eyebrow">CONTINUE CONVERSATION</span><h2>{agentTask?.title || '新对话'}</h2><p>继续问体检变化、影像记录或下次就诊准备。</p></div>{agentTask && <button type="button" className="subtle-button" disabled={busy || agentTask.dialogue.length < 4} onClick={() => void compactAgentTask()}>整理上下文</button>}</div>
          <div className="agent-session-meta"><span>{agentTask?.version ? '已保存，可随时继续' : '发送第一条消息后保存'}</span>{Boolean(agentTask?.fold_count) && <span>长对话已整理 {agentTask?.fold_count} 次</span>}</div>
          <div className="agent-messages" aria-live="polite">{agentTask?.dialogue.length ? agentTask.dialogue.map((item, index) => <div className={item.role === 'user' ? 'agent-message user' : 'agent-message assistant'} key={index}><strong>{item.role === 'user' ? '你' : 'HealthVault Agent'}</strong><p>{item.text}</p></div>) : <EmptyState title="开始一段对话" detail="例如：对比近两年体检变化，并列出下次就诊想问的问题。"/>}</div>
          <label className="review-check model-consent"><input type="checkbox" checked={answerConsent} onChange={event => setAnswerConsent(event.target.checked)}/>我同意将本次提问、所选对话的历史上下文及只读工具取得的必要证据发送至已配置模型服务，用于对话、个人工作流学习和自动验证。</label>
          <label className="review-check model-consent"><input type="checkbox" checked={allowWebSearch} onChange={event => setAllowWebSearch(event.target.checked)}/>允许联网查询官方网页（默认关闭；本地知识不足时仅发送通用主题词）。</label>
          <form onSubmit={event => { event.preventDefault(); void sendAgentMessage() }}><label className="question-label" htmlFor="agent-prompt">输入你的问题</label><textarea id="agent-prompt" value={agentPrompt} onChange={event => { agentInputRevision.current += 1; setAgentPrompt(event.target.value) }} maxLength={2000} rows={4} placeholder="例如：继续上次的分析，结节尺寸有哪些变化？复诊时该问什么？"/><div className="question-actions"><small>{agentPrompt.length}/2000</small><button type="submit" className="primary-button" disabled={busy || agentPrompt.trim().length < 3 || !agentTaskId.trim()}><Icon name="spark" size={18}/>{busy ? '处理中…' : '发送'}</button></div></form>
          <p className="qa-privacy">会话历史保存在本机；继续对话或折叠时，相关历史上下文也会发送给已配置模型服务处理。</p>
          {agentTask && <details className="workflow-feedback"><summary>纠正本次任务的工作流程</summary><p>可提交工具选择、执行顺序或停止条件的纠正。请只描述通用流程，不填写报告内容、个人偏好或医疗结论。</p><label className="field-label">流程纠正<textarea value={workflowCorrection} onChange={event => setWorkflowCorrection(event.target.value)} rows={3} maxLength={500}/></label><label className="review-check"><input type="checkbox" checked={workflowConfirmed} onChange={event => setWorkflowConfirmed(event.target.checked)}/>我确认这项纠正适用于相同条件下的通用工作流程。</label><label className="review-check"><input type="checkbox" checked={workflowConsent} onChange={event => setWorkflowConsent(event.target.checked)}/>我同意将流程纠正和脱敏工具轨迹发送至模型服务，用于生成和验证我的个人工作流候选。</label><button type="button" className="subtle-button" disabled={busy || workflowCorrection.trim().length < 3 || !workflowConfirmed || !workflowConsent} onClick={() => void submitWorkflowCorrection()}>提交纠正</button></details>}
        </section>
      </div>}
      {view === 'knowledge' && <QuestionPage kind="knowledge" question={knowledgeQuestion} setQuestion={setKnowledgeQuestion} onSubmit={askKnowledge} answer={knowledgeAnswer} history={answerHistory.filter(item => item.mode === 'knowledge')} onSelectHistory={setKnowledgeAnswer} busy={busy} modelConsent={answerConsent} setModelConsent={setAnswerConsent} allowWebSearch={allowWebSearch} setAllowWebSearch={setAllowWebSearch} onOpenReport={openReport}/>}
      {view === 'visit' && <QuestionPage kind="visit" question={visitQuestion} setQuestion={setVisitQuestion} onSubmit={askVisit} answer={visitAnswer} history={answerHistory.filter(item => item.mode === 'visit')} onSelectHistory={setVisitAnswer} busy={busy} reportCount={timeline.report_count} modelConsent={answerConsent} setModelConsent={setAnswerConsent} onOpenReport={openReport} brief={visitBrief}/>}
      <footer className="app-footer">HealthVault · 所有健康结论以原始报告和医生意见为准</footer>
    </main>
  </div>
}

function TrendChart({ series, onOpenReport }: { series: TrendSeries; onOpenReport: (reportId: string, evidence?: Evidence) => void }) {
  const points = series.points.slice(-200)
  const values = points.map(item => item.value)
  const low = Math.min(...values); const high = Math.max(...values)
  const padding = Math.max((high - low) * 0.15, Math.abs(high) * 0.02, 0.1)
  const yMin = low - padding; const yMax = high + padding
  const x = (index: number) => 62 + index * (654 / Math.max(points.length - 1, 1))
  const y = (value: number) => 222 - ((value - yMin) / (yMax - yMin)) * 188
  const line = points.map((item, index) => `${x(index)},${y(item.value)}`).join(' ')
  const change = points.at(-1)!.value - points[0].value
  return <div className="chart-panel"><div className="chart-heading"><div><span className="eyebrow">TREND CHART</span><h3>{series.name}</h3><p>{series.code} · {series.unit} · {series.points.length} 次记录</p></div><span className={change === 0 ? 'delta neutral' : 'delta'}>{change > 0 ? '+' : ''}{formatValue(Number(change.toFixed(2)))} {series.unit}</span></div>
    <div className="chart-wrap"><svg className="chart-desktop" viewBox="0 0 760 270" role="img" aria-label={`${series.name}趋势折线图，共${points.length}个记录点`}>
      {[0, 1, 2, 3].map(index => { const value = yMax - index * (yMax - yMin) / 3; const yy = y(value); return <g key={index}><line x1="62" x2="716" y1={yy} y2={yy} stroke="#e3ece7"/><text x="52" y={yy + 4} textAnchor="end" className="axis-text">{Number(value.toFixed(2))}</text></g> })}
      <polyline points={line} fill="none" stroke="#217a57" strokeWidth="3" strokeLinejoin="round" strokeLinecap="round"/>
      {points.map((item, index) => <circle key={`${item.reportId}-${index}`} cx={x(index)} cy={y(item.value)} r="5.5" fill="#fff" stroke="#217a57" strokeWidth="3" tabIndex={0}><title>{`${item.date} · ${item.value} ${series.unit} · ${item.reportId}`}</title></circle>)}
      <text x="62" y="258" className="axis-text">{points[0].date}</text><text x="716" y="258" textAnchor="end" className="axis-text">{points.at(-1)!.date}</text>
    </svg><svg className="chart-mobile" viewBox="0 0 320 220" role="img" aria-label={`${series.name}趋势折线图，共${points.length}个记录点`}>
      {[0, 1, 2].map(index => { const yy = 30 + index * 75; return <line key={index} x1="24" x2="304" y1={yy} y2={yy} stroke="#e3ece7"/> })}
      <polyline points={points.map((item, index) => `${24 + index * (280 / Math.max(points.length - 1, 1))},${180 - ((item.value - yMin) / (yMax - yMin)) * 150}`).join(' ')} fill="none" stroke="#217a57" strokeWidth="2.5" strokeLinejoin="round" strokeLinecap="round"/>
      {points.map((item, index) => <circle key={`${item.reportId}-${index}`} cx={24 + index * (280 / Math.max(points.length - 1, 1))} cy={180 - ((item.value - yMin) / (yMax - yMin)) * 150} r="4" fill="#fff" stroke="#217a57" strokeWidth="2.5" tabIndex={0}><title>{`${item.date} · ${item.value} ${series.unit} · ${item.reportId}`}</title></circle>)}
      <text x="24" y="211" className="axis-text">{points[0].date}</text><text x="304" y="211" textAnchor="end" className="axis-text">{points.at(-1)!.date}</text>
    </svg></div>
    {series.points.length > 200 && <p className="review-hint">折线图显示最近 200 次记录；下表保留全部记录。</p>}
    <div className="trend-table-wrap"><table className="trend-table"><caption>完整记录明细</caption><thead><tr><th scope="col">报告日期</th><th scope="col">结果</th><th scope="col">来源报告</th></tr></thead><tbody>{series.points.map((item, index) => <tr key={`${item.reportId}-${index}`}><td>{item.date}</td><td><strong>{formatValue(item.value)} {series.unit}</strong></td><td><button type="button" className="text-button" onClick={() => onOpenReport(item.reportId, item.evidence)}>查看原报告 · {item.reportId}</button></td></tr>)}</tbody></table></div>
  </div>
}

function AnswerSourceCard({ source, onOpenReport }: { source: AnswerSource; onOpenReport: (id: string, evidence?: Evidence) => void }) {
  const kindLabel = source.kind === 'record' ? '已确认记录' : source.kind === 'official_web' ? '官方网页' : '本地知识库'
  const retrieved = source.retrieved_on ? new Date(`${source.retrieved_on}T00:00:00`).toLocaleDateString('zh-CN') : null
  return <article className="answer-source">
    <span>{kindLabel} · {source.citation_id}</span>
    <h4>{source.title}</h4>
    {source.kind === 'official_web' && <p className="source-meta">{source.publisher || 'MedlinePlus'}{retrieved ? ` · 检索于 ${retrieved}` : ''}{source.updated_on ? ` · 更新于 ${source.updated_on}` : ' · 页面未提供更新日期'}</p>}
    <p>{source.detail}</p>
    {source.report_id && <button type="button" className="text-button" onClick={() => onOpenReport(source.report_id!, { ...source.evidence_locator, quote: source.detail })}>查看原报告与字段位置</button>}
    {source.report_links?.map(link => <button type="button" className="text-button" key={link.report_id} onClick={() => onOpenReport(link.report_id, link.evidence)}>查看 {link.title} 原报告</button>)}
    {source.url.startsWith('https://') && <a href={source.url} target="_blank" rel="noopener noreferrer">查看原始来源 ↗</a>}
  </article>
}

function QuestionPage({ kind, question, setQuestion, onSubmit, answer, history, onSelectHistory, busy, reportCount, modelConsent, setModelConsent, allowWebSearch, setAllowWebSearch, onOpenReport, brief }: {
  kind: 'knowledge' | 'visit'; question: string; setQuestion: (value: string) => void
  onSubmit: () => Promise<void>; answer: AnswerResult | null; history: AnswerHistoryItem[]
  onSelectHistory: (answer: AnswerResult) => void; busy: boolean; reportCount?: number
  modelConsent: boolean; setModelConsent: (value: boolean) => void
  allowWebSearch?: boolean; setAllowWebSearch?: (value: boolean) => void
  onOpenReport: (id: string, evidence?: Evidence) => void; brief?: VisitBrief | null
}) {
  const isVisit = kind === 'visit'
  const [copyMessage, setCopyMessage] = useState('')
  const [manualCopy, setManualCopy] = useState('')
  async function copyVisit() {
    if (!brief?.brief_markdown) return
    const text = [brief.brief_markdown, ...(answer?.status === 'answered' ? ['本次提问：' + answer.question, answer.answer,
      '回答来源：', ...answer.sources.map(source => `${source.title} [${source.citation_id}]\n${source.detail}${source.url ? '\n' + source.url : ''}`)] : [])].join('\n\n')
    try { await navigator.clipboard.writeText(text); setCopyMessage('已复制就诊摘要、报告日期、关键数值、待问问题和来源。'); setManualCopy('') }
    catch { setCopyMessage('自动复制未完成，可以选中下方文字复制。'); setManualCopy(text) }
  }
  return <div className="qa-layout"><section className="panel qa-composer"><div className="panel-heading"><div><span className="eyebrow">{isVisit ? 'VISIT PREPARATION' : 'CURATED KNOWLEDGE'}</span><h2>{isVisit ? '想为就诊准备什么？' : '你想了解什么？'}</h2><p>{isVisit ? '根据当前档案已确认的报告，整理可与医生讨论的信息。' : '先检索本地知识；仅在你允许且证据不足时查询获批的官方健康主题。'}</p></div></div>
    {isVisit && <div className="context-note"><Icon name="file" size={18}/><span>当前档案有 <strong>{reportCount ?? 0}</strong> 份已确认报告</span></div>}
    {isVisit && brief && brief.report_count > 0 && <div className="visit-copy"><button type="button" className="subtle-button" onClick={() => void copyVisit()}>一键复制就诊准备</button><details><summary>查看已有就诊摘要</summary><pre className="review-code">{brief.brief_markdown}</pre></details>{copyMessage && <p role="status">{copyMessage}</p>}{manualCopy && <label className="field-label">可复制的就诊准备<textarea readOnly value={manualCopy} rows={10} onFocus={event => event.target.select()}/></label>}</div>}
    <label className="review-check model-consent"><input type="checkbox" checked={modelConsent} onChange={event => setModelConsent(event.target.checked)}/>{isVisit ? '我同意将本次问题及用于回答的已确认记录摘录发送至已配置模型服务，用于回答与证据复核。' : '我同意将本次问题及检索到的知识摘录发送至已配置模型服务，用于回答与证据复核。'}</label>
    {!isVisit && <label className="review-check model-consent"><input type="checkbox" checked={allowWebSearch ?? false} onChange={event => setAllowWebSearch?.(event.target.checked)}/>允许联网查询官方网页（默认关闭；本地知识不足时仅发送通用主题词）。</label>}
    <form onSubmit={event => { event.preventDefault(); void onSubmit() }}><label className="question-label" htmlFor={`${kind}-question`}>你的问题</label><textarea id={`${kind}-question`} value={question} onChange={event => setQuestion(event.target.value)} maxLength={1000} rows={5} placeholder={isVisit ? '例如：最近的血糖结果有什么变化？就诊时我应该问医生什么？' : '例如：糖化血红蛋白反映什么？它和空腹血糖有什么区别？'}/><div className="question-actions"><small>{question.length}/1000</small><button type="submit" className="primary-button" disabled={busy || question.trim().length < 3}><Icon name="spark" size={18}/>{busy ? '正在整理回答…' : '获取回答'}</button></div></form>
    <p className="qa-privacy">{isVisit ? '提问时，问题及当前档案中用于回答的已确认记录摘要会发送到已配置模型服务。' : '问题与所选证据会发送到已配置模型服务；选择联网后，必要时从已批准的官方站点查询通用主题资料。'}</p>
  </section><section className="panel qa-answer"><div className="panel-heading"><div><span className="eyebrow">ANSWER & EVIDENCE</span><h2>回答与依据</h2></div></div>
    {answer ? <div aria-live="polite"><span className={answer.status === 'answered' ? 'answer-status' : 'answer-status muted'}>{answer.status === 'answered' ? '已生成回答' : '资料不足'}</span><h3 className="answer-question">{answer.question}</h3><p className="answer-text">{answer.answer.split(/(\[(?:REC|HVK|HVW):[^\]]+\])/g).map((part, index) => { const source = answer.sources.find(item => '[' + item.citation_id + ']' === part); return source?.report_id ? <button type="button" className="citation-link" key={index} onClick={() => onOpenReport(source.report_id!, { ...source.evidence_locator, quote: source.detail })}>{part}</button> : part })}</p>{answer.sources.length > 0 && <details className="answer-sources"><summary>查看回答依据（{answer.sources.length}）</summary><div className="source-card-list">{answer.sources.map(source => <AnswerSourceCard source={source} key={source.citation_id} onOpenReport={onOpenReport}/>)}</div></details>}</div> : <div className="answer-placeholder"><div className="answer-placeholder-icon"><Icon name="spark" size={26}/></div><h3>回答会显示在这里</h3><p>{isVisit ? '输入就诊相关问题，回答会附上当前档案中的原始记录依据。' : '输入健康知识问题，回答会附上可查看的知识来源。'}</p></div>}
    {history.length > 0 && <details className="answer-sources saved-answers"><summary>历史回答与引用（{history.length}）</summary><div className="saved-answer-list">{history.map((item, index) => <button type="button" key={`${item.created_at}-${index}`} onClick={() => onSelectHistory(item)}><time>{item.created_at}</time><span>{item.question}</span><small>{item.sources.length} 条来源</small></button>)}</div></details>}
  </section></div>
}

function EmptyState({ title, detail, action }: { title: string; detail: string; action?: () => void }) {
  return <div className="empty-state"><div className="empty-icon"><Icon name="file" size={27}/></div><h3>{title}</h3><p>{detail}</p>{action && <button type="button" className="text-button" onClick={action}>添加报告 <Icon name="arrow" size={16}/></button>}</div>
}
