import { useEffect, useState, type FormEvent } from 'react'
import './calendar.css'

type CalendarEvent = {
  event_id: string; title: string; scheduled_date: string; status: string; source: string
  remind_before_days: number; reminder_status: string; source_report_id?: string; source_report_date?: string
  source_original_text?: string; source_normalized_date?: string; source_evidence?: { quote?: string; line_start?: number }
}
type UnresolvedPlan = { clinical_item_id: string; original_text: string; report_id: string; report_date: string; evidence_quote: string; evidence_line_start: number }
type CalendarView = { pending_confirmation: CalendarEvent[]; upcoming_due: CalendarEvent[]; ended: CalendarEvent[]; unresolved_plans: UnresolvedPlan[] }
type Request = <T>(url: string, init?: RequestInit) => Promise<T>
const body = (value: unknown) => ({ headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(value) })
function today() { const now = new Date(); return `${now.getFullYear()}-${String(now.getMonth()+1).padStart(2,'0')}-${String(now.getDate()).padStart(2,'0')}` }
const reminderLabels: Record<string,string> = { due: '今日需留意', overdue: '已逾期', acknowledged: '已知悉', not_due: '尚未到提醒日', inactive: '提醒未激活' }

function DateForm({ date = '', days = 1, title, busy, onSubmit }: { date?: string; days?: number; title: string; busy: boolean; onSubmit: (date: string, days: number) => void }) {
  const [chosenDate, setDate] = useState(date)
  const [reminder, setReminder] = useState(days)
  return <form className="calendar-date-form" onSubmit={event => { event.preventDefault(); onSubmit(chosenDate, reminder) }}>
    <label className="field-label">安排日期<input type="date" required value={chosenDate} onChange={event => setDate(event.target.value)} disabled={busy}/></label>
    <label className="field-label">提前提醒（天）<input type="number" min="0" max="365" required value={reminder} onChange={event => setReminder(Number(event.target.value))} disabled={busy}/></label>
    <button className="primary-button" disabled={busy || !chosenDate} type="submit">{title}</button>
  </form>
}

export default function CalendarPanel({ profile, request, onOpenReport }: { profile: string; request: Request; onOpenReport: (id: string) => void }) {
  const [view, setView] = useState<CalendarView | null>(null)
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState('')
  const [message, setMessage] = useState('')
  const [refreshFailed, setRefreshFailed] = useState(false)
  const [title, setTitle] = useState('')
  const [manualDate, setManualDate] = useState(today)
  const [manualDays, setManualDays] = useState(1)
  const [requestKey, setRequestKey] = useState(() => crypto.randomUUID())
  const base = `/api/profiles/${encodeURIComponent(profile)}/calendar`
  const load = () => request<CalendarView>(`${base}?as_of=${today()}`)
  useEffect(() => {
    let active = true
    load().then(result => { if (active) setView(result) }).catch(reason => { if (active) setError((reason as Error).message) })
    return () => { active = false }
  }, [profile, request])
  async function refreshCalendar() {
    setBusy(true); setError('')
    try { setView(await load()); setRefreshFailed(false) }
    catch (reason) { if (!refreshFailed) setError((reason as Error).message) }
    finally { setBusy(false) }
  }
  async function mutate(path: string, value?: unknown, method = 'POST') {
    setBusy(true); setError(''); setMessage('')
    try {
      await request(base + path, { method, ...(value === undefined ? {} : body(value)) })
      setMessage('日历已保存。')
      try { setView(await load()); setRefreshFailed(false) }
      catch { setRefreshFailed(true) }
      return true
    } catch (reason) { setError((reason as Error).message); return false }
    finally { setBusy(false) }
  }
  async function create(event: FormEvent) {
    event.preventDefault()
    if (await mutate('/events', { idempotency_key: requestKey, title: title.trim(), scheduled_date: manualDate, remind_before_days: manualDays })) {
      setTitle(''); setRequestKey(crypto.randomUUID())
    }
  }
  function source(event: CalendarEvent) {
    return event.source_report_id ? <div className="calendar-source"><p>报告日期 {event.source_report_date} · 第 {event.source_evidence?.line_start || '—'} 行</p><blockquote>{event.source_evidence?.quote || event.source_original_text}</blockquote><button className="subtle-button" type="button" onClick={() => onOpenReport(event.source_report_id!)}>查看来源报告</button></div> : <p className="calendar-source">用户自行安排</p>
  }
  return <div className="calendar-layout">
    <section className="panel calendar-intro"><div className="panel-heading"><div><span className="eyebrow">HEALTH CALENDAR</span><h2>健康日历</h2><p>报告中的复诊计划核对后才进入安排；原文日期与日历日期分别保留。提醒在此页面显示。</p></div><button type="button" className="subtle-button" disabled={busy} onClick={() => void refreshCalendar()}>刷新</button></div>
      {error && <div className="notice error" role="alert">{error}</div>}{message && <div className="notice success" role="status">{message}</div>}
      {refreshFailed && <div className="notice refresh-warning" role="status">日历列表刷新失败，尚未更新，请刷新查看最新安排。<button type="button" className="subtle-button" disabled={busy} onClick={() => void refreshCalendar()}>仅刷新日历</button></div>}
      {!view && !error && <p role="status">正在载入日历…</p>}
    </section>
    {view && <>
      <section className="panel calendar-section"><h2>待核对的复诊计划 <span className="calendar-count">{view.pending_confirmation.length + view.unresolved_plans.length}</span></h2>
        {!view.pending_confirmation.length && !view.unresolved_plans.length && <p className="calendar-empty">暂无待核对计划。确认报告中的明示复诊建议后，会在这里显示。</p>}
        {view.pending_confirmation.map(event => <article className="calendar-card" key={event.event_id}><div className="calendar-card-heading"><h3>{event.source_original_text || event.title}</h3><span className="calendar-badge">待确认</span></div>{source(event)}<p>原文归一化日期：{event.source_normalized_date}</p><DateForm date={event.scheduled_date} days={event.remind_before_days} title="确认安排" busy={busy} onSubmit={(date, days) => void mutate(`/events/${event.event_id}/activate`, { scheduled_date: date, remind_before_days: days, acknowledged: true })}/><button className="subtle-button" type="button" disabled={busy} onClick={() => void mutate(`/events/${event.event_id}/cancel`)}>不加入安排</button></article>)}
        {view.unresolved_plans.map(plan => <article className="calendar-card" key={plan.clinical_item_id}><div className="calendar-card-heading"><h3>{plan.original_text}</h3><span className="calendar-badge">待补日期</span></div><div className="calendar-source"><p>报告日期 {plan.report_date} · 第 {plan.evidence_line_start} 行</p><blockquote>{plan.evidence_quote}</blockquote><button className="subtle-button" type="button" onClick={() => onOpenReport(plan.report_id)}>查看来源报告</button></div><p>原文未提供可唯一确定的未来日期，请核对后选择。</p><DateForm title="确认日期并安排" busy={busy} onSubmit={(date, days) => void mutate(`/plans/${plan.clinical_item_id}/schedule`, { scheduled_date: date, remind_before_days: days, acknowledged: true })}/></article>)}
      </section>
      <section className="panel calendar-section"><h2>已安排 <span className="calendar-count">{view.upcoming_due.length}</span></h2>{!view.upcoming_due.length && <p className="calendar-empty">暂无日历安排。</p>}{view.upcoming_due.map(event => <article className="calendar-card" key={event.event_id}><div className="calendar-card-heading"><div><time dateTime={event.scheduled_date} className="calendar-day">{event.scheduled_date}</time><h3>{event.title}</h3></div><span className="calendar-badge" data-state={event.reminder_status}>{reminderLabels[event.reminder_status] || event.reminder_status}</span></div>{source(event)}<div className="review-actions"><button className="primary-button" type="button" disabled={busy} onClick={() => void mutate(`/events/${event.event_id}/complete`)}>标记完成</button>{['due','overdue'].includes(event.reminder_status) && <button className="subtle-button" type="button" disabled={busy} onClick={() => void mutate(`/events/${event.event_id}/acknowledge`)}>已知悉提醒</button>}<button className="subtle-button" type="button" disabled={busy} onClick={() => void mutate(`/events/${event.event_id}/cancel`)}>取消安排</button></div><details><summary>调整日期或提醒</summary><DateForm date={event.scheduled_date} days={event.remind_before_days} title="保存修改" busy={busy} onSubmit={(date, days) => void mutate(`/events/${event.event_id}`, { scheduled_date: date, remind_before_days: days }, 'PATCH')}/></details></article>)}</section>
      <section className="panel calendar-section"><details><summary>已结束与既往就诊（{view.ended.length}）</summary>{view.ended.map(event => <article className="calendar-card" key={event.event_id}><div className="calendar-card-heading"><h3>{event.title}</h3><span>{event.status === 'cancelled' ? '已取消' : '已完成'} · {event.scheduled_date}</span></div>{source(event)}</article>)}</details></section>
    </>}
    <section className="panel calendar-section"><h2>新增个人安排</h2><form className="calendar-manual-form" onSubmit={event => void create(event)}><label className="field-label">安排名称<input required maxLength={200} placeholder="如：复诊、检查或健康活动" value={title} onChange={event => setTitle(event.target.value)} disabled={busy}/></label><label className="field-label">安排日期<input type="date" required value={manualDate} onChange={event => setManualDate(event.target.value)} disabled={busy}/></label><label className="field-label">提前提醒（天）<input type="number" required min="0" max="365" value={manualDays} onChange={event => setManualDays(Number(event.target.value))} disabled={busy}/></label><button type="submit" className="primary-button" disabled={busy || !title.trim()}>保存安排</button></form></section>
  </div>
}
