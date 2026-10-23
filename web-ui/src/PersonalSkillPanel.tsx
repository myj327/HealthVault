import { useEffect, useState } from 'react'

type Candidate = { candidate_id: string; status: string; unified_diff: string; source_trace_ids: string[]; applicability_conditions: string[]; evaluation: unknown; risk: unknown; explanation_zh?: string }
type Request = <T>(url: string, init?: RequestInit) => Promise<T>
type Usage = { active_sha256: string | null; injectable: boolean; window_size: number; evaluations: Array<{ skill_sha256: string; task_type: string; decision: string; effective_samples: number; success_count: number; validator_failures: number; safety_hard_failures: number }>; last_action?: { status: string; actions: string[] } | null; explanation_zh?: string | null }
const labels: Record<string, string> = { published: '已发布', pending_review: '等待验证或审核', watch: '继续观察', rejected: '已拒绝', gate_rejected: '检查未通过', stale: '版本已变化' }

export default function PersonalSkillPanel({ request }: { request: Request }) {
  const [manual, setManual] = useState(false)
  const [items, setItems] = useState<Candidate[]>([])
  const [busy, setBusy] = useState(false)
  const [ready, setReady] = useState(false)
  const [error, setError] = useState('')
  const [message, setMessage] = useState('')
  const [refreshFailed, setRefreshFailed] = useState(false)
  const [usage, setUsage] = useState<Usage | null>(null)
  async function refresh() {
    const result = await request<{ settings: { manual_review: boolean }; candidates: Candidate[]; usage: Usage }>('/api/skills')
    setManual(result.settings.manual_review); setItems(result.candidates); setUsage(result.usage); setReady(true); setRefreshFailed(false)
  }
  useEffect(() => { void refresh().catch(reason => setError(reason.message)) }, [request])
  async function retryRefresh() {
    setBusy(true); setError('')
    try { await refresh() }
    catch { /* Keep the saved result and the refresh-only action. */ }
    finally { setBusy(false) }
  }
  async function action(url: string, body?: object, method = 'POST') {
    setBusy(true); setError(''); setMessage('')
    try {
      await request(url, { method, headers: { 'Content-Type': 'application/json' }, body: body ? JSON.stringify(body) : undefined })
      if (url === '/api/skills/settings' && body && 'manual_review' in body && typeof body.manual_review === 'boolean') setManual(body.manual_review)
      setMessage(url.endsWith('/share') ? '已提交共享经验，等待管理员审核。' : '已保存。')
      try { await refresh() }
      catch { setRefreshFailed(true) }
    } catch (reason) { setError((reason as Error).message) }
    finally { setBusy(false) }
  }
  return <section className="panel skill-review" aria-busy={busy}>
    <div className="panel-heading"><div><span className="eyebrow">PERSONAL WORKFLOW</span><h2>我的工作流经验</h2><p>系统从有验证依据的任务中学习。通过检查的新版本默认只对你生效，可随时回滚。</p></div><button type="button" className="subtle-button" disabled={busy} onClick={() => void action('/api/skills/rollback')}>回滚我的上次更新</button></div>
    <label className="review-check"><input type="checkbox" checked={manual} disabled={busy || !ready} onChange={event => void action('/api/skills/settings', { manual_review: event.target.checked }, 'PUT')}/>生效前由我审核（可选）</label>
    {error && <p role="alert" className="notice error">{error}</p>}{message && <p role="status" className="notice success">{message}</p>}
    {refreshFailed && <div role="status" className="notice refresh-warning">工作流列表刷新失败，尚未更新，请刷新查看最新状态。<button type="button" className="subtle-button" disabled={busy} onClick={() => void retryRefresh()}>仅刷新工作流</button></div>}
    {usage && <div className="personal-usage"><h3>发布后的使用观察</h3><p>{usage.injectable ? '当前个人版本可用' : '个人版本已下线，对话继续使用项目共通工作流'} · 版本 {usage.active_sha256?.slice(0,12) || '—'}</p><p>按独立任务统计最近 {usage.window_size} 次实际注入，检查工具、结果状态与引用契约。此处统计不代表医疗回答准确率，也不能单独证明 Skill 带来的提升。</p><button type="button" className="subtle-button" disabled={busy || !usage.injectable} onClick={() => void action('/api/skills/disable')}>停用当前个人版本</button>{usage.last_action && <p>最近处理：{usage.last_action.actions.map(action => ({ keep: '保留', watch: '观察', disable: '下线', rollback: '回滚', archive: '归档' } as Record<string,string>)[action] || action).join('、')}</p>}
      {usage.explanation_zh && <p>{usage.explanation_zh}</p>}
      {!usage.evaluations.length ? <p className="muted-empty">发布并实际使用个人更新后，会显示观察结果。</p> : usage.evaluations.map(item => <p key={item.skill_sha256 + item.task_type}>{item.skill_sha256.slice(0,12)} · {item.task_type} · {({ keep: '保留', watch: '观察', disable: '下线' } as Record<string,string>)[item.decision]} · 有效独立任务 {item.effective_samples}，契约通过 {item.success_count}，未通过 {item.validator_failures}</p>)}
    </div>}
    {!ready ? <p role="status">正在读取个人经验…</p> : !items.length ? <p className="muted-empty">有足够验证依据后，学习记录会显示在这里。</p> : items.map(item => <details key={item.candidate_id}><summary>{labels[item.status] || item.status} · {item.candidate_id}</summary><p>{item.explanation_zh}</p><pre className="review-code">{item.unified_diff}</pre><p>{item.applicability_conditions.join('；')}</p><details><summary>验证与来源</summary><pre className="review-code">{JSON.stringify({ evaluation: item.evaluation, risk: item.risk, traces: item.source_trace_ids }, null, 2)}</pre></details>
      {['pending_review', 'watch'].includes(item.status) && <div className="review-actions">{(['approve', 'reject', 'watch'] as const).map((decision, index) => <button key={decision} type="button" disabled={busy} onClick={() => void action(`/api/skills/candidates/${item.candidate_id}/decision`, { decision })}>{['批准生效', '拒绝', '继续观察'][index]}</button>)}</div>}
      {['pending_review', 'watch', 'published'].includes(item.status) && <button type="button" className="subtle-button" disabled={busy} onClick={() => void action(`/api/skills/candidates/${item.candidate_id}/share`)}>提议作为共享工作流经验</button>}
    </details>)}
  </section>
}
