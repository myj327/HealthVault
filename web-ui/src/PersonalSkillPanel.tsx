import { useEffect, useState } from 'react'

type Candidate = { candidate_id: string; status: string; unified_diff: string; source_trace_ids: string[]; applicability_conditions: string[]; evaluation: unknown; risk: unknown; explanation_zh?: string }
type Request = <T>(url: string, init?: RequestInit) => Promise<T>
type Usage = { active_sha256: string | null; has_personal_update: boolean; injectable: boolean; window_size: number; evaluations: Array<{ skill_sha256: string; task_type: string; decision: string; effective_samples: number; success_count: number; validator_failures: number; safety_hard_failures: number }>; last_action?: { status: string; actions: string[] } | null; explanation_zh?: string | null }
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
    <div className="panel-heading"><div><span className="eyebrow">只对你的账户生效</span><h2>让助手记住更合适的做事方法</h2><p>助手可以学习如何选取依据、整理回答。只有通过检查的经验才会使用；不会把你的检查数值或医疗结论学成规则。</p></div></div>
    <label className="review-check"><input type="checkbox" checked={manual} disabled={busy || !ready} onChange={event => void action('/api/skills/settings', { manual_review: event.target.checked }, 'PUT')}/>每次更新先让我确认</label>
    <p className="learning-help">未勾选时，通过自动检查的更新会对你生效。发现不合适，可以停用或撤销。</p>
    {error && <p role="alert" className="notice error">{error}</p>}{message && <p role="status" className="notice success">{message}</p>}
    {refreshFailed && <div role="status" className="notice refresh-warning">学习记录暂未加载，请重试。<button type="button" className="subtle-button" disabled={busy} onClick={() => void retryRefresh()}>重新加载</button></div>}
    {usage && <div className="personal-usage"><h3>{!usage.has_personal_update ? '尚无生效的个人经验' : usage.injectable ? '个人经验正在使用' : '个人经验已停用'}</h3><p>{usage.has_personal_update && usage.injectable ? '只在问题符合适用条件时使用，其他问题沿用默认方法。' : '健康助手仍可正常使用。积累足够依据后，新的学习记录会出现在下方。'}</p><div className="review-actions"><button type="button" className="subtle-button" disabled={busy || !usage.has_personal_update || !usage.injectable} onClick={() => void action('/api/skills/disable')}>停用个人经验</button><button type="button" className="subtle-button" disabled={busy || !usage.has_personal_update} onClick={() => void action('/api/skills/rollback')}>撤销上次更新</button></div>{usage.last_action && <p>最近处理：{usage.last_action.actions.map(action => ({ keep: '保留', watch: '观察', disable: '停用', rollback: '撤销更新', archive: '归档' } as Record<string,string>)[action] || action).join('、')}</p>}
      {usage.explanation_zh && <p>{usage.explanation_zh}</p>}
      <details><summary>查看使用记录与技术详情</summary><p>版本 {usage.active_sha256?.slice(0,12) || '无'}。最近 {usage.window_size} 次实际使用会检查执行和引用是否符合要求；这不是医疗回答正确率，也不能单独证明学习收益。</p>{!usage.evaluations.length ? <p className="muted-empty">实际使用后会显示观察结果。</p> : usage.evaluations.map(item => <p key={item.skill_sha256 + item.task_type}>{item.skill_sha256.slice(0,12)} · {item.task_type} · {({ keep: '保留', watch: '观察', disable: '停用' } as Record<string,string>)[item.decision]} · 独立任务 {item.effective_samples}，检查通过 {item.success_count}，未通过 {item.validator_failures}</p>)}</details>
    </div>}
    <h3>学习记录</h3>
    {!ready ? <p role="status">正在读取个人经验…</p> : !items.length ? <p className="muted-empty">暂时没有更新。你可以在对话结束后告诉助手，怎样整理回答更合适。</p> : items.map(item => <details key={item.candidate_id}><summary>{labels[item.status] || item.status} · 查看这次更新</summary><p>{item.explanation_zh}</p><p><strong>适用情况：</strong>{item.applicability_conditions.join('；')}</p><details><summary>修改内容与验证来源</summary><pre className="review-code">{item.unified_diff}</pre><pre className="review-code">{JSON.stringify({ candidate_id: item.candidate_id, evaluation: item.evaluation, risk: item.risk, traces: item.source_trace_ids }, null, 2)}</pre></details>
      {['pending_review', 'watch'].includes(item.status) && <div className="review-actions">{(['approve', 'reject', 'watch'] as const).map((decision, index) => <button key={decision} type="button" disabled={busy} onClick={() => void action(`/api/skills/candidates/${item.candidate_id}/decision`, { decision })}>{['批准生效', '拒绝', '继续观察'][index]}</button>)}</div>}
      {['pending_review', 'watch', 'published'].includes(item.status) && <button type="button" className="subtle-button" disabled={busy} onClick={() => void action(`/api/skills/candidates/${item.candidate_id}/share`)}>提议分享这条经验</button>}
    </details>)}
  </section>
}
