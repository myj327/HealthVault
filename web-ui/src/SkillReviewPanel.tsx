import { useEffect, useState } from 'react'

type Candidate = {
  candidate_id: string; skill_name: string; status: string; created_at: string; unified_diff: string
  source_trace_ids: string[]; applicability_conditions: string[]; when_not_to_use: string[]
  verification: { status: string; reasons: string[] }
  evaluation: { passed?: boolean; baseline?: { passed: number; total: number }; candidate?: { passed: number; total: number }; held_out?: { status: string } }
  risk: { level: string; reasons: string[] }
  explanation_zh?: string
}
type Request = <T>(url: string, init?: RequestInit) => Promise<T>
const labels: Record<string, string> = { pending_review: '待审核', watch: '继续观察', rejected: '已拒绝', published: '已发布', stale: '基础版本已变化', gate_rejected: '检查未通过' }

export default function SkillReviewPanel({ request }: { request: Request }) {
  const [candidates, setCandidates] = useState<Candidate[]>([])
  const [selected, setSelected] = useState('')
  const [acknowledged, setAcknowledged] = useState(false)
  const [trace, setTrace] = useState<unknown>(null)
  const [busy, setBusy] = useState(false)
  const [ready, setReady] = useState(false)
  const [error, setError] = useState('')
  const [message, setMessage] = useState('')
  const candidate = candidates.find(item => item.candidate_id === selected) || candidates[0]

  async function refresh() {
    const result = await request<{ candidates: Candidate[] }>('/api/admin/skills/candidates')
    setCandidates(result.candidates)
  }

  useEffect(() => {
    let active = true
    request<{ candidates: Candidate[] }>('/api/admin/skills/candidates')
      .then(result => { if (active) setCandidates(result.candidates) })
      .catch((reason: Error) => { if (active) setError(reason.message) })
      .finally(() => { if (active) setReady(true) })
    return () => { active = false }
  }, [request])

  async function decide(decision: 'approve' | 'reject' | 'watch') {
    if (!candidate) return
    setBusy(true); setError(''); setMessage('')
    try {
      const result = await request<Candidate>(`/api/admin/skills/candidates/${candidate.candidate_id}/decision`, {
        method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ decision }),
      })
      setMessage(labels[result.status] || result.status)
      setAcknowledged(false)
      await refresh()
    } catch (reason) { setError((reason as Error).message) }
    finally { setBusy(false) }
  }

  async function rollback() {
    setBusy(true); setError(''); setMessage('')
    try {
      await request('/api/admin/skills/rollback', { method: 'POST' })
      setMessage('已恢复上一个共享 Skill 版本。'); setAcknowledged(false)
      await refresh()
    } catch (reason) { setError((reason as Error).message) }
    finally { setBusy(false) }
  }

  async function showTrace(id: string) {
    setBusy(true); setError('')
    try { setTrace(await request(`/api/admin/skills/traces/${id}`)) }
    catch (reason) { setError((reason as Error).message) }
    finally { setBusy(false) }
  }

  const reviewable = candidate && ['pending_review', 'watch'].includes(candidate.status)
  return <section className="panel skill-review" aria-busy={busy}>
    <div className="panel-heading"><div><span className="eyebrow">PROJECT WORKFLOW</span><h2>项目 Skill 发布管理</h2><p>这里管理项目级共通版本。个人学习在上方独立验证，共享提案获批后由开发者纳入项目版本。</p></div><button type="button" className="subtle-button" disabled={busy} onClick={() => void rollback()}>回滚上次发布</button></div>
    {error && <p className="notice error" role="alert">{error}</p>}
    {message && <p className="notice success" role="status">{message}</p>}
    {!ready ? <p role="status">正在读取候选…</p> : !candidates.length ? <p className="muted-empty">当前没有待发布的项目版本。</p> : <div className="skill-review-layout">
      <div className="agent-task-list" aria-label="选择候选">{candidates.map(item => <button type="button" key={item.candidate_id} className={candidate?.candidate_id === item.candidate_id ? 'agent-task-choice active' : 'agent-task-choice'} aria-pressed={candidate?.candidate_id === item.candidate_id} disabled={busy} onClick={() => { setSelected(item.candidate_id); setAcknowledged(false); setTrace(null) }}><strong>{labels[item.status] || item.status}</strong><small>{item.candidate_id} · {new Date(item.created_at).toLocaleString('zh-CN')}</small></button>)}</div>
      {candidate && <div className="skill-review-detail">
        <h3>{candidate.skill_name}</h3><p>状态：{labels[candidate.status] || candidate.status} · 风险：{{ low: '低', medium: '中', high: '高' }[candidate.risk.level] || candidate.risk.level}</p>
        <div className="context-note">静态检查：{candidate.evaluation.candidate ? `${candidate.evaluation.candidate.passed}/${candidate.evaluation.candidate.total}` : '未完成'}；基线：{candidate.evaluation.baseline ? `${candidate.evaluation.baseline.passed}/${candidate.evaluation.baseline.total}` : '未完成'}。正式 held-out 任务评测尚未运行，不能据此判断准确率提升。</div>
        <h4>修改内容</h4><p>{candidate.explanation_zh}</p><pre className="review-code">{candidate.unified_diff || '候选未通过检查，未生成可发布的修改。'}</pre>
        <h4>适用条件</h4><ul>{candidate.applicability_conditions.map(item => <li key={item}>{item}</li>)}</ul>
        <h4>不适用条件</h4><ul>{candidate.when_not_to_use.map(item => <li key={item}>{item}</li>)}</ul>
        <details><summary>检查结果与风险</summary><pre className="review-code">{JSON.stringify({ verification: candidate.verification, risk: candidate.risk }, null, 2)}</pre></details>
        <details><summary>来源轨迹（{candidate.source_trace_ids.length}）</summary><div className="review-traces">{candidate.source_trace_ids.map(id => <button className="subtle-button" type="button" key={id} disabled={busy} onClick={() => void showTrace(id)}>{id}</button>)}</div>{trace !== null && <pre className="review-code">{JSON.stringify(trace, null, 2)}</pre>}</details>
        {reviewable && <><label className="review-check"><input type="checkbox" checked={acknowledged} onChange={event => setAcknowledged(event.target.checked)}/>我已核对修改、来源和风险，确认可作为共享工作流发布。</label><div className="review-actions"><button className="primary-button" type="button" disabled={busy || !acknowledged || !candidate.evaluation.passed} onClick={() => void decide('approve')}>批准发布</button><button className="subtle-button" type="button" disabled={busy} onClick={() => void decide('watch')}>继续观察</button><button className="subtle-button" type="button" disabled={busy} onClick={() => void decide('reject')}>拒绝</button></div></>}
      </div>}
    </div>}
  </section>
}
