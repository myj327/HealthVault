import { useEffect, useState } from 'react'
type Item = { candidate_id: string; proposal_id?: string; owner_ref: string; status: string; unified_diff: string; applicability_conditions: string[]; source_trace_ids: string[]; evaluation: unknown; risk: unknown }
type Request = <T>(url: string, init?: RequestInit) => Promise<T>
export default function SharedExperiencePanel({ request }: { request: Request }) {
  const [items, setItems] = useState<Item[]>([])
  const [proposals, setProposals] = useState<Item[]>([])
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState('')
  const [trace, setTrace] = useState<unknown>(null)
  async function refresh() {
    const result = await request<{ personal_candidates: Item[]; shared_proposals: Item[] }>('/api/admin/skills/candidates')
    setItems(result.personal_candidates); setProposals(result.shared_proposals)
  }
  useEffect(() => { void refresh().catch(reason => setError(reason.message)) }, [request])
  async function decide(item: Item, decision: string) {
    setBusy(true); setError('')
    try { await request(`/api/admin/skills/proposals/${item.proposal_id}/decision`, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ decision }) }); await refresh() }
    catch (reason) { setError((reason as Error).message) }
    finally { setBusy(false) }
  }
  async function showTrace(item: Item, id: string) {
    setBusy(true); setError('')
    try { setTrace(await request(`/api/admin/skills/personal/${item.owner_ref}/traces/${id}`)) }
    catch (reason) { setError((reason as Error).message) }
    finally { setBusy(false) }
  }
  return <section className="panel skill-review" aria-busy={busy}><div className="panel-heading"><div><h2>个人学习与共享提案</h2><p>个人经验仅影响本人。共享提案审核通过后，由开发者决定是否纳入项目 Skill。</p></div></div>{error && <p className="notice error" role="alert">{error}</p>}
    {[...proposals, ...items].map(item => <details key={`${item.proposal_id || ''}:${item.owner_ref}:${item.candidate_id}`}><summary>{item.proposal_id ? '共享提案' : '个人候选'} · {({ published: '已生效', pending_review: '待审核', approved_for_developer: '已批准，待开发者发布', rejected: '已拒绝', watch: '继续观察', gate_rejected: '检查未通过' } as Record<string, string>)[item.status] || item.status}</summary><p>{item.applicability_conditions.join('；')}</p><pre className="review-code">{item.unified_diff}</pre><details><summary>检查与风险</summary><pre className="review-code">{JSON.stringify({ evaluation: item.evaluation, risk: item.risk }, null, 2)}</pre></details><div className="review-traces">{item.source_trace_ids.map(id => <button type="button" className="subtle-button" key={id} disabled={busy} onClick={() => void showTrace(item, id)}>{id}</button>)}</div>{item.proposal_id && ['pending_review', 'watch'].includes(item.status) && <div className="review-actions">{['approve', 'reject', 'watch'].map((decision, index) => <button key={decision} type="button" disabled={busy} onClick={() => void decide(item, decision)}>{['批准共享', '拒绝', '继续观察'][index]}</button>)}</div>}</details>)}
    {!items.length && !proposals.length && <p className="muted-empty">暂无个人学习记录或共享提案。</p>}{trace !== null && <pre className="review-code">{JSON.stringify(trace, null, 2)}</pre>}
  </section>
}
