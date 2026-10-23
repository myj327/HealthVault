import { useEffect, useState } from 'react'

type Request = <T>(url: string, init?: RequestInit) => Promise<T>
type Candidate = {
  candidate_id: string; corpus_version: string; status: string; changed_source_ids: string[]; unified_diff: string; risk: string[]
  provenance: Array<{ source_id: string; publisher: string; version: string; retrieved_on: string; published_on: string | null; updated_on: string | null; document_kind: string }>
  evaluation: { passed: boolean; retrieval_probes: Array<{ query: string; passed: boolean }>; answer_held_out: string }
  retrieval_mode: string; vector_build?: { reused: number; embedded: number; total: number }
}
type SourceObservation = { observation_id: string; source_id: string; publisher: string; checked_at: string; status: string; fetch_status: string; pending_snapshot?: string }
type ReleaseList = { active: { corpus_version: string; source_count: number; retrieval_mode: string }; candidates: Candidate[]; source_observations?: SourceObservation[]; sources?: Array<{ source_id: string; title: string; publisher: string }> }
const labels: Record<string, string> = { pending_review: '待审核', watch: '继续观察', rejected: '已拒绝', published: '已发布', stale: '基础版本已变化', gate_rejected: '检索检查未通过' }

export default function KnowledgeReviewPanel({ request }: { request: Request }) {
  const [releases, setReleases] = useState<ReleaseList | null>(null)
  const [selected, setSelected] = useState('')
  const [file, setFile] = useState<File | null>(null)
  const [checked, setChecked] = useState(false)
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState('')
  const [message, setMessage] = useState('')
  const [sourcePreview, setSourcePreview] = useState<{ source_id: string; text: string; unified_diff: string } | null>(null)
  const [sourceId, setSourceId] = useState('')
  const candidate = releases?.candidates.find(item => item.candidate_id === selected) || releases?.candidates[0]

  async function refresh() { setReleases(await request<ReleaseList>('/api/admin/knowledge/candidates')) }
  useEffect(() => {
    let active = true
    request<ReleaseList>('/api/admin/knowledge/candidates').then(value => { if (active) setReleases(value) })
      .catch((reason: Error) => { if (active) setError(reason.message) })
    return () => { active = false }
  }, [request])

  async function stage() {
    if (!file) return
    setBusy(true); setError(''); setMessage('')
    try {
      if (file.size > 2_000_000) throw new Error('更新文件不能超过 2 MB。')
      const payload: unknown = JSON.parse(await file.text())
      const result = await request<Candidate>('/api/admin/knowledge/candidates', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(payload) })
      setSelected(result.candidate_id); setChecked(false)
      setMessage('知识版本已进入待审核区，请检查来源、修改和检索结果。'); await refresh()
    } catch (reason) { setError((reason as Error).message) }
    finally { setBusy(false) }
  }

  async function checkSources() {
    setBusy(true); setError(''); setMessage('')
    try {
      const result = await request<{ changed: number; errors: unknown[] }>('/api/admin/knowledge/check-sources', { method: 'POST' })
      setMessage(`巡检完成：${result.changed} 份快照需要比对，${result.errors.length} 个来源暂未成功。`); await refresh()
    } catch (reason) { setError((reason as Error).message) }
    finally { setBusy(false) }
  }

  async function buildVectors() {
    if (!candidate) return
    setBusy(true); setError(''); setMessage('')
    try {
      await request(`/api/admin/knowledge/candidates/${candidate.candidate_id}/vectors`, { method: 'POST' })
      setChecked(false); setMessage('本地向量索引已构建，请重新核对候选版本。'); await refresh()
    } catch (reason) { setError((reason as Error).message) }
    finally { setBusy(false) }
  }

  async function downloadTemplate() {
    const identifier = sourceId || releases?.sources?.[0]?.source_id
    if (!identifier) return
    setBusy(true); setError('')
    try {
      const payload = await request(`/api/admin/knowledge/update-template?source_id=${encodeURIComponent(identifier)}`)
      const url = URL.createObjectURL(new Blob([JSON.stringify(payload, null, 2)], { type: 'application/json' }))
      const anchor = document.createElement('a'); anchor.href = url; anchor.download = `${identifier}-update.json`; anchor.click(); URL.revokeObjectURL(url)
    } catch (reason) { setError((reason as Error).message) }
    finally { setBusy(false) }
  }

  async function previewSource(identifier: string) {
    setBusy(true); setError('')
    try { setSourcePreview(await request(`/api/admin/knowledge/sources/${identifier}/pending`)) }
    catch (reason) { setError((reason as Error).message) }
    finally { setBusy(false) }
  }

  async function act(decision: 'approve' | 'reject' | 'watch' | 'rollback') {
    if (!candidate && decision !== 'rollback') return
    setBusy(true); setError(''); setMessage('')
    try {
      const url = decision === 'rollback' ? '/api/admin/knowledge/rollback' : `/api/admin/knowledge/candidates/${candidate!.candidate_id}/decision`
      await request(url, { method: 'POST', headers: { 'Content-Type': 'application/json' }, ...(decision !== 'rollback' ? { body: JSON.stringify({ decision }) } : {}) })
      setMessage(decision === 'rollback' ? '已恢复上一知识版本。' : decision === 'approve' ? '版本已处理，请核对当前生效版本。' : decision === 'reject' ? '已拒绝更新。' : '已标记为继续观察。')
      setChecked(false); await refresh()
    } catch (reason) { setError((reason as Error).message) }
    finally { setBusy(false) }
  }

  return <section className="panel skill-review knowledge-review" aria-busy={busy}>
    <div className="panel-heading"><div><span className="eyebrow">VERSIONED KNOWLEDGE</span><h2>知识版本管理</h2><p>{releases ? `当前：${releases.active.corpus_version} · ${releases.active.source_count} 个来源` : '正在读取当前版本…'}</p></div><button type="button" className="subtle-button" disabled={busy} onClick={() => void act('rollback')}>回滚知识版本</button></div>
    {error && <p className="notice error" role="alert">{error}</p>}{message && <p className="notice success" role="status">{message}</p>}
    <div className="knowledge-update-form"><button type="button" className="subtle-button" disabled={busy} onClick={() => void checkSources()}>检查官方来源更新</button><p className="review-hint">服务器每天检查一次；变化进入待比对快照。核对后上传局部来源更新文件，再审核发布。</p>{releases?.source_observations?.map(item => <article className="knowledge-source-meta" key={item.observation_id}><strong>{item.publisher} · {item.source_id}</strong><p>检查：{item.checked_at} · {item.status === 'needs_review' ? '待比对' : item.status}</p>{item.pending_snapshot && <button type="button" className="subtle-button" disabled={busy} onClick={() => void previewSource(item.observation_id)}>查看官方快照与差异</button>}</article>)}{sourcePreview && <details open><summary>{sourcePreview.source_id} 官方快照</summary><pre className="review-code">{sourcePreview.unified_diff}</pre><details><summary>完整抓取正文</summary><pre className="review-code">{sourcePreview.text}</pre></details></details>}</div>
    <form className="knowledge-update-form" onSubmit={event => { event.preventDefault(); void stage() }}><label className="field-label">选择需要更新的来源<select value={sourceId || releases?.sources?.[0]?.source_id || ''} onChange={event => setSourceId(event.target.value)} disabled={busy}>{releases?.sources?.map(item => <option value={item.source_id} key={item.source_id}>{item.publisher} · {item.title}</option>)}</select></label><button type="button" className="subtle-button" disabled={busy || !releases?.sources?.length} onClick={() => void downloadTemplate()}>下载该来源的更新模板</button><label className="field-label">知识更新文件（JSON）<input type="file" accept="application/json,.json" disabled={busy} onChange={event => setFile(event.target.files?.[0] || null)}/></label><p className="review-hint">在模板中更新对应章节、日期与检索问题，保留 source_sha256 为 auto 可由服务器计算正文哈希。未修改来源由服务器保留，新版本经审核后生效。</p><button type="submit" className="subtle-button" disabled={busy || !file}>{busy ? '正在处理…' : '生成待审核版本'}</button></form>
    {releases && !releases.candidates.length && <p className="muted-empty">尚无知识更新候选。</p>}
    {candidate && <div className="skill-review-layout"><div className="agent-task-list" aria-label="知识更新候选">{releases!.candidates.map(item => <button type="button" className={candidate.candidate_id === item.candidate_id ? 'agent-task-choice active' : 'agent-task-choice'} key={item.candidate_id} disabled={busy} aria-pressed={candidate.candidate_id === item.candidate_id} onClick={() => { setSelected(item.candidate_id); setChecked(false) }}><strong>{item.corpus_version}</strong><small>{labels[item.status] || item.status} · {item.changed_source_ids.length} 个来源</small></button>)}</div><div className="skill-review-detail">
      <h3>{candidate.corpus_version}</h3><p>状态：{labels[candidate.status] || candidate.status}</p><div className="context-note">检索方式：{candidate.retrieval_mode === 'hybrid' ? '关键词与语义向量混合检索' : '全文关键词检索'}。检索探针：{candidate.evaluation.retrieval_probes.filter(item => item.passed).length}/{candidate.evaluation.retrieval_probes.length}；正式回答 held-out 评测尚未运行。{candidate.vector_build && <p>向量缓存复用 {candidate.vector_build.reused}/{candidate.vector_build.total}；本次生成 {candidate.vector_build.embedded} 个向量。</p>}</div>
      {['pending_review','watch'].includes(candidate.status) && <button type="button" className="subtle-button" disabled={busy} onClick={() => void buildVectors()}>构建或更新本地语义索引</button>}
      <h4>来源与日期</h4>{candidate.provenance.map(item => <article className="knowledge-source-meta" key={item.source_id}><strong>{item.source_id} · {item.publisher}</strong><p>{item.document_kind === 'formal_guideline' ? '正式指南' : '通用健康资料'} · 文档版本 {item.version}<br/>发布：{item.published_on || '未知'} · 更新：{item.updated_on || '未知'} · 抓取：{item.retrieved_on}</p></article>)}
      <h4>修改内容</h4><pre className="review-code">{candidate.unified_diff}</pre><details><summary>检索检查与风险</summary><pre className="review-code">{JSON.stringify({ evaluation: candidate.evaluation, risk: candidate.risk }, null, 2)}</pre></details>
      {['pending_review', 'watch'].includes(candidate.status) && <><label className="review-check"><input type="checkbox" checked={checked} onChange={event => setChecked(event.target.checked)}/>我已人工核对官方来源、内容变化、日期与检索结果，批准作为新的知识版本。</label><div className="review-actions"><button type="button" className="primary-button" disabled={busy || !checked || !candidate.evaluation.passed} onClick={() => void act('approve')}>批准发布知识</button><button type="button" className="subtle-button" disabled={busy} onClick={() => void act('watch')}>继续观察</button><button type="button" className="subtle-button" disabled={busy} onClick={() => void act('reject')}>拒绝</button></div></>}
    </div></div>}
  </section>
}
