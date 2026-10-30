export type Evidence = { quote?: string; line_start?: number; span_start?: number; span_end?: number; regions?: Region[]; fields?: Record<string, { text: string; regions?: Region[] }> }
type Region = { page?: number; row_index?: number; column_index?: number; polygon: number[][]; page_width?: number; page_height?: number; corrected_text?: boolean }
export default function EvidenceLocation({ evidence, imageUrl, pdf = false }: { evidence: Evidence; imageUrl?: string | null; pdf?: boolean }) {
  const regions = evidence.regions || []
  const first = regions[0]
  const pageUrl = pdf && first?.page && imageUrl ? imageUrl.replace(/\/(image|original)(?=\?|$)/, `/pages/${first.page}`) : imageUrl
  const canDraw = pageUrl && first?.page_width && first?.page_height
  return <details className="evidence"><summary>查看原文与位置</summary><code>{evidence.quote}</code>
    <p>{evidence.line_start ? `第 ${evidence.line_start} 行` : ''}{evidence.span_start !== undefined ? ` · 原文位置 ${evidence.span_start}–${evidence.span_end}` : ''}</p>
    {regions.map((region, index) => <small key={index}>第 {region.page || 1} 页{region.row_index ? ` · 第 ${region.row_index} 行` : ''}{region.column_index ? ` · 第 ${region.column_index} 列` : ''}{region.corrected_text ? ' · 已人工修正文本文字' : ''} </small>)}
    {canDraw && <svg className="evidence-region" viewBox={`0 0 ${first.page_width} ${first.page_height}`} role="img" aria-label="原件中对应的字段区域"><image href={pageUrl!} width={first.page_width} height={first.page_height}/>{regions.filter(region => region.page === first.page).map((region, index) => <polygon key={index} points={region.polygon.map(point => point.join(',')).join(' ')} fill="rgba(234,170,25,.16)" stroke="#a25b00" strokeWidth={Math.max(1, first.page_width! / 300)}/>)}</svg>}
    {evidence.fields && <details><summary>字段对应关系</summary>{Object.entries(evidence.fields).map(([name, field]) => <p key={name}>{({ original_text: '明示临床项目', date_expression: '原文日期表达', indicator: '指标', value: '结果', report_date: '报告日期', unit: '单位', reference_range: '参考范围', reference_lower: '参考下限', reference_upper: '参考上限', dimensions: '尺寸', body_site: '部位', location: '位置', laterality: '左右侧', finding_type: '类型' } as Record<string, string>)[name] || name}：{field.text}</p>)}</details>}
  </details>
}
