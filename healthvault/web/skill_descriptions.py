"""Short explanations derived from stored changes and lifecycle decisions."""
import re


def candidate_explanation(record: dict) -> str:
    labels = {'When to Use': '适用条件', 'Prerequisites': '执行前检查', 'Steps': '工具选择与执行顺序',
              'Failure Handling': '失败后的处理与停止条件', 'Verification': '结果与引用核对'}
    sections = []
    for patch in record.get('patches', []):
        if patch.get('target_section') not in labels:
            continue
        label = labels[patch['target_section']]
        step = re.match(r'\s*(\d+)\.', patch.get('target_paragraph', ''))
        if patch['target_section'] == 'Steps' and step:
            label += f'（第 {step[1]} 步）'
        if label not in sections:
            sections.append(label)
    return '这次调整：' + '、'.join(sections) + '。具体修改见下方对比。' if sections else '候选尚未形成可核对的流程修改。'


def usage_explanation(summary: dict) -> str | None:
    action = summary.get('last_action') or {}
    actions = action.get('actions', [])
    if action.get('status') == 'active_skill_missing':
        return '没有可停用的个人工作流版本。'
    if action.get('reason') == 'user_requested':
        return '你主动回滚了上次个人工作流更新。' if 'rollback' in actions else '你主动停用了当前个人工作流版本。'
    version = action.get('problem_sha256')
    evaluations = [item for item in summary.get('evaluations', []) if item.get('skill_sha256') == version]
    if 'disable' in actions:
        if not evaluations:
            return '该个人版本已停用；旧记录未保存具体原因。'
        reason = '发生了禁止工具或权限边界的安全失败' if any(item.get('safety_hard_failures', 0) for item in evaluations) else '连续任务的工具、结果或引用检查未达到要求'
        return reason + ('，已停用该版本并恢复之前版本。' if 'rollback' in actions else '，该版本已停用。')
    if 'watch' in actions:
        return '有效任务数量或通过情况尚不足，继续观察实际使用结果。'
    if 'keep' in actions:
        return '近期任务的工具、结果和引用检查通过，继续使用当前版本。'
    return None
