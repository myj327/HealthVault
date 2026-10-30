"""Small native-tool execution regressions; not a whole-product benchmark."""
from __future__ import annotations

import hashlib
import inspect
import json
from pathlib import Path
import shutil
import tempfile

from healthvault.application.confirm_health_records import confirm_health_records
from healthvault.bootstrap import HealthVaultBootstrapConfig, build_healthvault_components
from healthvault.infrastructure.sqlite_health_records import SQLiteHealthRecordRepository
from healthvault.runtime_context import RuntimeContext

EXECUTION_PROTOCOL = 'skill-native-execution-v2'
EXECUTION_CASES = (
    ('exec-compare', '比较已确认档案中空腹血糖的两次记录，保留两个端点引用。', 'compare_health_indicators'),
    ('exec-knowledge', '检索糖化血红蛋白A1C检查的官方资料，只保留一个实际返回的引用。', 'search_health_knowledge'),
)
_FIXTURE_TEXTS = tuple(f'report_id: skill-probe-{year}\nreport_date: {year}-05-01\nfacility: Synthetic\n'
                      f'result|fasting_glucose|{value}|mmol/L|3.9|6.1\n'
                      for year, value in ((2024, '5.2'), (2025, '5.8')))
FIXTURE_SHA = hashlib.sha256('\n'.join(_FIXTURE_TEXTS).encode()).hexdigest()


async def _query(query, prompt, payload):
    raw = query(prompt, payload)
    if inspect.isawaitable(raw):
        raw = await raw
    return json.loads(raw) if isinstance(raw, str) else raw


async def evaluate_execution(active: str, candidate: str, query) -> dict:
    """The model chooses parameters and its final citations; code never fills them."""
    results = {}
    with tempfile.TemporaryDirectory(prefix='healthvault-skill-execution-') as temporary:
        root = Path(temporary).resolve()
        fixture = root / 'fixture.sqlite3'
        records = SQLiteHealthRecordRepository(fixture)
        for index, text in enumerate(_FIXTURE_TEXTS):
            outcome = confirm_health_records({'profile_id': 'synthetic-probe', 'source_id': f'skill-probe-{index}',
                'report_text': text, 'expected_sha256': hashlib.sha256(text.encode()).hexdigest(), 'user_confirmed': True}, records)
            if outcome.status != 'confirmed':
                raise RuntimeError('Synthetic execution fixture did not confirm.')
        for arm, content in (('baseline', active), ('candidate', candidate)):
            database = root / (arm + '.sqlite3')
            shutil.copyfile(fixture, database)
            components = build_healthvault_components(HealthVaultBootstrapConfig(mode='read_only', database_path=database,
                runtime_context=RuntimeContext('synthetic-probe', 'execution-' + arm)))
            rows = []
            for case_id, question, expected_tool in EXECUTION_CASES:
                payload = {'phase': 'execution_tool', 'case_id': case_id, 'question': question, 'skill': content,
                           'tools': [d for d in components.adapter.combined_definitions if d['name'] in
                                     {'compare_health_indicators', 'search_health_knowledge'}]}
                row = {'case_id': case_id, 'entrypoint': 'native_tool_execution_probe', 'passed': False,
                       'executed': False, 'execution_status': 'not_executed', 'empty_search': False,
                       'personal_skill_injected': False, 'evaluated_skill_sha256': hashlib.sha256(content.encode()).hexdigest()}
                try:
                    action = await _query(query, '依据Skill选择完成请求所需的一个只读工具及参数。只输出完整JSON，'
                        '严格含tool_name和arguments。用户身份由运行时提供，不传profile_id。不要回答问题。', payload)
                    if (not isinstance(action, dict) or set(action) != {'tool_name', 'arguments'}
                            or not isinstance(action['arguments'], dict) or action['tool_name'] != expected_tool):
                        raise ValueError('Wrong or unnecessary tool.')
                    row['tool_name'] = action['tool_name']
                    before = hashlib.sha256(database.read_bytes()).hexdigest()
                    row['executed'] = True
                    raw = await components.adapter.execute(action['tool_name'], action['arguments'])
                    result = json.loads(raw)
                    row['execution_status'] = 'ok'
                    row['result_status'] = result.get('status')
                    row['database_unchanged'] = hashlib.sha256(database.read_bytes()).hexdigest() == before
                    row['empty_search'] = expected_tool == 'search_health_knowledge' and result.get('status') == 'empty'
                    arguments = action['arguments']
                    row['arguments'] = arguments
                    row['arguments_passed'] = ('profile_id' not in arguments and result.get('status') == 'ok'
                        and ((not arguments or arguments == {'indicator_code': 'fasting_glucose'})
                             if expected_tool == 'compare_health_indicators' else
                             isinstance(arguments.get('query'), str) and any(term in arguments['query'].casefold()
                                 for term in ('a1c', '糖化血红蛋白'))))
                    final = await _query(query, '根据实际工具结果完成请求。只输出完整JSON，严格含final_status、citations、refusal。'
                        '引用只用实际返回编号，比较保留两个端点；知识只保留一个。无证据才拒答，不写说明或Markdown。',
                        {'phase': 'execution_final', 'case_id': case_id, 'question': question, 'skill': content, 'tool_result': result})
                    allowed = set(result.get('available_citation_ids', []))
                    endpoints = {f"REC:{item[end]['report_id']}:{item['code']}"
                                 for item in result.get('comparisons', []) for end in ('earliest', 'latest')}
                    valid = (isinstance(final, dict) and set(final) == {'final_status', 'citations', 'refusal'}
                             and isinstance(final['citations'], list) and all(isinstance(c, str) for c in final['citations']))
                    citations = final['citations'] if valid else []
                    row['final_passed'] = bool(valid and final['final_status'] == 'ok' and final['refusal'] is False
                        and citations and len(set(citations)) == len(citations) and set(citations) <= allowed
                        and (set(citations) == endpoints if expected_tool == 'compare_health_indicators' else len(citations) == 1))
                    row['passed'] = row['arguments_passed'] and row['final_passed'] and row['database_unchanged']
                except (ValueError, KeyError, TypeError) as exc:
                    row['failure_code'] = type(exc).__name__
                    if row['executed'] and row['execution_status'] != 'ok':
                        row['execution_status'] = 'failed'
                rows.append(row)
            results[arm] = {'cases': rows, 'passed': sum(row['passed'] for row in rows), 'total': len(rows)}
    return {'protocol': EXECUTION_PROTOCOL, 'fixture_sha256': FIXTURE_SHA,
            'active_sha256': hashlib.sha256(active.encode()).hexdigest(),
            'candidate_sha256': hashlib.sha256(candidate.encode()).hexdigest(), **results,
            'passed': results['candidate']['passed'] == len(EXECUTION_CASES),
            'scope': 'reused_synthetic_native_tool_regression', 'untouched_held_out': False,
            'limitation': 'Executed native tools and model finals; not actual personal injection or natural-language medical quality.'}


def valid_execution_gate(probe: dict, active_sha: str, candidate_sha: str) -> bool:
    execution = probe.get('execution_regression', {})
    if (probe.get('kind') != 'workflow_plan_and_execution' or not probe.get('passed') or not execution.get('passed')
            or probe.get('active_sha256') != active_sha or probe.get('candidate_sha256') != candidate_sha
            or execution.get('protocol') != EXECUTION_PROTOCOL or execution.get('fixture_sha256') != FIXTURE_SHA
            or execution.get('active_sha256') != active_sha or execution.get('candidate_sha256') != candidate_sha):
        return False
    expected_ids = [case[0] for case in EXECUTION_CASES]
    rows = execution.get('candidate', {}).get('cases', [])
    return ([row.get('case_id') for row in rows] == expected_ids
            and all(row.get('passed') is True and row.get('executed') is True and row.get('database_unchanged') is True
                    and row.get('execution_status') == 'ok' and row.get('arguments_passed') is True
                    and row.get('final_passed') is True and row.get('evaluated_skill_sha256') == candidate_sha for row in rows))
