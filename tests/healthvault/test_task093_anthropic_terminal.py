import contextlib
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import AsyncMock, MagicMock, patch

from agents import agent as agent_module
from agents.ui import quiet_agent_ui
from healthvault.agent_runtime import HealthVaultAgentRuntimeConfig, build_healthvault_agent_runtime
from healthvault.model_config import HealthVaultModelConfig
from healthvault.runtime_context import RuntimeContext
from tests.healthvault.test_task086_completion_contract import final


class AnthropicTerminalTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        scratch = Path(__file__).resolve().parents[2] / '.task_outputs/task-093-tests'
        scratch.mkdir(parents=True, exist_ok=True)
        self.temp = tempfile.TemporaryDirectory(dir=scratch)
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.stack = contextlib.ExitStack()
        self.addCleanup(self.stack.close)
        self.stack.enter_context(quiet_agent_ui())
        self.stack.enter_context(patch.object(agent_module.anthropic, 'AsyncAnthropic', return_value=MagicMock()))
        self.runtime = build_healthvault_agent_runtime(HealthVaultAgentRuntimeConfig(
            context=RuntimeContext('alice', 'req'), database_path=self.root / 'records.sqlite3',
            model_config=HealthVaultModelConfig('anthropic', 'claude-fake', 'https://invalid.local', 'synthetic-key')))
        self.executor = AsyncMock(return_value='{"status":"empty","available_citation_ids":[]}')
        self.runtime.agent._tool_executor = self.executor

    def query(self, text):
        return SimpleNamespace(type='tool_use', id='call-'+text, name='search_health_knowledge', input={'query': text})

    def script(self, blocks):
        pending = iter(blocks)
        async def response(on_tool_block_complete=None):
            block = next(pending)
            if block.type == 'text':
                self.runtime.agent._emit_text(block.text)
            return SimpleNamespace(content=[block], usage=SimpleNamespace(input_tokens=10, output_tokens=5))
        supplier = AsyncMock(side_effect=response)
        self.runtime.agent._call_anthropic_stream = supplier
        return supplier

    async def test_duplicate_rejection_stops_supplier_and_records_zero_extra_usage(self):
        supplier = self.script([self.query('A1C'), self.query(' a1c ')])
        original_openai = self.runtime.agent._call_openai_stream
        reply = await self.runtime.run('查询知识资料')
        self.assertEqual('rejected', json.loads(reply.text)['final_status'])
        self.assertEqual(2, supplier.await_count)
        self.executor.assert_awaited_once()
        self.assertEqual({'input': 20, 'output': 10}, reply.tokens)
        self.assertIs(supplier, self.runtime.agent._call_anthropic_stream)
        self.assertEqual(original_openai, self.runtime.agent._call_openai_stream)
        self.assertEqual('text', self.runtime.agent._anthropic_messages[-1]['content'][0]['type'])

    async def test_legal_flow_passes_through_callback_and_keeps_completion_contract(self):
        supplier = self.script([self.query('A1C'), SimpleNamespace(type='text', text=final('insufficient_evidence', (), True))])
        reply = await self.runtime.run('查询知识资料')
        self.assertEqual('insufficient_evidence', json.loads(reply.text)['final_status'])
        self.assertEqual(2, supplier.await_count)
        self.assertIn('on_tool_block_complete', supplier.await_args.kwargs)
        self.assertEqual({'input': 20, 'output': 10}, reply.tokens)

    async def test_exception_restores_both_supplier_methods(self):
        supplier = AsyncMock(side_effect=RuntimeError('synthetic_supplier_failure'))
        self.runtime.agent._call_anthropic_stream = supplier
        original_openai = self.runtime.agent._call_openai_stream
        with self.assertRaisesRegex(RuntimeError, 'synthetic_supplier_failure'):
            await self.runtime.run('查询知识资料')
        self.assertIs(supplier, self.runtime.agent._call_anthropic_stream)
        self.assertEqual(original_openai, self.runtime.agent._call_openai_stream)
        self.assertIsNone(self.runtime.agent.completion_check)

    async def test_terminal_state_does_not_leak_into_next_request(self):
        self.script([self.query('A1C'), self.query(' a1c ')])
        await self.runtime.run('查询知识资料')
        supplier = self.script([self.query('A1C'), SimpleNamespace(type='text', text=final('insufficient_evidence', (), True))])
        reply = await self.runtime.run('查询知识资料')
        self.assertEqual('insufficient_evidence', json.loads(reply.text)['final_status'])
        self.assertEqual(2, supplier.await_count)
        self.assertEqual(2, self.executor.await_count)


if __name__ == '__main__':
    unittest.main()
