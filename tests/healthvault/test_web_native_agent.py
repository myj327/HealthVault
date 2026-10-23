import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

from fastapi.testclient import TestClient

from healthvault.agent_runtime import HealthVaultChatResult
from healthvault.model_config import HealthVaultModelConfig
from healthvault.ui.controller import HealthVaultUIController
from healthvault.web.api import create_app


SKILL = """---
name: HealthVault-Task-Execution
description: Synthetic web runtime Skill.
---
## When to Use
Use for tests.
## Prerequisites
Read-only.
## Steps
Use one tool.
## Failure Handling
Stop safely.
## Verification
Verify output.
## Safety Boundaries
Never write.
"""


class WebNativeAgentTests(unittest.TestCase):
    def test_web_agent_uses_native_registry_and_loads_project_skill(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            skill_root = root / ".bear" / "skills"
            skill_path = skill_root / "HealthVault-Task-Execution" / "SKILL.md"
            skill_path.parent.mkdir(parents=True)
            skill_path.write_text(SKILL, encoding="utf-8")
            controller = HealthVaultUIController(root / "records.sqlite3")
            fake_agent = SimpleNamespace(
                _openai_messages=[],
                _folded_session_memories=[],
                restore_session=MagicMock(),
            )
            fake_runtime = SimpleNamespace(
                agent=fake_agent,
                manifest={"tool_names": ["query_health_timeline"]},
                trace_recorder=None,
                run=AsyncMock(return_value=HealthVaultChatResult(
                    text="native ok",
                    tokens={"input": 1, "output": 1},
                    request_id="request",
                    tool_names=["query_health_timeline"],
                    project_skills=[{"name": "HealthVault-Task-Execution", "sha256": "a" * 64}],
                )),
            )
            app = create_app(
                controller,
                skill_root=skill_root,
                skill_evolution_root=root / "evolution",
                skill_patch_side_query=AsyncMock(),
                model_question_planner=lambda *_a, **_k: {'protocol': 'question_plan_v1', 'additional_points': []},
            )
            model = HealthVaultModelConfig(
                "openai-compatible", "fake", "https://invalid.local/v1", "key"
            )
            with patch(
                "healthvault.web.api.configured_web_model", return_value=model,
            ), patch(
                "healthvault.agent_runtime.build_healthvault_agent_runtime",
                return_value=fake_runtime,
            ) as builder, patch(
                "healthvault.mcp.runtime_bridge.attach_approved_mcp",
                new=AsyncMock(side_effect=AssertionError("Web must not attach MCP")),
            ) as attach:
                with TestClient(app) as client:
                    registered = client.post("/api/auth/register", json={
                        "username": "native-user", "password": "synthetic-password-123",
                    }).json()
                    headers = {"x-csrf-token": registered["csrf_token"]}
                    profile = registered["profile_id"]
                    capabilities = client.get(f"/api/profiles/{profile}/agent/capabilities")
                    self.assertEqual(200, capabilities.status_code, capabilities.text)
                    response = client.post(
                        f"/api/profiles/{profile}/agent/tasks/task-native/messages",
                        headers=headers,
                        json={
                            "prompt": "show my timeline",
                            "expected_version": 0,
                            "model_consent": True,
                            "allow_web_search": False,
                        },
                    )

            self.assertEqual(200, response.status_code, response.text)
            self.assertEqual(["query_health_timeline"], response.json()["tool_names"])
            config = builder.call_args.args[0]
            self.assertEqual(("HealthVault-Task-Execution",), config.skill_names)
            self.assertEqual(skill_root, config.skill_root)
            self.assertIsNotNone(config.evolution_root)
            self.assertFalse(config.allow_web_search)
            attach.assert_not_awaited()


if __name__ == "__main__":
    unittest.main()
