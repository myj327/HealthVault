import unittest

from healthvault.tools.registry import ToolRegistry


SYNTHETIC_TOOL = {
    "name": "synthetic_healthvault_probe",
    "description": "Return a deterministic synthetic probe result.",
    "input_schema": {
        "type": "object",
        "properties": {
            "marker": {"type": "string"},
        },
        "required": ["marker"],
    },
}


class ToolRegistryTests(unittest.IsolatedAsyncioTestCase):
    async def test_register_contains_lists_and_executes_tool(self):
        received_inputs = []

        async def handler(tool_input):
            received_inputs.append(tool_input)
            return "TASK004_REGISTRY_OK"

        registry = ToolRegistry()
        registry.register(SYNTHETIC_TOOL, handler)
        tool_input = {"marker": "TASK004_SYNTHETIC"}

        result = await registry.execute("synthetic_healthvault_probe", tool_input)

        self.assertTrue(registry.contains("synthetic_healthvault_probe"))
        self.assertFalse(registry.contains("missing_healthvault_tool"))
        self.assertEqual([SYNTHETIC_TOOL], registry.definitions)
        self.assertEqual([tool_input], received_inputs)
        self.assertEqual("TASK004_REGISTRY_OK", result)

    async def test_duplicate_tool_name_fails_immediately(self):
        async def handler(tool_input):
            return str(tool_input)

        registry = ToolRegistry()
        registry.register(SYNTHETIC_TOOL, handler)

        with self.assertRaisesRegex(
            ValueError,
            "synthetic_healthvault_probe",
        ):
            registry.register(dict(SYNTHETIC_TOOL), handler)

    async def test_execute_unknown_tool_raises_clear_error(self):
        registry = ToolRegistry()

        with self.assertRaisesRegex(KeyError, "missing_healthvault_tool"):
            await registry.execute("missing_healthvault_tool", {"marker": "none"})


if __name__ == "__main__":
    unittest.main()
