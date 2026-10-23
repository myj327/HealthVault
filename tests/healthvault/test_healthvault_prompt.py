import hashlib
import unittest

from healthvault.prompt import (
    HEALTHVAULT_SYSTEM_PROMPT,
    build_healthvault_system_prompt,
)


class HealthVaultPromptTests(unittest.TestCase):
    def test_prompt_is_deterministic_safe_and_complete(self):
        first = build_healthvault_system_prompt()
        second = build_healthvault_system_prompt()

        self.assertEqual(HEALTHVAULT_SYSTEM_PROMPT, first)
        self.assertEqual(first, second)
        self.assertTrue(first.strip())
        first.encode("utf-8")
        required_phrases = (
            "synthetic",
            "not a substitute for a doctor",
            "record citation",
            "knowledge citation",
            "do not diagnose",
            "risk",
            "treatment",
            "improvement or deterioration",
            "profile_id",
            "read-only",
            "confirm_health_records",
            "file, Shell, generic MCP, or other native tools",
            "native ToolRegistry",
        )
        for phrase in required_phrases:
            with self.subTest(phrase=phrase):
                self.assertIn(phrase.casefold(), first.casefold())
        for forbidden in (
            "API_KEY",
            "sk-",
            "C:\\Users\\",
            "/Users/",
            "2026-",
            "real patient",
        ):
            self.assertNotIn(forbidden, first)

        digest = hashlib.sha256(first.encode("utf-8")).hexdigest()
        self.assertEqual(64, len(digest))


if __name__ == "__main__":
    unittest.main()
