import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from agents.skills import (
    ProjectSkillWhitelistError,
    format_project_skill_resources,
    load_project_skill_resources,
    project_skill_manifest,
)


def write_skill(root: Path, directory_name: str, parsed_name: str, body: str) -> None:
    skill_dir = root / directory_name
    skill_dir.mkdir(parents=True)
    (skill_dir / "SKILL.md").write_text(
        f"---\nname: {parsed_name}\ndescription: Synthetic test Skill.\n---\n\n{body}\n",
        encoding="utf-8",
    )


class ProjectSkillWhitelistTests(unittest.TestCase):
    def test_empty_allowlist_does_not_scan_or_require_a_root(self):
        with patch(
            "agents.skills.discover_skills",
            side_effect=AssertionError("global Skill discovery must not run"),
        ) as discover:
            resources = load_project_skill_resources(None, ())

        self.assertEqual((), resources)
        self.assertEqual([], project_skill_manifest(resources))
        self.assertEqual("", format_project_skill_resources(resources))
        discover.assert_not_called()

    def test_only_exact_allowed_utf8_skill_is_loaded_and_bounded(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir) / "project" / ".bear" / "skills"
            write_skill(root, "allowed-skill", "allowed-skill", "允许的中文流程。")
            write_skill(root, "ignored-skill", "ignored-skill", "MUST_NOT_APPEAR")

            with patch(
                "agents.skills.discover_skills",
                side_effect=AssertionError("global Skill discovery must not run"),
            ) as discover:
                resources = load_project_skill_resources(
                    root,
                    ("allowed-skill",),
                )

        prompt = format_project_skill_resources(resources)
        manifest = project_skill_manifest(resources)
        self.assertEqual(["allowed-skill"], [resource.name for resource in resources])
        self.assertIn("允许的中文流程。", prompt)
        self.assertIn("BEGIN HEALTHVAULT PROJECT SKILL: allowed-skill", prompt)
        self.assertNotIn("MUST_NOT_APPEAR", prompt)
        self.assertEqual(["allowed-skill"], [item["name"] for item in manifest])
        self.assertRegex(manifest[0]["sha256"], r"^[0-9a-f]{64}$")
        self.assertNotIn(str(root), str(manifest))
        self.assertNotIn("允许的中文流程。", str(manifest))
        discover.assert_not_called()

    def test_invalid_allowlists_and_resources_fail_stably(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir) / ".bear" / "skills"
            write_skill(root, "mismatch", "different-name", "body")
            write_skill(root, "oversized", "oversized", "x" * 100)
            invalid_utf8_dir = root / "invalid-utf8"
            invalid_utf8_dir.mkdir(parents=True)
            (invalid_utf8_dir / "SKILL.md").write_bytes(b"\xff\xfe")
            wrong_root = Path(temp_dir) / "not-project-skills"
            write_skill(wrong_root, "wrong-root", "wrong-root", "body")

            cases = (
                (("missing",), {}, "missing"),
                (("mismatch",), {}, "does not match"),
                (("same", "same"), {}, "duplicate"),
                (("../escape",), {}, "invalid"),
                (("oversized",), {"max_skill_bytes": 32}, "too large"),
                (("invalid-utf8",), {}, "UTF-8"),
            )
            for names, options, marker in cases:
                with self.subTest(names=names):
                    with self.assertRaisesRegex(
                        ProjectSkillWhitelistError,
                        marker,
                    ) as error:
                        load_project_skill_resources(root, names, **options)
                    self.assertNotIn(str(Path.home()), str(error.exception))

            with self.assertRaisesRegex(
                ProjectSkillWhitelistError,
                r"\.bear/skills",
            ):
                load_project_skill_resources(wrong_root, ("wrong-root",))


if __name__ == "__main__":
    unittest.main()
