import unittest
from importlib.metadata import version
from pathlib import Path

from healthvault.rag import acquire_official_sources as acquisition


FIXTURE = (
    Path(__file__).parents[1]
    / "fixtures"
    / "health_knowledge"
    / "official_page.html"
)


class KnowledgeAcquisitionTests(unittest.TestCase):
    def normalize_fixture(self):
        title_paths = {
            ("Fixture Health Guide", "Allowed Topic"): ("result_communication",),
            (
                "Fixture Health Guide",
                "Allowed Topic",
                "Allowed Detail",
            ): ("result_communication",),
            (
                "Fixture Health Guide",
                "Allowed Topic",
                "Allowed Detail",
                "Allowed Leaf",
            ): ("visit_preparation",),
            ("Fixture Health Guide", "Short standalone fact"): (
                "visit_preparation",
            ),
        }
        spec = acquisition.SourceSpec(
            filename=FIXTURE.name,
            source_id="fixture-official-page",
            publisher="Synthetic Federal Publisher",
            title="Fixture Health Guide",
            version="fixture-v1",
            url="https://example.test/official-page",
            title_paths=title_paths,
        )
        return acquisition.normalize_official_html(
            spec,
            FIXTURE.read_text(encoding="utf-8"),
        )

    def test_trafilatura_normalization_removes_navigation_and_ui_prompts(self):
        snapshot = self.normalize_fixture()
        body = str(snapshot)

        for excluded in (
            "Book traversal links",
            "Previous Next",
            "PDF version",
            "Page introduction outside",
        ):
            with self.subTest(excluded=excluded):
                self.assertNotIn(excluded, body)

    def test_real_h1_h2_h3_h4_are_title_paths_not_heading_blocks(self):
        snapshot = self.normalize_fixture()

        self.assertEqual(
            [
                ["Fixture Health Guide", "Allowed Topic"],
                ["Fixture Health Guide", "Allowed Topic", "Allowed Detail"],
                [
                    "Fixture Health Guide",
                    "Allowed Topic",
                    "Allowed Detail",
                    "Allowed Leaf",
                ],
                ["Fixture Health Guide", "Short standalone fact"],
            ],
            [section["title_path"] for section in snapshot["sections"]],
        )
        self.assertEqual(
            snapshot["title_path_whitelist"],
            [section["title_path"] for section in snapshot["sections"]],
        )
        self.assertFalse(
            any(
                block["kind"] == "heading"
                for section in snapshot["sections"]
                for block in section["blocks"]
            )
        )

    def test_non_whitelisted_chapters_never_enter_snapshot(self):
        snapshot = self.normalize_fixture()
        body = str(snapshot)

        self.assertNotIn("Complete blood count", body)
        self.assertNotIn("References", body)
        self.assertNotIn("excluded test", body)
        self.assertNotIn("Third-party reference", body)

    def test_trafilatura_is_confined_to_build_time_acquisition_module(self):
        runtime_root = Path(__file__).parents[2] / "healthvault"
        importers = []
        for path in runtime_root.rglob("*.py"):
            if path.name == "acquire_official_sources.py":
                continue
            if "trafilatura" in path.read_text(encoding="utf-8"):
                importers.append(path)
        self.assertEqual([], importers)
        project_root = Path(__file__).parents[2]
        self.assertEqual(
            "trafilatura==2.2.0\n",
            (project_root / "requirements-rag-build.txt").read_text(
                encoding="utf-8"
            ),
        )
        self.assertNotIn(
            "trafilatura",
            (project_root / "requirements.txt").read_text(encoding="utf-8"),
        )
        self.assertEqual("2.2.0", version("trafilatura"))


if __name__ == "__main__":
    unittest.main()
