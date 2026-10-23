from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from healthvault.evaluation import current_delivery as delivery


class CurrentDeliveryTests(unittest.TestCase):
    def test_results_cannot_overwrite_a_previous_run(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "result.json"
            delivery.save(path, {"original": True})
            original = path.read_bytes()
            with self.assertRaises(FileExistsError):
                delivery.save(path, {"original": False})
            self.assertEqual(original, path.read_bytes())

    def test_implementation_binding_detects_source_changes_without_credentials(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "healthvault").mkdir()
            source = root / "healthvault/current.py"
            source.write_text("version = 1\n", encoding="utf-8")
            for name in ("AGENTS.md", "web-ui/package.json", "requirements-web.txt"):
                path = root / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text("public configuration\n", encoding="utf-8")
            (root / ".env").write_text("SYNTHETIC_SECRET=do_not_collect\n", encoding="utf-8")
            with patch.object(delivery, "ROOT", root):
                before = delivery.implementation_binding()
                source.write_text("version = 2\n", encoding="utf-8")
                after = delivery.implementation_binding()
            self.assertNotEqual(before["sha256"], after["sha256"])
            self.assertNotIn(".env", before["files"])

    def test_changed_corpus_is_rejected_before_preflight_is_saved(self):
        with tempfile.TemporaryDirectory() as directory, patch.object(delivery, "digest", return_value="mismatched"):
            output = Path(directory)
            with self.assertRaisesRegex(ValueError, "signed Gold"):
                delivery.prepare(output)
            self.assertFalse((output / "implementation-manifest.json").exists())


if __name__ == "__main__":
    unittest.main()
