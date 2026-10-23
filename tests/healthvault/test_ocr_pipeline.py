import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from healthvault.ocr.pipeline import OCRPipeline, OCRProcessingError
from healthvault.ui.controller import HealthVaultUIController


FIXTURE_ROOT = Path(__file__).parents[1] / "fixtures" / "health_reports"
IMAGE_FIXTURE = FIXTURE_ROOT / "images" / "synthetic_checkup_2026.png"


class FakeRapidOCREngine:
    def __init__(self, *, failure: Exception | None = None) -> None:
        self.failure = failure
        self.calls = 0

    def __call__(self, _image_path: str):
        self.calls += 1
        if self.failure is not None:
            raise self.failure
        return SimpleNamespace(
            boxes=[
                [[10, 10], [310, 10], [310, 40], [10, 40]],
                [[10, 50], [310, 50], [310, 80], [10, 80]],
            ],
            txts=("report_id: HV-SYN-2026-001", "report_date: 2026-06-18"),
            scores=(0.99, 0.71),
        )


class OCRPipelineTests(unittest.TestCase):
    def test_real_cpu_ocr_produces_parser_accepted_preview(self):
        with tempfile.TemporaryDirectory() as directory:
            pipeline = OCRPipeline(Path(directory) / "ocr")
            result = pipeline.process(
                IMAGE_FIXTURE.read_bytes(), IMAGE_FIXTURE.name
            )
            controller = HealthVaultUIController(Path(directory) / "records.sqlite3")
            preview = controller.preview_ocr_report(
                source_id=f"image:{result['image_sha256']}",
                report_text=result["text"],
                image_sha256=result["image_sha256"],
                ocr_text_sha256=result["text_sha256"],
                reviewed=True,
            )

            self.assertEqual("ready", preview["preview"]["status"])
            self.assertTrue(preview["preview"]["can_confirm"])
            self.assertGreaterEqual(len(result["blocks"]), 4)
            self.assertEqual("rapidocr", result["engine"]["name"])
            self.assertEqual("onnxruntime", result["engine"]["runtime"])

    def test_artifact_has_hashes_blocks_confidence_and_reuses_same_image(self):
        image = b"synthetic-png-bytes"
        engine = FakeRapidOCREngine()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "ocr"
            pipeline = OCRPipeline(root, engine=engine)
            first = pipeline.process(image, "report.png")
            second = pipeline.process(image, "renamed.jpg")

            image_hash = hashlib.sha256(image).hexdigest()
            artifact_dir = root / image_hash
            saved = json.loads(
                (artifact_dir / "ocr.json").read_text(encoding="utf-8")
            )
            self.assertEqual(1, engine.calls)
            self.assertFalse(first["artifact"]["reused"])
            self.assertTrue(second["artifact"]["reused"])
            self.assertEqual(image_hash, saved["image_sha256"])
            self.assertEqual(
                hashlib.sha256(saved["text"].encode("utf-8")).hexdigest(),
                saved["text_sha256"],
            )
            self.assertEqual(1, saved["page"])
            self.assertEqual(0.85, saved["confidence"]["average"])
            self.assertEqual(0.71, saved["confidence"]["minimum"])
            self.assertEqual([2], saved["confidence"]["low_confidence_blocks"])
            self.assertEqual(2, len(saved["blocks"]))
            self.assertTrue((artifact_dir / saved["original_filename"]).is_file())

    def test_review_gate_change_invalidation_and_confirmation_are_idempotent(self):
        text = (FIXTURE_ROOT / "synthetic_checkup_2024.txt").read_text(
            encoding="utf-8"
        )
        image_hash = "a" * 64
        ocr_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()
        with tempfile.TemporaryDirectory() as directory:
            controller = HealthVaultUIController(Path(directory) / "records.sqlite3")
            unchecked = controller.preview_ocr_report(
                "image:synthetic", text, image_hash, ocr_hash, reviewed=False
            )
            denied = controller.confirm_preview(
                profile_id="ocr-profile",
                preview_state=unchecked,
                current_source_id="image:synthetic",
                current_report_text=text,
                current_image_sha256=image_hash,
                ocr_reviewed=False,
            )
            self.assertEqual("ocr_review_required", denied["status"])
            self.assertEqual(0, controller.timeline("ocr-profile")["report_count"])

            preview = controller.preview_ocr_report(
                "image:synthetic", text, image_hash, ocr_hash, reviewed=True
            )
            for source, changed_text, changed_image, reviewed in (
                ("image:changed", text, image_hash, True),
                ("image:synthetic", text + "\n# edit", image_hash, True),
                ("image:synthetic", text, "b" * 64, True),
                ("image:synthetic", text, image_hash, False),
            ):
                self.assertFalse(
                    controller.preview_matches(
                        preview,
                        source,
                        changed_text,
                        current_image_sha256=changed_image,
                        ocr_reviewed=reviewed,
                    )
                )

            request = {
                "profile_id": "ocr-profile",
                "preview_state": preview,
                "current_source_id": "image:synthetic",
                "current_report_text": text,
                "current_image_sha256": image_hash,
                "ocr_reviewed": True,
            }
            self.assertEqual("confirmed", controller.confirm_preview(**request)["status"])
            self.assertEqual(
                "already_confirmed", controller.confirm_preview(**request)["status"]
            )
            self.assertEqual(1, controller.timeline("ocr-profile")["report_count"])

    def test_ocr_failure_is_readable_and_does_not_change_database(self):
        with tempfile.TemporaryDirectory() as directory:
            pipeline = OCRPipeline(
                Path(directory) / "ocr",
                engine=FakeRapidOCREngine(failure=RuntimeError("bad image")),
            )
            controller = HealthVaultUIController(
                Path(directory) / "records.sqlite3", ocr_pipeline=pipeline
            )
            with self.assertRaisesRegex(OCRProcessingError, "本地 OCR 失败"):
                controller.process_image(b"broken", "broken.png")
            self.assertEqual(0, controller.timeline("ocr-profile")["report_count"])


if __name__ == "__main__":
    unittest.main()
