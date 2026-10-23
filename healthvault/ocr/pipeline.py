"""Deterministic local OCR artifacts backed by RapidOCR on CPU."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
from pathlib import Path
from typing import Any


DEFAULT_ARTIFACT_ROOT = Path(".task_outputs") / "healthvault-ui" / "ocr"
SUPPORTED_EXTENSIONS = {".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg"}
LOW_CONFIDENCE_THRESHOLD = 0.8


class OCRProcessingError(RuntimeError):
    """A safe, user-readable failure at the local OCR boundary."""


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_text(value: str) -> str:
    return _sha256_bytes(value.encode("utf-8"))


def _package_version(name: str) -> str:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return "unknown"


def _canonical_json(payload: dict[str, Any]) -> str:
    return json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n"


class OCRPipeline:
    """Run one-page OCR and persist a content-addressed, reusable artifact."""

    def __init__(
        self,
        artifact_root: str | Path = DEFAULT_ARTIFACT_ROOT,
        *,
        engine: Any | None = None,
    ) -> None:
        self.artifact_root = Path(artifact_root)
        self._engine = engine

    def _get_engine(self) -> Any:
        if self._engine is None:
            try:
                from rapidocr import RapidOCR

                self._engine = RapidOCR()
            except Exception as exc:
                raise OCRProcessingError(
                    f"本地 OCR 引擎初始化失败：{exc}"
                ) from exc
        return self._engine

    def process(self, image_bytes: bytes, filename: str) -> dict[str, Any]:
        if not isinstance(image_bytes, bytes) or not image_bytes:
            raise OCRProcessingError("图片内容为空，无法执行本地 OCR。")
        extension = Path(filename).suffix.lower()
        if extension not in SUPPORTED_EXTENSIONS:
            raise OCRProcessingError("仅支持 PNG、JPG 或 JPEG 图片。")

        image_sha256 = _sha256_bytes(image_bytes)
        artifact_dir = self.artifact_root / image_sha256
        artifact_path = artifact_dir / "ocr.json"
        if artifact_path.is_file():
            return self._load_reusable_artifact(
                artifact_dir, artifact_path, image_bytes, image_sha256
            )

        artifact_dir.mkdir(parents=True, exist_ok=True)
        original_filename = f"original{extension}"
        original_path = artifact_dir / original_filename
        if not original_path.is_file():
            original_path.write_bytes(image_bytes)

        try:
            from healthvault.ocr.preprocess import prepare_image, restore_blocks
            prepared_path = artifact_dir / "prepared.png"
            preparation = prepare_image(image_bytes, prepared_path)
            output = self._get_engine()(str(prepared_path if preparation.get("status") == "prepared" else original_path))
            payload = self._build_payload(
                output,
                image_sha256=image_sha256,
                original_filename=original_filename,
                media_type=SUPPORTED_EXTENSIONS[extension],
            )
            restore_blocks(payload["blocks"], preparation)
            from healthvault.web.evidence import layout_blocks
            payload["text"], payload["blocks"] = layout_blocks(payload["blocks"])
            payload["text_sha256"] = _sha256_text(payload["text"])
            payload["preprocessing"] = preparation
        except OCRProcessingError:
            raise
        except Exception as exc:
            raise OCRProcessingError(f"本地 OCR 失败：{exc}") from exc

        temporary_path = artifact_dir / "ocr.json.tmp"
        temporary_path.write_text(_canonical_json(payload), encoding="utf-8", newline="\n")
        temporary_path.replace(artifact_path)
        return self._with_artifact_paths(payload, artifact_dir, reused=False)

    def _build_payload(
        self,
        output: Any,
        *,
        image_sha256: str,
        original_filename: str,
        media_type: str,
    ) -> dict[str, Any]:
        raw_texts = getattr(output, "txts", None)
        raw_boxes = getattr(output, "boxes", None)
        raw_scores = getattr(output, "scores", None)
        texts = tuple(raw_texts) if raw_texts is not None else ()
        boxes = tuple(raw_boxes) if raw_boxes is not None else ()
        scores = tuple(raw_scores) if raw_scores is not None else ()
        if not texts:
            raise OCRProcessingError("本地 OCR 未识别到可用文本。")
        if not (len(texts) == len(boxes) == len(scores)):
            raise OCRProcessingError("本地 OCR 返回的文本、坐标和置信度数量不一致。")

        blocks: list[dict[str, Any]] = []
        for index, (text, box, score) in enumerate(zip(texts, boxes, scores), start=1):
            normalized_text = str(text).strip()
            if not normalized_text:
                continue
            try:
                polygon = [
                    [round(float(point[0]), 3), round(float(point[1]), 3)]
                    for point in box
                ]
                confidence = round(float(score), 6)
            except (TypeError, ValueError, IndexError) as exc:
                raise OCRProcessingError("本地 OCR 返回了无效坐标或置信度。") from exc
            blocks.append(
                {
                    "index": index,
                    "page": 1,
                    "text": normalized_text,
                    "polygon": polygon,
                    "confidence": confidence,
                }
            )
        if not blocks:
            raise OCRProcessingError("本地 OCR 未识别到可用文本。")

        full_text = "\n".join(block["text"] for block in blocks)
        confidence_values = [block["confidence"] for block in blocks]
        low_confidence = [
            block["index"]
            for block in blocks
            if block["confidence"] < LOW_CONFIDENCE_THRESHOLD
        ]
        return {
            "schema_version": 1,
            "page": 1,
            "original_filename": original_filename,
            "media_type": media_type,
            "image_sha256": image_sha256,
            "text": full_text,
            "text_sha256": _sha256_text(full_text),
            "blocks": blocks,
            "confidence": {
                "average": round(sum(confidence_values) / len(confidence_values), 6),
                "minimum": min(confidence_values),
                "low_confidence_threshold": LOW_CONFIDENCE_THRESHOLD,
                "low_confidence_blocks": low_confidence,
            },
            "engine": {
                "name": "rapidocr",
                "version": _package_version("rapidocr"),
                "runtime": "onnxruntime",
                "runtime_version": _package_version("onnxruntime"),
                "device": "cpu",
            },
        }

    def _load_reusable_artifact(
        self,
        artifact_dir: Path,
        artifact_path: Path,
        image_bytes: bytes,
        expected_image_sha256: str,
    ) -> dict[str, Any]:
        try:
            payload = json.loads(artifact_path.read_text(encoding="utf-8"))
            original_path = artifact_dir / payload["original_filename"]
            if payload["image_sha256"] != expected_image_sha256:
                raise ValueError("image hash mismatch")
            if _sha256_bytes(original_path.read_bytes()) != expected_image_sha256:
                raise ValueError("saved image hash mismatch")
            if _sha256_bytes(image_bytes) != expected_image_sha256:
                raise ValueError("uploaded image hash mismatch")
            if _sha256_text(payload["text"]) != payload["text_sha256"]:
                raise ValueError("OCR text hash mismatch")
            if not payload["blocks"] or payload["page"] != 1:
                raise ValueError("OCR blocks are missing")
        except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
            raise OCRProcessingError(f"已有 OCR artifact 无效：{exc}") from exc
        return self._with_artifact_paths(payload, artifact_dir, reused=True)

    @staticmethod
    def _with_artifact_paths(
        payload: dict[str, Any], artifact_dir: Path, *, reused: bool
    ) -> dict[str, Any]:
        result = dict(payload)
        result["artifact"] = {
            "directory": str(artifact_dir),
            "image_path": str(artifact_dir / payload["original_filename"]),
            "json_path": str(artifact_dir / "ocr.json"),
            "reused": reused,
        }
        return result
