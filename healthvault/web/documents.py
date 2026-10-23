"""Bounded text extraction for uploaded PDF checkup reports."""

from __future__ import annotations

from typing import Any, Callable


class PDFProcessingError(RuntimeError):
    pass


def extract_pdf_text(
    raw: bytes,
    process_image: Callable[[bytes, str], dict[str, Any]],
) -> str:
    return extract_pdf_document(raw, process_image)["text"]


def extract_pdf_document(
    raw: bytes,
    process_image: Callable[[bytes, str], dict[str, Any]],
) -> dict[str, Any]:
    try:
        import pymupdf as fitz
    except ImportError as exc:
        raise PDFProcessingError("PDF 支持尚未安装，请安装 requirements-web.txt。") from exc
    try:
        document = fitz.open(stream=raw, filetype="pdf")
    except Exception as exc:
        raise PDFProcessingError("PDF 文件无法读取。") from exc
    try:
        if not 1 <= len(document) <= 12:
            raise PDFProcessingError("仅支持 1–12 页的 PDF 报告。")
        pages: list[str] = []
        blocks: list[dict[str, Any]] = []
        page_artifacts: list[dict[str, Any]] = []
        warnings: list[dict[str, Any]] = []
        for page_number, page in enumerate(document, start=1):
            rotation = page.rotation
            page.set_rotation(0)
            native_text = page.get_text("text", sort=True).strip()
            page_blocks = []
            words = page.get_text("words", sort=True)
            for word in words:
                x0,y0,x1,y1,text = word[:5]
                if str(text).strip():
                    page_blocks.append({"page":page_number, "text":str(text), "polygon":[[x0,y0],[x1,y0],[x1,y1],[x0,y1]],
                                        "coordinate_space":"pdf_unrotated_points", "page_width":page.rect.width,
                                        "page_height":page.rect.height, "confidence":1.0})
            images = [fitz.Rect(item["bbox"]) & page.rect for item in page.get_image_info()]
            regions = []
            for rectangle in images:
                native_chars = sum(len(str(word[4])) for word in words if (fitz.Rect(word[:4]) & rectangle).get_area() > 0)
                if not rectangle.is_empty and native_chars < 20:
                    if not any(existing.contains(rectangle) for existing in regions):
                        regions = [existing for existing in regions if not rectangle.contains(existing)]
                        regions.append(rectangle)
            if len(native_text) < 20 and not regions:
                regions = [page.rect]
            artifact: dict[str, Any] = {"page": page_number, "native_text": native_text, "mode": "ocr" if regions else "text",
                                       "rotation": rotation, "regions":[]}
            ocr_texts = []
            for region_index, rectangle in enumerate(regions, 1):
                image = page.get_pixmap(matrix=fitz.Matrix(2, 2), clip=rectangle, alpha=False)
                try:
                    ocr = process_image(image.tobytes("png"), f"page-{page_number}.png")
                    ocr_texts.append(str(ocr.get("text") or ""))
                    artifact.update(confidence=ocr.get("confidence"), image_sha256=ocr.get("image_sha256"))
                    artifact["regions"].append({"region":region_index, "bbox":list(rectangle), "status":"recognized"})
                    incoming = ocr.get("blocks", [])
                    if not incoming and ocr.get("text"):
                        incoming = [{"text":ocr["text"], "polygon":[[0,0],[image.width,0],[image.width,image.height],[0,image.height]], "confidence":0.0}]
                    for block in incoming:
                        polygon = [[point[0]/2+image.x/2, point[1]/2+image.y/2] for point in block.get("polygon", [])]
                        if not polygon:
                            continue
                        box = fitz.Rect(min(point[0] for point in polygon), min(point[1] for point in polygon), max(point[0] for point in polygon), max(point[1] for point in polygon))
                        duplicate = any(str(existing["text"]).strip() == str(block.get("text", "")).strip()
                                        and (fitz.Rect(min(p[0] for p in existing["polygon"]), min(p[1] for p in existing["polygon"]), max(p[0] for p in existing["polygon"]), max(p[1] for p in existing["polygon"])) & box).get_area() > 0
                                        for existing in page_blocks)
                        if not duplicate:
                            page_blocks.append({**block, "polygon":polygon, "page":page_number, "coordinate_space":"pdf_unrotated_points",
                                                "page_width":page.rect.width, "page_height":page.rect.height})
                except Exception as exc:
                    from healthvault.ocr.pipeline import OCRProcessingError
                    if not isinstance(exc, OCRProcessingError):
                        raise
                    artifact.update(mode="ocr_failed", error=str(exc))
                    artifact["regions"].append({"region":region_index, "bbox":list(rectangle), "status":"failed"})
                    warnings.append({"page": page_number, "code": "page_ocr_failed", "message": str(exc)})
            from healthvault.web.evidence import layout_blocks
            text, page_blocks = layout_blocks(page_blocks)
            text = text or native_text or "\n".join(ocr_texts).strip()
            artifact["ocr_text"] = "\n".join(ocr_texts)
            if text:
                prefix = "" if len(document) == 1 else f"第{page_number}页\n"
                offset = sum(len(part)+1 for part in pages) + len(prefix)
                for block in page_blocks:
                    block["span_start"] += offset; block["span_end"] += offset
                blocks.extend(page_blocks)
                pages.append(prefix + text)
            else:
                warnings.append({"page": page_number, "code": "empty_page_text", "message": "该页未识别到文字，请对照原件核对。"})
            page_artifacts.append(artifact)
        combined = "\n".join(pages)
        if len(combined) > 100_000:
            raise PDFProcessingError("PDF 提取文字超过 100000 字，请分拆报告。")
        return {"text": combined, "blocks": blocks, "pages": page_artifacts, "warnings": warnings}
    finally:
        document.close()
