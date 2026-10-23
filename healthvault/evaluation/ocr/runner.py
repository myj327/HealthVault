"""Offline, patient-split synthetic OCR evaluation using the real CPU pipeline."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from collections import Counter
from decimal import Decimal
from pathlib import Path
from typing import Any

from healthvault.application.report_preview import build_report_preview
from healthvault.ocr.pipeline import LOW_CONFIDENCE_THRESHOLD, OCRPipeline, OCRProcessingError


TEMPLATES = ("clean", "compact", "rotated", "faint", "jpeg")
INDICATORS = ("fasting_glucose", "hba1c", "ldl_cholesterol")
FONT_CANDIDATES = (
    Path("C:/Windows/Fonts/arial.ttf"),
    Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"),
)


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def make_gold(seed: int) -> list[dict[str, Any]]:
    """Generate 10 disjoint synthetic patients, two visits each, five fields per visit."""
    rng = random.Random(seed)
    patients = list(range(10))
    rng.shuffle(patients)
    test_patients = set(patients[:4])
    reports = []
    for patient in range(10):
        for visit in range(2):
            report_id = f"HV-SYN-OCR-P{patient:02d}-V{visit + 1}"
            report_date = f"202{4 + visit}-{patient % 12 + 1:02d}-{10 + patient:02d}"
            values = {
                "fasting_glucose": f"{5.0 + patient * 0.1 + visit * 0.2:.1f}",
                "hba1c": f"{5.1 + patient * 0.1 + visit * 0.1:.1f}",
                "ldl_cholesterol": f"{2.1 + patient * 0.1 + visit * 0.2:.1f}",
            }
            fields = {
                "report_id": report_id,
                "report_date": report_date,
                **{code: {"value": value, "unit": "%" if code == "hba1c" else "mmol/L"}
                   for code, value in values.items()},
            }
            lines = [
                f"report_id: {report_id}",
                f"report_date: {report_date}",
                "facility: Synthetic OCR Clinic",
                f"result|fasting_glucose|{values['fasting_glucose']}|mmol/L|3.9|6.1",
                f"result|hba1c|{values['hba1c']}|%|4.0|6.0",
                f"result|ldl_cholesterol|{values['ldl_cholesterol']}|mmol/L|0.0|3.4",
            ]
            reports.append({
                "profile_id": f"synthetic-ocr-patient-{patient:02d}",
                "report_id": report_id,
                "split": "test" if patient in test_patients else "dev",
                "template": TEMPLATES[(patient * 2 + visit) % len(TEMPLATES)],
                "fields": fields,
                "source_lines": lines,
            })
    return reports


def render_report(report: dict[str, Any], image_path: Path, *, seed: int) -> None:
    """Render from Gold only; template variation never changes the source fields."""
    from PIL import Image, ImageDraw, ImageFilter, ImageFont

    template = report["template"]
    font_path = next((path for path in FONT_CANDIDATES if path.is_file()), None)
    if font_path is None:
        raise RuntimeError("No TrueType font found; supply a local Arial or DejaVuSans font.")
    font_size = 21 if template in {"compact", "faint"} else 25
    font = ImageFont.truetype(str(font_path), font_size)
    background = (245, 245, 245) if template == "faint" else (255, 255, 255)
    foreground = (115, 115, 115) if template == "faint" else (25, 25, 25)
    image = Image.new("RGB", (1300, 460), background)
    draw = ImageDraw.Draw(image)
    spacing = 48 if template == "compact" else 58
    for index, line in enumerate(report["source_lines"]):
        draw.text((55, 48 + index * spacing), line, fill=foreground, font=font)
    if template == "rotated":
        angle = random.Random(f"{seed}:{report['report_id']}").choice((-1.5, 1.5))
        image = image.rotate(angle, fillcolor=(255, 255, 255))
    if template == "faint":
        image = image.filter(ImageFilter.GaussianBlur(0.45))
    image_path.parent.mkdir(parents=True, exist_ok=True)
    if template == "jpeg":
        image.save(image_path, quality=74, subsampling=2)
    else:
        image.save(image_path)


def extract_fields(preview: dict[str, Any]) -> dict[str, Any]:
    fields: dict[str, Any] = {}
    for name in ("report_id", "report_date"):
        value = preview["report"].get(name)
        if value is not None:
            fields[name] = value
    for indicator in preview["indicators"]:
        fields[indicator["code"]] = {
            "value": str(indicator["value"]), "unit": indicator["unit"]
        }
    return fields


def _same_value(actual: str, expected: str) -> bool:
    try:
        return Decimal(actual) == Decimal(expected)
    except Exception:
        return False


def _block_for_field(blocks: list[dict[str, Any]], field: str, template: str) -> dict[str, Any] | None:
    prefix = f"result|{field}|" if field in INDICATORS else f"{field}:"
    exact = next((block for block in blocks if block["text"].startswith(prefix)), None)
    if exact is not None:
        return exact
    line_index = {"report_id": 0, "report_date": 1, "fasting_glucose": 3,
                  "hba1c": 4, "ldl_cholesterol": 5}.get(field)
    if line_index is None:
        return None
    spacing = 48 if template == "compact" else 58
    font_size = 21 if template in {"compact", "faint"} else 25
    expected_y = 48 + line_index * spacing + font_size / 2
    candidates = [
        (abs(sum(point[1] for point in block["polygon"]) / len(block["polygon"]) - expected_y), block)
        for block in blocks if block.get("polygon")
    ]
    if not candidates:
        return None
    distance, block = min(candidates, key=lambda pair: pair[0])
    return block if distance <= 24 else None


def score_case(gold: dict[str, Any], extracted: dict[str, Any], blocks: list[dict[str, Any]]) -> tuple[dict[str, int], list[dict[str, Any]]]:
    expected = gold["fields"]
    expected_keys, actual_keys = set(expected), set(extracted)
    counts = Counter(key_tp=len(expected_keys & actual_keys), key_fp=len(actual_keys - expected_keys), key_fn=len(expected_keys - actual_keys))
    counts.update(value_total=len(INDICATORS), unit_total=len(INDICATORS), date_total=1, field_total=len(expected))
    badcases = []
    for field in sorted(expected_keys | actual_keys):
        wanted, actual = expected.get(field), extracted.get(field)
        if field in INDICATORS:
            value_ok = isinstance(wanted, dict) and isinstance(actual, dict) and _same_value(actual.get("value", ""), wanted["value"])
            unit_ok = isinstance(wanted, dict) and isinstance(actual, dict) and actual.get("unit") == wanted["unit"]
            if field in expected_keys:
                counts["value_correct"] += int(value_ok)
                counts["unit_correct"] += int(unit_ok)
            correct = value_ok and unit_ok
        else:
            correct = field in expected_keys and actual == wanted
            if field == "report_date":
                counts["date_correct"] += int(correct)
        if correct:
            counts["tp"] += 1
            counts["indicator_tp"] += int(field in INDICATORS)
            continue
        counts["fp"] += int(field in actual_keys)
        counts["fn"] += int(field in expected_keys)
        if field in INDICATORS:
            counts["indicator_fp"] += int(field in actual_keys)
            counts["indicator_fn"] += int(field in expected_keys)
        counts["correction_fields"] += 1
        block = _block_for_field(blocks, field, gold["template"])
        low_confidence = block is not None and block["confidence"] < LOW_CONFIDENCE_THRESHOLD
        counts["low_confidence_corrections"] += int(low_confidence)
        badcases.append({
            "report_id": gold["report_id"], "profile_id": gold["profile_id"],
            "split": gold["split"], "template": gold["template"],
            "field": field, "kind": "missed" if actual is None else "incorrect",
            "gold": wanted, "extracted": actual,
            "ocr_block": block["text"] if block else None,
            "confidence": block["confidence"] if block else None,
            "low_confidence_flagged": low_confidence,
        })
    return dict(counts), badcases


def _ratio(numerator: int, denominator: int) -> float | None:
    return round(numerator / denominator, 6) if denominator else None


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    counts = Counter()
    for row in rows:
        counts.update(row["counts"])
    precision = _ratio(counts["tp"], counts["tp"] + counts["fp"])
    recall = _ratio(counts["tp"], counts["tp"] + counts["fn"])
    f1 = round(2 * precision * recall / (precision + recall), 6) if precision is not None and recall is not None and precision + recall else 0.0
    key_precision = _ratio(counts["key_tp"], counts["key_tp"] + counts["key_fp"])
    key_recall = _ratio(counts["key_tp"], counts["key_tp"] + counts["key_fn"])
    key_f1 = round(2 * key_precision * key_recall / (key_precision + key_recall), 6) if key_precision is not None and key_recall is not None and key_precision + key_recall else 0.0
    indicator_precision = _ratio(counts["indicator_tp"], counts["indicator_tp"] + counts["indicator_fp"])
    indicator_recall = _ratio(counts["indicator_tp"], counts["indicator_tp"] + counts["indicator_fn"])
    indicator_f1 = round(2 * indicator_precision * indicator_recall / (indicator_precision + indicator_recall), 6) if indicator_precision is not None and indicator_recall is not None and indicator_precision + indicator_recall else 0.0
    return {
        "report_count": len(rows), "gold_field_count": counts["field_total"],
        "field_micro": {"tp": counts["tp"], "fp": counts["fp"], "fn": counts["fn"], "precision": precision, "recall": recall, "f1": f1},
        "field_key_micro": {"tp": counts["key_tp"], "fp": counts["key_fp"], "fn": counts["key_fn"], "precision": key_precision, "recall": key_recall, "f1": key_f1},
        "indicator_field_micro": {"tp": counts["indicator_tp"], "fp": counts["indicator_fp"], "fn": counts["indicator_fn"], "precision": indicator_precision, "recall": indicator_recall, "f1": indicator_f1},
        "preview_ready": {"count": sum(row.get("preview_status") == "ready" for row in rows), "denominator": len(rows), "rate": _ratio(sum(row.get("preview_status") == "ready" for row in rows), len(rows))},
        "value_exact": {"correct": counts["value_correct"], "gold_denominator": counts["value_total"], "rate": _ratio(counts["value_correct"], counts["value_total"])},
        "unit_exact": {"correct": counts["unit_correct"], "gold_denominator": counts["unit_total"], "rate": _ratio(counts["unit_correct"], counts["unit_total"])},
        "date_exact": {"correct": counts["date_correct"], "gold_denominator": counts["date_total"], "rate": _ratio(counts["date_correct"], counts["date_total"])},
        "correction_burden": {"fields": counts["correction_fields"], "gold_denominator": counts["field_total"], "rate": _ratio(counts["correction_fields"], counts["field_total"])},
        "low_confidence_recall": {"flagged_corrections": counts["low_confidence_corrections"], "correction_denominator": counts["correction_fields"], "rate": _ratio(counts["low_confidence_corrections"], counts["correction_fields"])},
    }


def run(output: Path, seed: int, *, source_run: Path | None = None, replay_text: bool = False) -> dict[str, Any]:
    from PIL import __version__ as pillow_version
    import rapidocr

    if replay_text and source_run is None:
        raise ValueError("Saved text replay requires a frozen source run.")
    frozen_rows = {}
    if source_run is not None:
        source_run = source_run.resolve()
        source_manifest = json.loads((source_run / "manifest.json").read_text(encoding="utf-8"))
        source_gold = (source_run / "gold.json").read_bytes()
        if _sha256(source_gold) != source_manifest["gold_sha256"]:
            raise ValueError("Frozen OCR Gold hash changed.")
        gold = json.loads(source_gold)
        seed = source_manifest["seed"]
        frozen_rows = {r["report_id"]: r for r in json.loads((source_run / "results.json").read_text(encoding="utf-8"))}
        if set(frozen_rows) != {r["report_id"] for r in gold} or len(gold) != 20:
            raise ValueError("Frozen OCR report set changed.")
        for report in gold:
            row = frozen_rows[report["report_id"]]
            image_path = Path(row["image_path"])
            if not image_path.resolve().is_relative_to(source_run / "images"):
                raise ValueError("Frozen OCR image escaped its source directory.")
            if _sha256(image_path.read_bytes()) != source_manifest["image_hashes"][report["report_id"]] or row["image_sha256"] != source_manifest["image_hashes"][report["report_id"]]:
                raise ValueError("Frozen OCR image hash changed.")
    else:
        gold = make_gold(seed)
    output.mkdir(parents=True, exist_ok=False)
    _write_json(output / "gold.json", gold)
    pipeline = None if replay_text else OCRPipeline(output / "raw_ocr")
    rows, badcases = [], []
    for report in gold:
        extension = ".jpg" if report["template"] == "jpeg" else ".png"
        image_path = Path(frozen_rows[report["report_id"]]["image_path"]) if source_run else output / "images" / f"{report['report_id']}{extension}"
        if source_run is None:
            render_report(report, image_path, seed=seed)
        image_bytes = image_path.read_bytes()
        try:
            if replay_text:
                prior = frozen_rows[report["report_id"]]
                if prior["failure"] or not prior["ocr_json_path"]:
                    raise OCRProcessingError("Frozen OCR source failed.")
                artifact = json.loads(Path(prior["ocr_json_path"]).read_text(encoding="utf-8"))
                ocr = {"text": prior["ocr_text"], "blocks": artifact["blocks"], "engine": prior["engine"],
                       "artifact": {"json_path": prior["ocr_json_path"]}}
            else:
                ocr = pipeline.process(image_bytes, image_path.name)
            raw_text, blocks, engine, failure = ocr["text"], ocr["blocks"], ocr["engine"], None
            ocr_json_path = ocr["artifact"]["json_path"]
        except OCRProcessingError as exc:
            raw_text, blocks, engine, failure = "", [], None, str(exc)
            ocr_json_path = None
        preview = build_report_preview(f"image:{_sha256(image_bytes)}", raw_text).to_dict()
        extracted = extract_fields(preview)
        counts, case_badcases = score_case(report, extracted, blocks)
        rows.append({
            "report_id": report["report_id"], "profile_id": report["profile_id"],
            "split": report["split"], "template": report["template"],
            "image_path": str(image_path), "image_sha256": _sha256(image_bytes),
            "ocr_json_path": ocr_json_path, "ocr_text": raw_text,
            "engine": engine, "failure": failure,
            "preview_status": preview["status"], "preview_issues": preview["issues"],
            "preview": preview,
            "extracted_fields": extracted, "counts": counts,
        })
        badcases.extend(case_badcases)
    _write_json(output / "results.json", rows)
    _write_json(output / "badcases.json", badcases)
    summary = {
        "protocol": "task-078-ocr-replay-v1" if source_run else "task-046-ocr-eval-v2", "seed": seed,
        "mode": "saved_text_regression" if replay_text else "real_cpu_ocr",
        "untouched_held_out": False if source_run else None,
        "targets": {"field_f1": 0.92, "value_unit_date_exact": 0.95},
        "overall": summarize(rows),
        "by_split": {split: summarize([r for r in rows if r["split"] == split]) for split in ("dev", "test")},
        "by_template": {template: summarize([r for r in rows if r["template"] == template]) for template in TEMPLATES},
        "ocr_failures": sum(row["failure"] is not None for row in rows),
        "badcase_count": len(badcases),
    }
    _write_json(output / "summary.json", summary)
    _write_json(output / "manifest.json", {
        "protocol": summary["protocol"], "seed": seed,
        "run_command": f'python -B -m healthvault.evaluation.ocr.runner --output "{output}" --seed {seed}'
                       + (f' --source-run "{source_run}"' if source_run else "")
                       + (" --replay-text" if replay_text else ""),
        "patient_split": {r["profile_id"]: r["split"] for r in gold},
        "templates": list(TEMPLATES), "low_confidence_threshold": LOW_CONFIDENCE_THRESHOLD,
        "gold_sha256": _sha256((output / "gold.json").read_bytes()),
        "pillow_version": pillow_version,
        "engines": [json.loads(value) for value in sorted({json.dumps(r["engine"], sort_keys=True) for r in rows if r["engine"]})],
        "gold_source": "unchanged verified Gold from source_run" if source_run else "make_gold(seed); fields fixed before rendering and OCR",
        "normalization": {"numeric_value": "Decimal exact", "unit": "exact string", "date": "exact ISO string", "field_match": "key and complete value/unit or date/id exact"},
        "model_sha256": {
            path.name: _sha256(path.read_bytes())
            for path in sorted((Path(rapidocr.__file__).parent / "models").glob("*.onnx"))
        },
        "image_hashes": {r["report_id"]: r["image_sha256"] for r in rows},
        "source_run": str(source_run) if source_run else None,
        "source_bindings": {name: _sha256((source_run / name).read_bytes()) for name in ("manifest.json", "gold.json", "results.json")} if source_run else {},
        "mode": summary["mode"],
    })
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=46)
    parser.add_argument("--source-run", type=Path)
    parser.add_argument("--replay-text", action="store_true")
    args = parser.parse_args()
    print(json.dumps(run(args.output, args.seed, source_run=args.source_run, replay_text=args.replay_text), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
