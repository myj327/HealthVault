"""Build normalized source snapshots from downloaded official HTML.

Trafilatura is imported only by this build utility. Runtime code consumes committed
JSON snapshots and never imports the acquisition dependency or accesses the network.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator
from xml.etree import ElementTree


CORPUS_VERSION = "r1-2026-08-24-v1"
RETRIEVED_ON = "2026-08-24"
REUSE_BASIS = (
    "NIH website text; most NIH information is public domain and reusable unless "
    "otherwise marked. No images, logos, references, or third-party material are "
    "included. Policy: https://www.nih.gov/about-nih/frequently-asked-questions"
)
MEDLINEPLUS_REUSE_BASIS = (
    "MedlinePlus Medical test information is public domain. Attribution retained "
    "as required: Source: MedlinePlus, National Library of Medicine. No images, "
    "logos, references, or third-party material are included. Policy: "
    "https://medlineplus.gov/about/using/usingcontent/"
)
FEDERAL_REUSE_BASIS = (
    "U.S. federal government website text; public domain under 17 U.S.C. 105 "
    "unless otherwise marked. No images, logos, references, or third-party "
    "material are included."
)


@dataclass(frozen=True)
class SourceSpec:
    filename: str
    source_id: str
    publisher: str
    title: str
    version: str
    url: str
    title_paths: dict[tuple[str, ...], tuple[str, ...]]


def _paths(
    title: str,
    *entries: tuple[tuple[str, ...], tuple[str, ...]],
) -> dict[tuple[str, ...], tuple[str, ...]]:
    return {(title, *relative): tags for relative, tags in entries}


def _flat_paths(
    title: str,
    entries: tuple[tuple[str, tuple[str, ...]], ...],
) -> dict[tuple[str, ...], tuple[str, ...]]:
    return _paths(title, *(((heading,), tags) for heading, tags in entries))


SOURCE_SPECS = (
    SourceSpec(
        "niddk-tests.html", "niddk-diabetes-tests", "NIDDK",
        "Diabetes Tests & Diagnosis", "retrieved-2026-08-24",
        "https://www.niddk.nih.gov/health-information/diabetes/overview/tests-diagnosis",
        _paths(
            "Diabetes Tests & Diagnosis",
            (("Who should be tested for diabetes?",), ("blood_glucose", "visit_preparation")),
            (("Who should be tested for diabetes?", "Type 2 diabetes"), ("blood_glucose", "visit_preparation")),
            (("What tests are used to diagnose diabetes?",), ("blood_glucose", "fasting_glucose", "hba1c", "lab_preparation")),
            (("What tests are used to diagnose diabetes?", "Fasting plasma glucose test"), ("blood_glucose", "fasting_glucose", "lab_preparation")),
            (("What tests are used to diagnose diabetes?", "A1C test"), ("blood_glucose", "hba1c", "lab_preparation")),
            (("What test results tell me if I have diabetes or prediabetes?",), ("blood_glucose", "hba1c", "result_communication")),
            (("What test results tell me if I have diabetes or prediabetes?", "Test results for diagnosis of prediabetes and diabetes"), ("blood_glucose", "fasting_glucose", "hba1c", "result_communication")),
        ),
    ),
    SourceSpec(
        "niddk-a1c.html", "niddk-a1c-test", "NIDDK",
        "The A1C Test & Diabetes", "retrieved-2026-08-24",
        "https://www.niddk.nih.gov/health-information/diagnostic-tests/a1c-test",
        _paths(
            "The A1C Test & Diabetes",
            (("What is the A1C test?",), ("hba1c", "blood_glucose")),
            (("Why should a person get the A1C test?",), ("hba1c", "visit_preparation")),
            (("How is the A1C test used to diagnose type 2 diabetes and prediabetes?",), ("hba1c", "result_communication")),
            (("How is the A1C test used to diagnose type 2 diabetes and prediabetes?", "A1C results and what the numbers mean"), ("hba1c", "result_communication")),
            (("Can other blood glucose tests be used to diagnose type 2 diabetes and prediabetes?",), ("hba1c", "fasting_glucose", "result_communication")),
            (("Can the A1C test result in a different diagnosis than the blood glucose tests?",), ("hba1c", "result_communication")),
            (("Why do diabetes blood test results vary?",), ("hba1c", "blood_glucose", "result_communication")),
            (("Why do diabetes blood test results vary?", "Blood glucose levels move up and down"), ("hba1c", "blood_glucose", "result_communication")),
            (("Why do diabetes blood test results vary?", "Blood glucose levels move up and down", "Blood Glucose Measurements Compared with A1C Measurements over 4 Days"), ("hba1c", "blood_glucose", "result_communication")),
            (("Why do diabetes blood test results vary?", "A1C tests can be affected by changes in red blood cells or hemoglobin"), ("hba1c", "result_communication")),
            (("Why do diabetes blood test results vary?", "Small changes in temperature, equipment, or sample handling"), ("hba1c", "lab_preparation", "result_communication")),
            (("How precise is the A1C test?",), ("hba1c", "result_communication")),
            (("How does A1C relate to estimated average glucose?",), ("hba1c", "blood_glucose", "result_communication")),
            (("Will the A1C test show short-term changes in blood glucose levels?",), ("hba1c", "blood_glucose", "result_communication")),
        ),
    ),
    SourceSpec(
        "medline-glucose.html", "medlineplus-blood-glucose-test", "MedlinePlus",
        "Blood Glucose Test", "reviewed-2024",
        "https://medlineplus.gov/lab-tests/blood-glucose-test/",
        _flat_paths("Blood Glucose Test", (
            ("What is a blood glucose test?", ("blood_glucose", "fasting_glucose")),
            ("What is it used for?", ("blood_glucose",)),
            ("Why do I need a blood glucose test?", ("blood_glucose", "visit_preparation")),
            ("What happens during a blood glucose test?", ("blood_glucose", "lab_preparation")),
            ("Will I need to do anything to prepare for the test?", ("fasting_glucose", "lab_preparation")),
            ("What do the results mean?", ("blood_glucose", "result_communication")),
            ("Is there anything else I should know about a blood glucose test?", ("blood_glucose", "result_communication")),
        )),
    ),
    SourceSpec(
        "medline-a1c.html", "medlineplus-hba1c-test", "MedlinePlus",
        "Hemoglobin A1C (HbA1c) Test", "reviewed-2025",
        "https://medlineplus.gov/lab-tests/hemoglobin-a1c-hba1c-test/",
        _flat_paths("Hemoglobin A1C (HbA1c) Test", (
            ("What is a hemoglobin A1C (HbA1C) test?", ("hba1c", "blood_glucose")),
            ("What is it used for?", ("hba1c",)),
            ("Why do I need an A1C test?", ("hba1c", "visit_preparation")),
            ("What happens during an A1C test?", ("hba1c", "lab_preparation")),
            ("What do the results mean?", ("hba1c", "result_communication")),
            ("Is there anything else I need to know about an A1C test?", ("hba1c", "result_communication")),
        )),
    ),
    SourceSpec(
        "medline-fasting.html", "medlineplus-fasting-blood-test", "MedlinePlus",
        "Fasting for a Blood Test", "reviewed-2025",
        "https://medlineplus.gov/lab-tests/fasting-for-a-blood-test/",
        _flat_paths("Fasting for a Blood Test", (
            ("What is fasting for a blood test?", ("fasting_glucose", "lab_preparation")),
            ("How long do I have to fast before the test?", ("fasting_glucose", "lab_preparation")),
            ("Why do I need to fast before my blood test?", ("fasting_glucose", "lab_preparation")),
            ("What types of blood tests require fasting?", ("fasting_glucose", "ldl_cholesterol", "lab_preparation")),
            ("Can I drink anything besides water during a fast?", ("lab_preparation",)),
            ("What if I make a mistake and have something to eat or drink besides water during my fast?", ("lab_preparation", "result_communication")),
            ("When can I eat and drink normally again?", ("lab_preparation",)),
            ("Is there anything else I need to know about fasting before a blood test?", ("lab_preparation",)),
        )),
    ),
    SourceSpec(
        "medline-prepare.html", "medlineplus-lab-preparation", "MedlinePlus",
        "How to Prepare for a Lab Test", "reviewed-2024",
        "https://medlineplus.gov/lab-tests/how-to-prepare-for-a-lab-test/",
        _flat_paths("How to Prepare for a Lab Test", (
            ("How do I prepare for a lab test?", ("lab_preparation",)),
            ("Will I need to take other steps to prepare for my lab test?", ("lab_preparation",)),
            ("What types of lab tests require special preparation?", ("lab_preparation",)),
            ("Is there anything else I should know about preparing for a lab test?", ("lab_preparation", "result_communication")),
        )),
    ),
    SourceSpec(
        "medline-cholesterol.html", "medlineplus-cholesterol-levels", "MedlinePlus",
        "Cholesterol Levels", "reviewed-2024",
        "https://medlineplus.gov/lab-tests/cholesterol-levels/",
        _flat_paths("Cholesterol Levels", (
            ("What is a cholesterol test?", ("ldl_cholesterol",)),
            ("What is it used for?", ("ldl_cholesterol",)),
            ("Why do I need a cholesterol test?", ("ldl_cholesterol", "visit_preparation")),
            ("What happens during a cholesterol test?", ("ldl_cholesterol", "lab_preparation")),
            ("Will I need to do anything to prepare for the test?", ("ldl_cholesterol", "lab_preparation")),
            ("What do the results mean?", ("ldl_cholesterol", "result_communication")),
            ("Is there anything else I need to know about my cholesterol levels?", ("ldl_cholesterol", "result_communication")),
        )),
    ),
    SourceSpec(
        "nhlbi-cholesterol.html", "nhlbi-cholesterol-diagnosis", "NHLBI",
        "Blood Cholesterol Diagnosis", "updated-2024-04-18",
        "https://www.nhlbi.nih.gov/health/blood-cholesterol/diagnosis",
        _flat_paths("Blood Cholesterol Diagnosis", (
            ("Medical history and physical exam", ("ldl_cholesterol", "visit_preparation")),
            ("Screening for high cholesterol", ("ldl_cholesterol", "lab_preparation")),
            ("Lipoprotein (lipid) panel", ("ldl_cholesterol", "result_communication")),
            ("Lipoprotein-a", ("ldl_cholesterol", "result_communication")),
        )),
    ),
    SourceSpec(
        "medline-results.html", "medlineplus-understand-lab-results", "MedlinePlus",
        "How to Understand Your Lab Results", "reviewed-2025",
        "https://medlineplus.gov/lab-tests/how-to-understand-your-lab-results/",
        _flat_paths("How to Understand Your Lab Results", (
            ("What is a laboratory test?", ("result_communication",)),
            ("Why do I need a lab test?", ("result_communication", "visit_preparation")),
            ("What is a reference range?", ("result_communication",)),
            ("What do negative, positive, and inconclusive results mean?", ("result_communication",)),
            ("What are false positive and false negative results?", ("result_communication",)),
            ("What can affect my results?", ("result_communication", "lab_preparation")),
            ("What if I do a home test?", ("result_communication",)),
        )),
    ),
    SourceSpec(
        "nhlbi-blood-tests.html", "nhlbi-blood-tests", "NHLBI", "Blood Tests",
        "updated-2022-03-24", "https://www.nhlbi.nih.gov/health/blood-tests",
        _paths(
            "Blood Tests",
            (("Types of blood tests", "Blood chemistry tests/basic metabolic panel"), ("blood_glucose", "fasting_glucose", "lab_preparation")),
            (("Types of blood tests", "Lipoprotein panel"), ("ldl_cholesterol", "lab_preparation", "result_communication")),
        ),
    ),
    SourceSpec(
        "niddk-professional-tests.html", "niddk-diabetes-prediabetes-tests", "NIDDK",
        "Diabetes & Prediabetes Tests", "retrieved-2026-08-24",
        "https://www.niddk.nih.gov/health-information/professionals/clinical-tools-patient-management/diabetes/diabetes-prediabetes",
        _paths(
            "Diabetes & Prediabetes Tests",
            (("Confirming diagnosis of type 2 diabetes and prediabetes",), ("blood_glucose", "hba1c", "result_communication")),
            (("Interpreting laboratory results",), ("result_communication",)),
            (("Comparing diabetes blood tests", "Fasting plasma glucose (FPG) test"), ("blood_glucose", "fasting_glucose", "lab_preparation", "result_communication")),
            (("Comparing diabetes blood tests", "A1C test"), ("blood_glucose", "hba1c", "result_communication")),
            (("Comparing diagnoses",), ("blood_glucose", "fasting_glucose", "hba1c", "result_communication")),
        ),
    ),
    SourceSpec(
        "medline-doctor.html", "medlineplus-talking-with-doctor", "MedlinePlus",
        "Talking With Your Doctor", "reviewed-2024",
        "https://medlineplus.gov/talkingwithyourdoctor.html",
        _flat_paths("Talking With Your Doctor", (
            ("Summary", ("visit_preparation", "result_communication")),
        )),
    ),
    SourceSpec(
        "odphp-get-screened.html", "odphp-get-screened", "ODPHP", "Get Screened",
        "updated-2026-03-31",
        "https://odphp.health.gov/myhealthfinder/doctor-visits/screening-tests/get-screened",
        _paths(
            "Get Screened",
            (("Take Action", "Get Ready", "Make a list of questions to ask your doctor."), ("visit_preparation",)),
            (("Take Action", "See Your Doctor", "Talk with your doctor about getting screened."), ("visit_preparation",)),
            (("Take Action", "See Your Doctor", "Make sure you get the results from every screening."), ("result_communication",)),
        ),
    ),
    SourceSpec(
        "odphp-take-charge.html", "odphp-take-charge-health-care", "ODPHP",
        "Take Charge of Your Health Care", "updated-2026-03-31",
        "https://odphp.health.gov/myhealthfinder/doctor-visits/talking-doctor/take-charge-your-health-care",
        _paths(
            "Take Charge of Your Health Care",
            (("The Basics", "Overview"), ("visit_preparation", "result_communication")),
            (("Take Action", "Keep a Record", "Keep track of important health information."), ("visit_preparation", "result_communication")),
            (("Take Action", "Be Prepared", "Make the most of doctor visits."), ("visit_preparation",)),
            (("Take Action", "Be Prepared", "Follow up after your appointment."), ("visit_preparation", "result_communication")),
        ),
    ),
)


_PRUNE_XPATH = (
    "//nav", "//footer", "//*[contains(@class, 'usa-modal')]",
    "//*[contains(@class, 'book-navigation')]",
    "//*[contains(@class, 'pdf-prompt')]", "//*[contains(@class, 'pager')]",
)
_UI_SNIPPETS = ("book traversal links", "view the content below as", "pdf version")
_MEDICATION_RE = re.compile(
    r"\b(?:treatments?|medicines?|medications?|drugs?|statins?|warfarin|heparin)\b",
    re.IGNORECASE,
)


def _clean(value: str) -> str:
    return " ".join(value.replace("\xa0", " ").split())


def _heading_title(value: str, source_title: str) -> str:
    cleaned = _clean(value)
    duplicated = f"{source_title} {source_title}"
    return source_title if cleaned.casefold() == duplicated.casefold() else cleaned


def _iter_structural_elements(element: ElementTree.Element) -> Iterator[ElementTree.Element]:
    for child in element:
        tag = child.tag.rsplit("}", 1)[-1]
        if tag in {"head", "p", "list", "table"}:
            yield child
        else:
            yield from _iter_structural_elements(child)


def _excluded_text(text: str) -> bool:
    folded = text.casefold()
    return (
        any(snippet in folded for snippet in _UI_SNIPPETS)
        or folded in {"previous", "next", "previous next"}
        or _MEDICATION_RE.search(text) is not None
    )


def _append_text_block(section: dict[str, Any], kind: str, text: str) -> None:
    cleaned = _clean(text)
    if cleaned and not _excluded_text(cleaned):
        section["blocks"].append({"kind": kind, "text": cleaned})


def normalize_official_html(spec: SourceSpec, html: str) -> dict[str, Any]:
    """Extract main content, then admit only exact real-heading whitelist paths."""
    try:
        from trafilatura import extract
    except ImportError as error:  # pragma: no cover - environment contract
        raise RuntimeError(
            "Install requirements-rag-build.txt before corpus acquisition"
        ) from error

    extracted = extract(
        html, output_format="xml", include_comments=False, include_tables=True,
        include_images=False, include_links=False, with_metadata=False,
        favor_precision=True, deduplicate=False, prune_xpath=list(_PRUNE_XPATH),
    )
    if not extracted:
        raise ValueError(f"Trafilatura found no main content in {spec.filename}")
    root = ElementTree.fromstring(extracted)
    main = root.find("main")
    if main is None:
        raise ValueError(f"Trafilatura returned no main tree for {spec.filename}")

    heading_stack: dict[int, str] = {1: spec.title}
    current: dict[str, Any] | None = None
    sections: dict[tuple[str, ...], dict[str, Any]] = {}
    for element in _iter_structural_elements(main):
        tag = element.tag.rsplit("}", 1)[-1]
        if tag == "head":
            match = re.fullmatch(r"#?h([1-4])", element.get("rend", ""))
            if match is None:
                current = None
                continue
            level = int(match.group(1))
            title = _heading_title("".join(element.itertext()), spec.title)
            heading_stack = {
                existing: value for existing, value in heading_stack.items()
                if existing < level
            }
            if level == 1:
                if title.casefold() != spec.title.casefold():
                    raise ValueError(f"Official H1 does not match source title in {spec.filename}")
                heading_stack[1] = spec.title
                current = None
                continue
            heading_stack.setdefault(1, spec.title)
            heading_stack[level] = title
            title_path = tuple(heading_stack[item] for item in sorted(heading_stack))
            tags = spec.title_paths.get(title_path)
            if tags is None:
                current = None
                continue
            if title_path in sections:
                raise ValueError(f"Duplicate selected heading in {spec.filename}")
            current = {
                "title_path": list(title_path),
                "topic_tags": list(tags),
                "blocks": [],
            }
            sections[title_path] = current
            continue

        if current is None:
            continue
        if tag == "p":
            _append_text_block(current, "paragraph", "".join(element.itertext()))
        elif tag == "list":
            for item in element.iter("item"):
                _append_text_block(current, "list", "".join(item.itertext()))
        elif tag == "table":
            rows = []
            for row in element.iter("row"):
                cells = [_clean("".join(cell.itertext())) for cell in row.iter("cell")]
                cells = [cell for cell in cells if cell]
                if cells:
                    rows.append(" | ".join(cells))
            if rows:
                header = rows[0]
                for row in rows[1:]:
                    if not _excluded_text(row):
                        current["blocks"].append(
                            {"kind": "table", "text": row, "table_header": header}
                        )

    expected_paths = tuple(spec.title_paths)
    if tuple(sections) != expected_paths:
        missing = [" / ".join(path) for path in expected_paths if path not in sections]
        unexpected = [" / ".join(path) for path in sections if path not in spec.title_paths]
        raise ValueError(
            f"Selected heading mismatch in {spec.filename}; "
            f"missing={missing}, unexpected={unexpected}"
        )
    ordered_sections = [sections[path] for path in expected_paths]
    empty = [
        " / ".join(section["title_path"])
        for section in ordered_sections if not section["blocks"]
    ]
    if empty:
        raise ValueError(f"Selected headings contain no usable content: {empty}")
    content_bytes = json.dumps(
        ordered_sections, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return {
        "schema_version": 1,
        "corpus_version": CORPUS_VERSION,
        "source_id": spec.source_id,
        "publisher": spec.publisher,
        "title": spec.title,
        "version": spec.version,
        "url": spec.url,
        "language": "en",
        "retrieved_on": RETRIEVED_ON,
        "reuse_basis": (
            MEDLINEPLUS_REUSE_BASIS
            if spec.publisher == "MedlinePlus"
            else FEDERAL_REUSE_BASIS
            if spec.publisher == "ODPHP"
            else REUSE_BASIS
        ),
        "source_sha256": hashlib.sha256(content_bytes).hexdigest(),
        "title_path_whitelist": [list(path) for path in expected_paths],
        "sections": ordered_sections,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--html-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--source-id", action="append", default=[])
    args = parser.parse_args(argv)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    requested = set(args.source_id)
    known = {spec.source_id for spec in SOURCE_SPECS}
    if requested - known:
        raise ValueError("Unknown source_id requested")
    selected_specs = tuple(
        spec for spec in SOURCE_SPECS if not requested or spec.source_id in requested
    )
    for spec in selected_specs:
        snapshot = normalize_official_html(
            spec, (args.html_dir / spec.filename).read_text(encoding="utf-8")
        )
        output_path = args.output_dir / f"{spec.source_id}.json"
        output_path.write_text(
            json.dumps(snapshot, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    if not requested:
        selected_ids = {spec.source_id for spec in selected_specs}
        for path in args.output_dir.glob("*.json"):
            if path.stem not in selected_ids:
                path.unlink()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
