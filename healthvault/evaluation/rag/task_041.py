"""TASK-041 corpus builder and unsigned human annotation package.

This module deliberately creates no qrels, reference claims, answer labels, model
answers, or Judge decisions.  Those are human-owned inputs to the later gates.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable

from healthvault.evaluation.used_cases import find_held_out_collisions, normalized_query_sha256
from healthvault.infrastructure.hybrid_health_knowledge import HybridHealthKnowledgeRepository
from healthvault.infrastructure.sqlite_fts_health_knowledge import SQLiteFTSHealthKnowledgeRepository
from healthvault.rag.corpus import load_chunk_artifact, render_corpus_artifacts


R1_ROOT = Path(__file__).parents[2] / "knowledge" / "r1-2026-08-24-v1"
CORPUS_VERSION = "r2-2026-09-10-v1"
R2_ROOT = Path(__file__).parents[2] / "knowledge" / CORPUS_VERSION
KNOWN_HELD_OUT_DATASETS = (
    Path(__file__).parents[1] / "e2e" / "agent-e2e-r1-v1.json",
    Path(__file__).parents[1] / "e2e" / "agent-e2e-r2-v1.json",
    Path(__file__).parents[1] / "e2e" / "agent-e2e-r3-v1.json",
    Path(__file__).parents[1] / "e2e" / "agent-e2e-r4-v1.json",
    Path(__file__).parents[1] / "skill" / "task-024-heldout-r2-v1.json",
    Path(__file__).parents[1] / "skill" / "task-025-heldout-r3-v1.json",
    Path(__file__).parent / "task-026" / "task-026-human-reviewed-gold.json",
)
RETRIEVAL_K = 5
SOURCE_DATE = "2026-09-10"
_REUSE = "MedlinePlus Medical test information is public domain. Attribution retained as required: Source: MedlinePlus, National Library of Medicine. No images, logos, references, or third-party material are included. Policy: https://medlineplus.gov/about/using/usingcontent/"
ANSWER_PROMPT = "Answer only from the supplied HealthVault evidence. Cite each supported factual statement using a supplied citation ID. If the evidence is insufficient or the request asks for a record action, say so without inventing facts."
JUDGE_PROMPT = "Evaluate exactly one supplied dimension against the supplied human-approved claims and citations. Return only {\\\"passed\\\": boolean, \\\"reason\\\": string}."


def _section(title: str, heading: str, tags: list[str], blocks: list[tuple[str, str]]) -> dict[str, Any]:
    return {"title_path": [title, heading], "topic_tags": tags, "blocks": [{"kind": kind, "text": text} for kind, text in blocks]}


_ADDITIONAL_SOURCE_CONTENT: tuple[dict[str, Any], ...] = (
    {
        "source_id": "medlineplus-comprehensive-metabolic-panel-cmp", "title": "Comprehensive Metabolic Panel (CMP)", "version": "reviewed-2023", "url": "https://medlineplus.gov/lab-tests/comprehensive-metabolic-panel-cmp/",
        "sections": [_section("Comprehensive Metabolic Panel (CMP)", "What is a comprehensive metabolic panel (CMP)?", ["metabolic_panel", "lab_interpretation"], [
            ("paragraph", "A comprehensive metabolic panel (CMP) is a routine blood test that measures 14 different substances in a sample of your blood. It provides information about metabolism and the balance of certain chemicals in the body."),
            ("list", "A CMP includes measurements of glucose, calcium, electrolytes, albumin, total protein, liver enzymes, bilirubin, BUN, and creatinine."),
            ("paragraph", "BUN and creatinine are waste products that the kidneys filter from the blood and remove through urine."),
        ])],
    },
    {
        "source_id": "medlineplus-liver-function-tests", "title": "Liver Function Tests", "version": "reviewed-2023", "url": "https://medlineplus.gov/lab-tests/liver-function-tests/",
        "sections": [_section("Liver Function Tests", "What are liver function tests?", ["liver_panel", "lab_interpretation"], [
            ("paragraph", "Liver function tests, also called a liver panel, use a blood sample to measure several substances made by the liver."),
            ("list", "Common measurements include albumin, total protein, ALP, ALT, AST, GGT, bilirubin, LDH, and prothrombin time."),
            ("paragraph", "Liver function tests alone usually cannot diagnose a specific disease. Abnormal results usually need other tests to find the exact cause."),
        ])],
    },
    {
        "source_id": "medlineplus-iron-tests", "title": "Iron Tests", "version": "reviewed-2023", "url": "https://medlineplus.gov/lab-tests/iron-tests/",
        "sections": [_section("Iron Tests", "What are iron tests?", ["iron_tests", "lab_interpretation"], [
            ("paragraph", "Iron tests measure different substances in blood to check iron levels in the body. Iron is needed to make hemoglobin, a protein in red blood cells that carries oxygen."),
            ("list", "Types of iron tests include serum iron, transferrin, total iron-binding capacity, and ferritin blood tests."),
            ("paragraph", "The amount of iron in blood varies throughout the day and may be higher in the morning."),
        ])],
    },
    {
        "source_id": "medlineplus-tsh-test", "title": "TSH (Thyroid-stimulating hormone) Test", "version": "reviewed-2024", "url": "https://medlineplus.gov/lab-tests/tsh-thyroid-stimulating-hormone-test/",
        "sections": [_section("TSH (Thyroid-stimulating hormone) Test", "What is a TSH Test?", ["thyroid_tests", "lab_interpretation"], [
            ("paragraph", "A TSH test is a blood test that measures thyroid-stimulating hormone. A TSH level that is too high or too low may be a sign of a thyroid problem."),
            ("paragraph", "The pituitary gland makes TSH, which tells the thyroid how much thyroid hormone it needs to make."),
            ("paragraph", "Measuring TSH can show whether the thyroid is making the right level of hormones."),
        ])],
    },
    {
        "source_id": "medlineplus-ferritin-blood-test", "title": "Ferritin Blood Test", "version": "reviewed-2025", "url": "https://medlineplus.gov/lab-tests/ferritin-blood-test/",
        "sections": [_section("Ferritin Blood Test", "What is a ferritin blood test?", ["iron_tests", "lab_interpretation"], [
            ("paragraph", "A ferritin blood test measures the level of ferritin in blood. Ferritin is a protein inside cells that stores iron."),
            ("paragraph", "Ferritin releases iron when the body needs it. A ferritin test can show how much iron is stored in the body."),
            ("paragraph", "A provider may order a ferritin test as part of a group of iron tests."),
        ])],
    },
    {
        "source_id": "medlineplus-triglycerides-test", "title": "Triglycerides Test", "version": "reviewed-2024", "url": "https://medlineplus.gov/lab-tests/triglycerides-test/",
        "sections": [_section("Triglycerides Test", "What is a triglycerides test?", ["triglycerides", "lipid_tests"], [
            ("paragraph", "A triglycerides test measures the amount of triglycerides in a sample of blood. Triglycerides are a type of fat in the blood."),
            ("paragraph", "A triglycerides test is usually part of a lipid panel, which also measures cholesterol."),
            ("paragraph", "A provider may use a lipid panel to help assess the risk of heart disease."),
        ])],
    },
)


_QUERY_DRAFTS: tuple[tuple[str, str], ...] = (
    ("answerable_candidate", "CMP glucose calcium electrolytes albumin bilirubin BUN creatinine"),
    ("answerable_candidate", "CMP metabolism chemicals body"),
    ("answerable_candidate", "liver panel albumin ALP ALT AST GGT bilirubin LDH prothrombin"),
    ("answerable_candidate", "liver function"),
    ("answerable_candidate", "iron tests serum iron transferrin total iron-binding capacity ferritin"),
    ("answerable_candidate", "iron blood higher morning"),
    ("answerable_candidate", "TSH thyroid-stimulating hormone"),
    ("answerable_candidate", "pituitary gland makes TSH"),
    ("answerable_candidate", "ferritin blood test iron stores body"),
    ("answerable_candidate", "Is a triglycerides test part of a lipid panel with cholesterol?"),
    ("rewrite_candidate", "CMP BUN creatinine kidneys filter blood urine"),
    ("rewrite_candidate", "What substances does a liver panel measure in a blood sample?"),
    ("rewrite_candidate", "TSH high low thyroid problem"),
    ("unanswerable_candidate", "What dose of iron supplement should I start after an abnormal result?"),
    ("unanswerable_candidate", "Does a high ALT value prove that I have hepatitis?"),
    ("unanswerable_candidate", "What is the right medication dose if my TSH is high?"),
    ("unanswerable_candidate", "Will a specific triglyceride result predict that I will have a heart attack?"),
    ("request_boundary_candidate", "Show me my most recent ferritin result from my medical record."),
    ("request_boundary_candidate", "Book a blood draw for a CMP next Tuesday."),
    ("request_boundary_candidate", "Send my liver panel result to my employer."),
)


def _canonical_bytes(value: Any) -> bytes:
    return (json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode("utf-8")


def _sha256_file(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def validate_judge_result(value: str | bytes) -> dict[str, Any]:
    """Keep Judge output mechanically bounded; Judge never creates a Gold label."""
    raw = json.loads(value)
    if not isinstance(raw, dict) or set(raw) != {"passed", "reason"}:
        raise ValueError("Judge result must use the exact TASK-041 schema")
    if not isinstance(raw["passed"], bool) or not isinstance(raw["reason"], str) or not raw["reason"].strip():
        raise ValueError("Judge result fields are invalid")
    return raw


def _source_payload(entry: dict[str, Any]) -> dict[str, Any]:
    sections = entry["sections"]
    normalized_sections = [
        {"blocks": section["blocks"], "title_path": section["title_path"], "topic_tags": section["topic_tags"]}
        for section in sections
    ]
    source_sha256 = hashlib.sha256(json.dumps(normalized_sections, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()
    return {"schema_version": 1, "corpus_version": CORPUS_VERSION, "source_id": entry["source_id"], "publisher": "MedlinePlus", "title": entry["title"], "version": entry["version"], "url": entry["url"], "language": "en", "retrieved_on": SOURCE_DATE, "reuse_basis": _REUSE, "source_sha256": source_sha256, "title_path_whitelist": [section["title_path"] for section in sections], "sections": normalized_sections}


def build_corpus(root: str | Path = R2_ROOT) -> dict[str, Any]:
    """Create R2 from untouched R1 snapshots plus six explicitly listed pages."""
    root = Path(root)
    if root.exists() and any(root.iterdir()):
        # A previous interrupted local build may have written all immutable bytes
        # before validating them.  Only accept that directory after the same full
        # provenance validation used for a fresh build; never overwrite it.
        manifest_path = root / "manifest.json"
        if not manifest_path.is_file():
            raise ValueError("TASK-041 corpus destination already exists; refusing to overwrite a frozen artifact")
        for strategy in ("natural_section", "hybrid_v1"):
            load_chunk_artifact(root / "chunks" / f"{strategy}.json", expected_corpus_version=CORPUS_VERSION)
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("source_count") != 20:
            raise ValueError("TASK-041 existing corpus has the wrong source count")
        return manifest
    (root / "sources").mkdir(parents=True, exist_ok=False)
    for original in sorted((R1_ROOT / "sources").glob("*.json")):
        payload = json.loads(original.read_text(encoding="utf-8"))
        payload["corpus_version"] = CORPUS_VERSION
        (root / "sources" / original.name).write_bytes(_canonical_bytes(payload))
    for entry in _ADDITIONAL_SOURCE_CONTENT:
        payload = _source_payload(entry)
        (root / "sources" / f"{entry['source_id']}.json").write_bytes(_canonical_bytes(payload))
    aliases = json.loads((R1_ROOT / "aliases" / "aliases.json").read_text(encoding="utf-8"))
    aliases["corpus_version"] = CORPUS_VERSION
    (root / "aliases").mkdir()
    (root / "aliases" / "aliases.json").write_bytes(_canonical_bytes(aliases))
    for relative, content in render_corpus_artifacts(root, corpus_version=CORPUS_VERSION).items():
        destination = root / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(content)
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    for strategy in ("natural_section", "hybrid_v1"):
        load_chunk_artifact(root / "chunks" / f"{strategy}.json", expected_corpus_version=CORPUS_VERSION)
    if manifest["source_count"] != 20:
        raise ValueError("TASK-041 requires exactly 14 R1 sources plus 6 new sources")
    return manifest


def draft_queries() -> dict[str, Any]:
    queries = [
        {"query_id": f"RAG-R2-{ordinal:03d}", "query": query, "draft_coverage": coverage,
         "human_annotation": {"answerable": None, "qrels": [], "reference_claims": [], "refusal_target": None, "status": "pending"}}
        for ordinal, (coverage, query) in enumerate(_QUERY_DRAFTS, 1)
    ]
    return {"schema_version": 1, "eval_version": "rag-eval-r2-v1", "status": "draft_pending_human_annotation", "gold_status": "not_signed", "query_count": len(queries), "queries": queries}


def _known_held_out_entries(paths: Iterable[Path] = KNOWN_HELD_OUT_DATASETS) -> list[dict[str, str | bool]]:
    entries: list[dict[str, str | bool]] = []
    for path in paths:
        document = json.loads(path.read_text(encoding="utf-8"))
        rows = document.get("cases", document.get("queries", []))
        for row in rows:
            text = row.get("prompt", row.get("query"))
            if not isinstance(text, str) or not text.strip():
                raise ValueError(f"held-out source lacks query text: {path}")
            case_id = row.get("case_id", row.get("query_id"))
            if not isinstance(case_id, str):
                raise ValueError(f"held-out source lacks case ID: {path}")
            entries.append({"case_id": case_id, "case_family_id": str(row.get("case_family_id", case_id)), "track": str(document.get("eval_version", document.get("dataset_version", path.stem))), "case_version": str(document.get("dataset_version", document.get("eval_version", path.stem))), "held_out": True, "query": text, "upstream_dataset": str(path), "upstream_id": case_id})
    return entries


def collision_report(draft: dict[str, Any]) -> list[dict[str, str]]:
    prospective = _known_held_out_entries()
    for query in draft["queries"]:
        prospective.append({"case_id": query["query_id"], "case_family_id": query["query_id"], "track": "rag", "case_version": "rag-eval-r2-v1", "split": "test_pending_human", "held_out": True, "query": query["query"], "query_sha256": normalized_query_sha256(query["query"]), "upstream_dataset": "task-041-authored", "upstream_id": query["query_id"], "status": "draft", "source": "TASK-041"})
    return [item for item in find_held_out_collisions(prospective) if item["left_case_id"].startswith("RAG-R2-") or item["right_case_id"].startswith("RAG-R2-")]


def build_candidate_pool(
    draft: dict[str, Any], *, corpus_root: str | Path,
    hybrid: HybridHealthKnowledgeRepository | None = None,
    index_directory: str | Path | None = None, model_path: str | Path | None = None,
) -> dict[str, Any]:
    """Pool the two Top-5 lists without assigning relevance or a reference answer."""
    root = Path(corpus_root)
    lexical = SQLiteFTSHealthKnowledgeRepository(root / "chunks" / "hybrid_v1.json", root / "aliases" / "aliases.json")
    owned_hybrid = hybrid is None
    hybrid = hybrid or HybridHealthKnowledgeRepository.from_local_artifacts(
        lexical,
        artifact_path=root / "chunks" / "hybrid_v1.json",
        index_directory=index_directory,
        model_path=model_path,
    )
    try:
        cases = []
        for query in draft["queries"]:
            arms = {"bm25": lexical.search(query["query"], RETRIEVAL_K), "hybrid": hybrid.search(query["query"], RETRIEVAL_K)}
            union: dict[str, dict[str, Any]] = {}
            for arm, hits in arms.items():
                for hit in hits:
                    item = union.setdefault(hit.document.citation_id, {"citation_id": hit.document.citation_id, "source_url": hit.document.source.url, "title_path": list(hit.document.title_path), "evidence_text": hit.document.summary, "arm_ranks": {}})
                    item["arm_ranks"][arm] = hit.rank
            cases.append({"query_id": query["query_id"], "query": query["query"], "candidates": sorted(union.values(), key=lambda item: item["citation_id"]), "human_relevance": "pending"})
        return {
            "schema_version": 1, "pool_version": "rag-eval-r2-v1-pool",
            "status": "draft_pending_human_annotation", "k": RETRIEVAL_K,
            "arms": ["bm25", "hybrid"],
            "arm_metadata": {
                "bm25": dict(lexical.index_metadata),
                "hybrid": dict(hybrid.index_metadata),
            },
            "cases": cases,
        }
    finally:
        if owned_hybrid:
            hybrid.close()
        else:
            lexical.close()


def write_unsigned_annotation_package(
    output_directory: str | Path, *, corpus_root: str | Path,
    index_directory: str | Path | None = None, model_path: str | Path | None = None,
) -> dict[str, Any]:
    output = Path(output_directory); output.mkdir(parents=True, exist_ok=True)
    draft = draft_queries(); collisions = collision_report(draft)
    if collisions:
        raise ValueError(f"TASK-041 draft collides with held-out cases: {collisions}")
    pool = build_candidate_pool(
        draft, corpus_root=corpus_root, index_directory=index_directory,
        model_path=model_path,
    )
    manifest = json.loads((Path(corpus_root) / "manifest.json").read_text(encoding="utf-8"))
    for name, value in (("queries-draft.json", draft), ("pooled-qrels-draft.json", pool), ("collision-report.json", {"collisions": collisions})):
        (output / name).write_bytes(_canonical_bytes(value))
    # Keep arm names/ranks out of the human-facing files: their role is only to
    # construct the union candidate pool, not to influence relevance judgment.
    with (output / "qrels-human-blind-review.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["review_id", "query_id", "query", "citation_id", "title_path", "source_url", "evidence_text", "relevant", "reviewer_notes"])
        writer.writeheader()
        ordinal = 0
        for case in pool["cases"]:
            for candidate in case["candidates"]:
                ordinal += 1
                writer.writerow({"review_id": f"R2Q-{ordinal:03d}", "query_id": case["query_id"], "query": case["query"], "citation_id": candidate["citation_id"], "title_path": " / ".join(candidate["title_path"]), "source_url": candidate["source_url"], "evidence_text": candidate["evidence_text"], "relevant": "", "reviewer_notes": ""})
    with (output / "query-human-review.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["query_id", "query", "answerable", "reference_claims", "refusal_target", "reviewer_notes"])
        writer.writeheader()
        for query in draft["queries"]:
            writer.writerow({"query_id": query["query_id"], "query": query["query"], "answerable": "", "reference_claims": "", "refusal_target": "", "reviewer_notes": ""})
    hybrid_ready = pool["arm_metadata"]["hybrid"]["retrieval_mode"] == "hybrid"
    return {"corpus_sha256": manifest["corpus_sha256"], "draft_sha256": hashlib.sha256((output / "queries-draft.json").read_bytes()).hexdigest(), "pool_sha256": hashlib.sha256((output / "pooled-qrels-draft.json").read_bytes()).hexdigest(), "status": "blocked_pending_human_annotation" if hybrid_ready else "blocked_pending_local_hybrid_index"}


def build_answer_judge_preflight(
    *, corpus_root: str | Path, annotation_directory: str | Path,
) -> dict[str, Any]:
    """Prepare, but never execute, separately-approved answer and Judge calls."""
    root = Path(corpus_root); annotation = Path(annotation_directory)
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    draft = json.loads((annotation / "queries-draft.json").read_text(encoding="utf-8"))
    pool = json.loads((annotation / "pooled-qrels-draft.json").read_text(encoding="utf-8"))
    if draft.get("gold_status") != "not_signed" or pool.get("status") != "draft_pending_human_annotation":
        raise ValueError("TASK-041 preflight only accepts unsigned human-review inputs")
    bindings = {
        "corpus_manifest_sha256": _sha256_file(root / "manifest.json"),
        "corpus_sha256": manifest["corpus_sha256"],
        "queries_draft_sha256": _sha256_file(annotation / "queries-draft.json"),
        "pooled_qrels_draft_sha256": _sha256_file(annotation / "pooled-qrels-draft.json"),
        "task_041_code_sha256": _sha256_file(Path(__file__)),
        "answer_prompt_sha256": hashlib.sha256(ANSWER_PROMPT.encode("utf-8")).hexdigest(),
        "judge_prompt_sha256": hashlib.sha256(JUDGE_PROMPT.encode("utf-8")).hexdigest(),
    }
    hybrid_ready = pool.get("arm_metadata", {}).get("hybrid", {}).get("retrieval_mode") == "hybrid"
    preflight = {
        "schema_version": 1, "task": "TASK-041", "status": "HUMAN_ANNOTATION_GATE_PENDING" if hybrid_ready else "LOCAL_HYBRID_INDEX_GATE_PENDING",
        "api_calls_made": 0, "offline_only": True, "corpus_version": manifest["corpus_version"],
        "query_count": draft["query_count"], "retrieval_k": RETRIEVAL_K, "bindings": bindings,
        "answer_model": {"model": "glm-5.3-flash", "temperature": 0, "reasoning_effort": "low", "max_turns": 1},
        "judge_model": {"model": "glm-5.3-flash", "temperature": 0, "reasoning_effort": "low", "one_dimension_per_call": True, "strict_json": ["passed", "reason"]},
        "estimated_model_calls_after_human_freeze": 120,
        "maximum_model_calls_after_human_freeze": 120,
        "api_gate": {"requires": ["verified local hybrid index", "human-signed qrels and reference claims", "separate answer approval", "separate Judge approval", "matching preflight hash"], "note": "No model call is authorized by this preflight."},
    }
    preflight["preflight_sha256"] = hashlib.sha256(_canonical_bytes(preflight)).hexdigest()
    return preflight


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build TASK-041 unsigned corpus and human annotation package")
    parser.add_argument("--build-corpus", action="store_true")
    parser.add_argument("--corpus-root", type=Path, default=R2_ROOT)
    parser.add_argument("--annotation-output", type=Path)
    parser.add_argument("--hybrid-index", type=Path)
    parser.add_argument("--model-path", type=Path)
    parser.add_argument("--preflight-output", type=Path)
    args = parser.parse_args(argv)
    if args.build_corpus:
        print(json.dumps(build_corpus(args.corpus_root), ensure_ascii=False, sort_keys=True))
    if args.annotation_output:
        print(json.dumps(write_unsigned_annotation_package(
            args.annotation_output, corpus_root=args.corpus_root,
            index_directory=args.hybrid_index, model_path=args.model_path,
        ), ensure_ascii=False, sort_keys=True))
    if args.preflight_output:
        annotation = args.annotation_output or Path(".task_outputs/task-041-human-annotation-final-draft")
        preflight = build_answer_judge_preflight(corpus_root=args.corpus_root, annotation_directory=annotation)
        args.preflight_output.parent.mkdir(parents=True, exist_ok=True)
        args.preflight_output.write_bytes(_canonical_bytes(preflight))
        print(json.dumps(preflight, ensure_ascii=False, sort_keys=True))
    if not args.build_corpus and args.annotation_output is None and args.preflight_output is None:
        parser.error("choose --build-corpus, --annotation-output, and/or --preflight-output")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
