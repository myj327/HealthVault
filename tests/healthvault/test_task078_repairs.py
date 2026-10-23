"""Direct regressions for TASK-075 failures, not model-effect measurements."""
import asyncio
import copy
import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from healthvault.application.report_preview import build_report_preview
from healthvault.evaluation import current_skill_run as skill
from healthvault.evaluation import current_e2e_run as e2e
from healthvault.infrastructure.hybrid_health_knowledge import HybridHealthKnowledgeRepository, rrf_fuse
from healthvault.infrastructure.sqlite_fts_health_knowledge import SQLiteFTSHealthKnowledgeRepository
from healthvault.repositories.dense_retrieval import DenseRetrievalError
from healthvault.web.answer_validation import verify_answer
from healthvault.evaluation import gold_review
from healthvault.evaluation.ocr import runner as ocr


class FormatRepairTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.plan = skill.prepare_run(self.root / "plan", api_base="https://example.invalid/v1")
        self.payload = {"model": "glm-5.3-flash", "temperature": 0, "max_tokens": 2048, "stream": False,
                        "messages": [{"role": "system", "content": "existing task rules"}, {"role": "user", "content": "case"}],
                        "tools": [{"function": {"name": name}} for name in self.plan["tool_names"]]}

    def transport(self, raw):
        class Client:
            is_real = False
            def complete(inner, payload):
                inner.payload = payload
                return {"choices": [{"finish_reason": "stop", "message": {"content": raw}}],
                        "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}}
        client = Client()
        output = self.root / str(len(list(self.root.iterdir())))
        output.mkdir()
        transport = skill.CurrentTransport(self.plan, output, client)
        transport.case_id, transport.arm_id = "R3H-001", "active_v1"
        return transport, client

    def test_request_json_mode_keeps_existing_context_and_raw_final(self):
        raw = '{"final_status":"ok","citations":["REC:r:glucose"],"refusal":false}'
        original = copy.deepcopy(self.payload)
        transport, client = self.transport(raw)
        response = asyncio.run(transport.post_json(self.plan["endpoint"], {}, self.payload, 60))
        self.assertEqual({"type": "json_object"}, client.payload["response_format"])
        self.assertIn("existing task rules", client.payload["messages"][0]["content"])
        self.assertIn(skill.FINAL_INSTRUCTION, client.payload["messages"][0]["content"])
        self.assertEqual(original, self.payload)
        self.assertEqual(raw, response["choices"][0]["message"]["content"])

    def test_json_followed_by_prose_remains_failure_with_usage_and_no_retry(self):
        for raw in ('{"final_status":"ok","citations":[],"refusal":false}\nCoverage window: 2024.',
                    '```json\n{"final_status":"ok","citations":[],"refusal":false}\n```\nInventory: empty.'):
            with self.subTest(raw=raw):
                transport, _ = self.transport(raw)
                with self.assertRaisesRegex(ValueError, "invalid_json"):
                    asyncio.run(transport.post_json(self.plan["endpoint"], {}, self.payload, 60))
                self.assertEqual("format_protocol", transport.calls[0]["failure_category"])
                self.assertEqual(15, transport.calls[0]["usage"]["total_tokens"])
                with self.assertRaisesRegex(RuntimeError, "budget"):
                    asyncio.run(transport.post_json(self.plan["endpoint"], {}, self.payload, 60))
                self.assertEqual(1, len(transport.calls))

    def test_changed_output_contract_is_rejected_before_client(self):
        transport, client = self.transport('{"final_status":"rejected","citations":[],"refusal":true}')
        transport.plan["structured_output"] = {"response_format": {"type": "text"}}
        with self.assertRaisesRegex(ValueError, "contract"):
            asyncio.run(transport.post_json(self.plan["endpoint"], {}, self.payload, 60))
        self.assertEqual([], transport.calls)
        self.assertFalse(hasattr(client, "payload"))


class OCRRepairTests(unittest.TestCase):
    def preview(self, row, date="2024-02-11"):
        text = f"report_id: regression\nreport_date: {date}\nfacility: synthetic\n{row}"
        return text, build_report_preview("ocr-regression", text)

    def test_observed_ldl_name_and_missing_prefix_delimiter_preserve_original_evidence(self):
        for row in ("result|Idl_cholesterol|2.2|mmol/L|0.0|3.4", "resultIdl_cholesterol|3.1|mmol/L|0.0|3.4"):
            with self.subTest(row=row):
                text, preview = self.preview(row)
                self.assertEqual(["ldl_cholesterol"], [item.code for item in preview.indicators])
                self.assertEqual(row, preview.indicators[0].evidence.quote)
                self.assertEqual(4, preview.indicators[0].evidence.line_start)
                self.assertEqual(hashlib.sha256(text.encode()).hexdigest(), preview.source.sha256)

    def test_glucose_delimiter_and_merged_value_unit_are_recovered_exactly(self):
        for row, value in (("result}fasting_glucose|5.1|mmol/L|3.9|6.1", 5.1),
                           ("result}fasting_glucose|6.0mmol/L|3.9|6.1", 6.0)):
            with self.subTest(row=row):
                _, preview = self.preview(row)
                self.assertEqual(1, len(preview.indicators))
                item = preview.indicators[0]
                self.assertEqual(("fasting_glucose", value, "mmol/L"), (item.code, item.value, item.unit))
                self.assertEqual(row, item.evidence.quote)

    def test_hdl_and_unknown_idl_are_never_changed_to_ldl(self):
        _, hdl = self.preview("result|hdl_cholesterol|1.3|mmol/L|1.0|2.0")
        self.assertEqual(["hdl_cholesterol"], [item.code for item in hdl.indicators])
        for code in ("idl_cholesterol", "IDL_CHOLESTEROL", "ldl_cholestero1"):
            _, unknown = self.preview(f"result|{code}|2.2|mmol/L|0.0|3.4")
            self.assertEqual([], unknown.indicators)
            self.assertIn("unsupported_indicator", [warning.code for warning in unknown.warnings])

    def test_no_global_character_replacement_and_original_validations_remain(self):
        for row, issue in (("result|Idl_cholesterol|2.I|mmol/L|0.0|3.4", "invalid_number"),
                           ("result}fasting_glucose|5.1mg/dL|3.9|6.1", "malformed_result"),
                           ("result|Idl_cholesterol|2.2|mg/dL|0.0|3.4", "unit_mismatch"),
                           ("result|Idl_cholesterol|2.2|mmol/L|4.0|3.4", "invalid_reference_range")):
            with self.subTest(row=row):
                _, preview = self.preview(row)
                self.assertEqual([], preview.indicators)
                self.assertIn(issue, [item.code for item in preview.issues])
        _, bad_date = self.preview("result|Idl_cholesterol|2.2|mmol/L|0.0|3.4", date="2024-02-31")
        self.assertTrue(any("date" in issue.code for issue in bad_date.issues))

    def frozen_run(self, root):
        source = root / "source"
        source.mkdir()
        gold = ocr.make_gold(46)
        rows, hashes = [], {}
        for report in gold:
            image = source / "images" / (report["report_id"] + ".png")
            image.parent.mkdir(exist_ok=True)
            image.write_bytes(b"test fixture, never processed as a real image")
            artifact = source / (report["report_id"] + ".json")
            ocr._write_json(artifact, {"blocks": []})
            hashes[report["report_id"]] = ocr._sha256(image.read_bytes())
            rows.append({"report_id": report["report_id"], "image_path": str(image), "image_sha256": hashes[report["report_id"]],
                         "ocr_json_path": str(artifact), "ocr_text": "\n".join(report["source_lines"]),
                         "engine": {"name": "offline fixture"}, "failure": None})
        ocr._write_json(source / "gold.json", gold)
        ocr._write_json(source / "results.json", rows)
        ocr._write_json(source / "manifest.json", {"seed": 46, "gold_sha256": ocr._sha256((source / "gold.json").read_bytes()), "image_hashes": hashes})
        return source

    def test_saved_ocr_text_is_replayed_without_rendering_or_ocr(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = self.frozen_run(root)
            with patch.object(ocr, "OCRPipeline") as pipeline, patch.object(ocr, "render_report") as render:
                result = ocr.run(root / "new", 999, source_run=source, replay_text=True)
            pipeline.assert_not_called()
            render.assert_not_called()
            self.assertEqual(100, result["overall"]["field_micro"]["tp"])
            self.assertEqual(46, result["seed"])
            with self.assertRaises(FileExistsError):
                ocr.run(root / "new", 46, source_run=source, replay_text=True)

    def test_changed_frozen_image_is_rejected_before_output(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = self.frozen_run(root)
            next((source / "images").iterdir()).write_bytes(b"changed")
            with self.assertRaisesRegex(ValueError, "image hash changed"):
                ocr.run(root / "new", 46, source_run=source, replay_text=True)
            self.assertFalse((root / "new").exists())


class RAGRepairTests(unittest.TestCase):
    def verify(self, answer, sources, mutate=None):
        def checker(prompt, data, **kwargs):
            self.payload = data
            claims = [{"claim_id": fact["claim_id"], "supported": True,
                       "evidence_ids": [item for item in fact["allowed_evidence_ids"] if "-detail-" in item]}
                      for fact in data["facts"]]
            if mutate:
                mutate(claims)
            return {"protocol": "fact_span_v1", "answer_supported": True, "answers_question": True, "claims": claims}
        return verify_answer("核对结果", answer, sources, query=checker)

    def test_allowed_spans_belong_to_each_fact_and_all_facts_are_checked(self):
        sources = [{"citation_id": "REC:a", "detail": "左叶 5 mm", "title": "左叶"},
                   {"citation_id": "REC:b", "detail": "右叶 8 mm", "title": "右叶"}]
        result = self.verify("左叶 5 mm [REC:a]\n右叶 8 mm [REC:b]", sources)
        self.assertTrue(result["passed"])
        spans = {span["evidence_id"]: span for span in self.payload["evidence_spans"]}
        for fact in self.payload["facts"]:
            self.assertTrue(fact["allowed_evidence_ids"])
            self.assertTrue(all(spans[i]["citation_id"] == fact["citation_id"] for i in fact["allowed_evidence_ids"]))
        self.assertFalse(self.verify("左叶 5 mm [REC:a]\n右叶 8 mm [REC:b]", sources, lambda claims: claims.pop())["passed"])

    def test_cross_citation_and_unknown_spans_still_reject(self):
        sources = [{"citation_id": "REC:a", "detail": "5 mm"}, {"citation_id": "REC:b", "detail": "5 mm"}]
        for identifier in ("S2-detail-1", "S99-detail-1"):
            self.assertFalse(self.verify("5 mm [REC:a]", sources, lambda claims: claims[0].update(evidence_ids=[identifier]))["passed"])
        self.assertFalse(self.verify("5 mm [REC:missing]", sources)["passed"])
        self.assertEqual([], self.payload["facts"][0]["allowed_evidence_ids"])

    def test_unfounded_number_unit_or_percentage_rejects_even_with_true_flags(self):
        sources = [{"citation_id": "REC:a", "detail": "血糖 5.2 mmol/L，比例 7%"}]
        for fact in ("血糖 6.2 mmol/L", "血糖 5.2 mg/dL", "比例 7", "尺寸 7 mm"):
            self.assertFalse(self.verify(fact + " [REC:a]", sources)["passed"])

    def test_three_saved_real_cross_citation_responses_remain_rejected(self):
        path = Path(__file__).parent / "fixtures/task078_rag_binding_replays.json"
        for case in json.loads(path.read_text(encoding="utf-8"))["cases"]:
            with self.subTest(query=case["query_id"]):
                result = verify_answer(case["question"], case["answer"], case["sources"], query=lambda *_args, **_kwargs: case["original_verifier"])
                self.assertFalse(result["passed"])
                corrected = verify_answer(case["question"], case["answer"], case["sources"], query=lambda *_args, **_kwargs: case["binding_only_replay"])
                if case["query_id"] == "MQ-001":
                    # Correct span ownership still leaves the blood-test subclaim unsupported.
                    self.assertFalse(corrected["passed"])
                    self.assertIsNone(corrected["answer"])
                    continue
                # Saved coarse verdicts cannot replace checks of new clause IDs.
                by_parent = {item["claim_id"]: item for item in case["binding_only_replay"]["claims"]}
                def checker(_prompt, data, **_kwargs):
                    return {**case["binding_only_replay"], "claims": [
                        {**by_parent[fact["claim_id"].split(".")[0]], "claim_id": fact["claim_id"]}
                        for fact in data["facts"]]}
                checked = verify_answer(case["question"], case["answer"], case["sources"], query=checker)
                self.assertTrue(checked["passed"])
                self.assertEqual(case["answer"], checked["answer"])


class HybridRepairTests(unittest.TestCase):
    def repo(self, *, failure=False):
        lexical = SQLiteFTSHealthKnowledgeRepository()
        self.addCleanup(lexical.close)
        hit = lexical.search("HbA1c", 1)[0]
        class Embedder:
            dimension = 2
            def embed(self, texts):
                if failure:
                    raise DenseRetrievalError("offline failure")
                return ((1.0, 0.0),)
        class Index:
            dimension = 2
            def search(self, vector, limit):
                return ((hit.document.chunk_id, 1.0),)
        return HybridHealthKnowledgeRepository(lexical, embedder=Embedder(), vector_index=Index())

    def test_dense_results_really_enter_rrf(self):
        repo = self.repo()
        hits = repo.search("HbA1c", 5)
        meta = repo.last_search_metadata
        self.assertTrue(meta["dense_participated"])
        self.assertEqual("hybrid", meta["retrieval_mode"])
        self.assertEqual([item[0] for item in rrf_fuse(meta["lexical_chunk_ids"], meta["dense_chunk_ids"], limit=5)],
                         [hit.document.chunk_id for hit in hits])

    def test_dense_query_failure_is_logged_as_lexical(self):
        repo = self.repo(failure=True)
        self.assertTrue(repo.search("HbA1c", 5))
        self.assertEqual("lexical", repo.last_search_metadata["retrieval_mode"])
        self.assertFalse(repo.last_search_metadata["dense_participated"])
        self.assertEqual("dense_query_failed", repo.last_search_metadata["dense_fallback_reason"])

    def test_current_twenty_scenarios_keep_oracles_and_document_missing_sources(self):
        corpus = e2e.active_corpus_root() / "chunks/hybrid_v1.json"
        old = e2e.legacy.load_dataset()
        current = e2e.load_current_dataset(corpus)
        self.assertEqual(20, len(current["cases"]))
        adjusted = {row["case_id"] for row in current["source_scope_adjustments"]}
        self.assertEqual({"R5E-010", "R5E-011", "R5E-012"}, adjusted)
        for old_case, case in zip(old["cases"], current["cases"]):
            self.assertEqual(old_case["prompt"], case["prompt"])
            self.assertEqual(old_case["scenario"], case["scenario"])
            if case["case_id"] in adjusted:
                self.assertEqual(["empty"], case["expected"]["required_tool_result_statuses"])
                self.assertEqual("rejected" if case["case_id"] == "R5E-012" else "insufficient_evidence", case["expected"]["final_status"])
                self.assertEqual([], case["expected"]["allowed_citation_ids"])
            else:
                self.assertEqual({k: v for k, v in old_case["expected"].items() if k != "allowed_citation_ids"},
                                 {k: v for k, v in case["expected"].items() if k != "allowed_citation_ids"})
        old_corpus = e2e.load_current_dataset(e2e.legacy.CORPUS_PATH)
        self.assertEqual([], old_corpus["source_scope_adjustments"])
        self.assertEqual(["ok", "ok", "rejected"], [c["expected"]["final_status"] for c in old_corpus["cases"] if c["case_id"] in adjusted])
        self.assertEqual(["outcome", "behavior", "response"], e2e.legacy.build_case_matrix(current, e2e.legacy.load_fixture())["oracle_order"])


class GoldReviewTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.source = gold_review.Path(skill.ROOT / "healthvault/evaluation/rag/task-026/task-026-human-reviewed-gold.json")
        self.old_hash = skill.digest(self.source)
        self.materials = self.root / "materials.json"
        self.materials.write_text("{}", encoding="utf-8")
        cases = json.loads(self.source.read_text(encoding="utf-8"))["queries"]
        identifiers = sorted(gold_review.REQUIRED) + [q["query_id"] for q in cases if q["query_id"] not in gold_review.REQUIRED][:6]
        self.decisions = {"status": "human_signed", "signed_by": "offline-test-human", "source_gold_sha256": self.old_hash,
                          "materials_sha256": skill.digest(self.materials), "cases": [{"query_id": i, "answer_A": "correct",
                          "answer_B": "appropriate_refusal", "gold_action": "keep"} for i in identifiers]}

    def finalize(self, name="new"):
        decision_path = self.root / (name + "-decisions.json")
        decision_path.write_text(json.dumps(self.decisions), encoding="utf-8")
        return gold_review.publish_reviewed_gold(self.source, self.materials, decision_path, self.root / name,
                                                 human_confirmed_by="offline-test-human")

    def test_pending_review_cannot_publish(self):
        self.decisions["status"] = "pending"
        with self.assertRaisesRegex(ValueError, "human confirmation"):
            self.finalize()
        self.assertFalse((self.root / "new").exists())

    def test_incomplete_review_or_missing_counterexample_cannot_publish(self):
        self.decisions["cases"].pop()
        with self.assertRaisesRegex(ValueError, "ten"):
            self.finalize()
        self.assertEqual(self.old_hash, skill.digest(self.source))

    def test_bad_claim_format_or_unknown_qrel_cannot_publish(self):
        row = self.decisions["cases"][0]
        row.update(gold_action="amend", answerable=True, qrels=["HVK:missing"],
                   reference_claims=[{"claim_id": "RC1", "text": "fact"}], refusal_target=None, notes="test only")
        with self.assertRaisesRegex(ValueError, "amendment"):
            self.finalize()
        row["qrels"] = json.loads(self.source.read_text(encoding="utf-8"))["queries"][0]["qrels"]
        row["reference_claims"] = ["bad string"]
        with self.assertRaisesRegex(ValueError, "individual facts"):
            self.finalize("second")

    def test_test_only_confirmation_creates_new_file_and_keeps_old_gold(self):
        audit = self.finalize()
        self.assertEqual(10, audit["reviewed_count"])
        self.assertEqual(audit["old_denominators"], audit["new_denominators"])
        self.assertEqual(self.old_hash, skill.digest(self.source))
        with self.assertRaises(FileExistsError):
            self.finalize()

    def test_user_requested_assistant_review_never_becomes_human_signed(self):
        self.decisions.update(status="agent_reviewed", signed_by="Codex")
        path = self.root / "assistant-decisions.json"
        path.write_text(json.dumps(self.decisions), encoding="utf-8")
        gold_review.publish_agent_reviewed_gold(self.source, self.materials, path, self.root / "assistant-gold",
                                                user_authorization="offline test authorization")
        new = json.loads((self.root / "assistant-gold/gold.json").read_text(encoding="utf-8"))
        self.assertEqual("agent_reviewed", new["gold_status"])
        self.assertIsNone(new["human_confirmation"])
        self.assertEqual("assistant", new["review_confirmation"]["reviewer_kind"])


if __name__ == "__main__":
    unittest.main()
