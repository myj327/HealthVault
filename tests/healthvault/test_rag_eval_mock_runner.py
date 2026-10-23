import json
import tempfile
import unittest
from pathlib import Path

from healthvault.evaluation.rag.generation import (
    GenerationJudgeRunner,
    MockModelClient,
    ModelConfig,
    ModelResponse,
    _deterministic_generation_metrics,
    summarize_generation_quality,
    summarize_usage,
)
from healthvault.evaluation.rag.run_control import RunLimits, build_run_manifest


def synthetic_provenance():
    return {
        "schema_version": 1,
        "current_head": "320254658709b05e285d7d547e52881cccb0c679",
        "worktree_dirty": True,
        "evaluation_source_sha256": "1" * 64,
        "evaluation_source_files": {
            "healthvault/evaluation/rag/generation.py": "1" * 64,
        },
        "prompt_sha256": "2" * 64,
        "validator_sha256": "3" * 64,
    }


class RagEvalMockRunnerTests(unittest.TestCase):
    def test_mock_runner_covers_answer_refusal_invalid_citation_retry_and_resume(self):
        citation = "HVK:r1-2026-08-24-v1:niddk-diabetes-tests:003:9a52fd667ccc"
        generator = MockModelClient(
            {
                "answer": [
                    ModelResponse(
                        '{"status":"answered","answer":"Supported.","claims":[{"claim_id":"C1","text":"Supported.","citation_ids":["' + citation + '"]}],"refusal_reason":null}',
                        input_tokens=20,
                        output_tokens=8,
                        latency_ms=4,
                    )
                ],
                "refuse": [
                    ModelResponse(
                        '{"status":"refused","answer":"Insufficient evidence.","claims":[],"refusal_reason":"The supplied evidence is insufficient."}',
                        input_tokens=18,
                        output_tokens=7,
                        latency_ms=3,
                    )
                ],
                "invalid": [
                    ModelResponse(
                        '{"status":"answered","answer":"Bad citation.","claims":[{"claim_id":"C1","text":"Bad citation.","citation_ids":["UNKNOWN"]}],"refusal_reason":null}',
                        input_tokens=19,
                        output_tokens=7,
                        latency_ms=3,
                    )
                ],
                "retry": [TimeoutError("synthetic timeout"), ModelResponse(
                    '{"status":"answered","answer":"Recovered.","claims":[{"claim_id":"C1","text":"Recovered.","citation_ids":["' + citation + '"]}],"refusal_reason":null}',
                    input_tokens=21,
                    output_tokens=8,
                    latency_ms=5,
                )],
                "failure": [TimeoutError("one"), TimeoutError("two")],
            }
        )
        judge = MockModelClient(
            {
                key: [ModelResponse(
                    '{"faithfulness":4,"relevance":4,"completeness":4,"expression":4,"reason":"mock","evidence_citation_ids":[]}',
                    input_tokens=12,
                    output_tokens=5,
                    latency_ms=2,
                )]
                for key in ("answer", "refuse", "invalid", "retry")
            }
        )
        cases = (
            {"query_id": "answer", "query": "q", "answerable": True,
             "reference_claims": ["Supported."], "expected_citations": [citation]},
            {"query_id": "refuse", "query": "q", "answerable": False,
             "reference_claims": [], "expected_citations": []},
            {"query_id": "invalid", "query": "q", "answerable": True,
             "reference_claims": [], "expected_citations": [citation]},
            {"query_id": "retry", "query": "q", "answerable": True,
             "reference_claims": [], "expected_citations": [citation]},
            {"query_id": "failure", "query": "q", "answerable": True,
             "reference_claims": [], "expected_citations": [citation]},
        )
        contexts = {case["query_id"]: [{"citation_id": citation, "body": "Supported."}]
                    for case in cases}

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output = root / "mock.jsonl"
            generation_config = ModelConfig("mock", "mock-generator", 0.0, 256)
            judge_config = ModelConfig("mock", "mock-judge", 0.0, 128)
            limits = RunLimits(20, 10_000, 10_000, 1.0)
            manifest = build_run_manifest(
                eval_version="rag-eval-r1-v2",
                dataset_sha256="a" * 64,
                frozen_test_sha256="b" * 64,
                corpus_sha256="c" * 64,
                artifact_sha256={"natural_section": "d" * 64, "hybrid_v1": "e" * 64},
                aliases_sha256="f" * 64,
                query_ids=tuple(case["query_id"] for case in cases),
                generation_config=generation_config,
                judge_config=judge_config,
                max_retries=1,
                limits=limits,
                implementation_provenance=synthetic_provenance(),
                created_at="2026-08-25T12:00:00+08:00",
            )
            runner = GenerationJudgeRunner(
                generator,
                judge,
                generation_config,
                judge_config,
                max_retries=1,
                limits=limits,
                run_manifest=manifest,
                manifest_path=root / "manifest.json",
                attempt_ledger_path=root / "attempts.jsonl",
            )
            first = runner.run(cases, contexts, "hybrid_v1", output)
            second = runner.run(cases, contexts, "hybrid_v1", output)
            records = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]

        self.assertEqual(first, second)
        self.assertEqual(5, len(records))
        by_id = {record["query_id"]: record for record in records}
        self.assertTrue(by_id["answer"]["deterministic"]["citation_valid"])
        self.assertTrue(by_id["refuse"]["deterministic"]["refusal_correct"])
        self.assertFalse(by_id["invalid"]["deterministic"]["citation_valid"])
        self.assertFalse(by_id["invalid"]["deterministic"]["citation_format_valid"])
        self.assertEqual(1, by_id["retry"]["generation_call"]["retries"])
        self.assertFalse(by_id["failure"]["generation_call"]["success"])
        self.assertEqual(1, by_id["failure"]["generation_call"]["retries"])
        usage = summarize_usage(records)
        self.assertGreater(usage["input_tokens"], 0)
        self.assertEqual(0.0, usage["cost_usd"])
        quality = summarize_generation_quality(records)
        self.assertEqual(5, quality["query_count"])
        self.assertEqual(4, quality["evaluated_count"])
        self.assertEqual(1, quality["generation_failure_count"])
        self.assertEqual(0.75, quality["citation_resolvable_rate"])
        self.assertEqual(4.0, quality["judge_faithfulness_mean"])
        self.assertEqual(4.0, quality["judge_completeness_mean"])
        self.assertEqual(4.0, quality["judge_expression_mean"])

    def test_deterministic_validator_rejects_duplicate_claim_ids_and_invalid_refusal(self):
        citation = "HVK:r1-2026-08-24-v1:niddk-diabetes-tests:003:9a52fd667ccc"
        case = {
            "query_id": "strict", "query": "q", "answerable": True,
            "reference_claims": [{"claim_id": "RC1", "text": "Supported."}],
            "expected_citations": [citation],
        }
        context = [{"citation_id": citation, "body": "Supported."}]
        duplicate = json.dumps({
            "status": "answered", "answer": "Supported.", "refusal_reason": None,
            "claims": [
                {"claim_id": "C1", "text": "Supported.", "citation_ids": [citation]},
                {"claim_id": "C1", "text": "Supported.", "citation_ids": [citation]},
            ],
        })
        invalid_refusal = json.dumps({
            "status": "refused", "answer": "No.", "claims": [], "refusal_reason": None,
        })
        duplicate_metrics, _ = _deterministic_generation_metrics(case, context, duplicate)
        refusal_metrics, _ = _deterministic_generation_metrics(case, context, invalid_refusal)
        self.assertFalse(duplicate_metrics["json_valid"])
        self.assertFalse(refusal_metrics["json_valid"])


if __name__ == "__main__":
    unittest.main()
