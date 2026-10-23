import unittest

from healthvault.evaluation.used_cases import (
    find_held_out_collisions,
    normalized_query_sha256,
    validate_used_case_registry,
)


def registry(entries):
    return {
        "schema_version": 1,
        "registry_version": "used-cases-r1-v1",
        "created_on": "2026-08-26",
        "normalization": "nfkc-casefold-alnum-v1",
        "tracks": [
            {"track": "rag", "case_version": "rag-eval-r1-v3-human", "purpose": "retrieval_and_generation"},
            {"track": "agent_e2e", "case_version": "agent-e2e-r1-v1", "purpose": "ordinary_end_to_end"},
            {"track": "long_fault", "case_version": "long-fault-r1-v1", "purpose": "long_horizon_and_faults"},
            {"track": "skill_evolution", "case_version": "skill-evolution-r1-v1", "purpose": "skill_evolution"},
        ],
        "entries": entries,
    }


def entry(case_id, family, track, version, query):
    return {
        "case_id": case_id,
        "case_family_id": family,
        "track": track,
        "case_version": version,
        "split": "test",
        "held_out": True,
        "query": query,
        "query_sha256": normalized_query_sha256(query),
        "upstream_dataset": "synthetic-test",
        "upstream_id": case_id,
        "status": "draft",
        "source": "synthetic-test",
    }


class UsedCaseRegistryTests(unittest.TestCase):
    def test_exact_family_and_near_synonym_collisions_are_rejected_across_held_out_tracks(self):
        cases = [
            entry("R1", "family-a", "rag", "rag-eval-r1-v3-human", "How long should I fast before a blood test?"),
            entry("E1", "family-b", "agent_e2e", "agent-e2e-r1-v1", "How long do I need to fast before my blood test?"),
        ]
        collisions = find_held_out_collisions(cases)
        self.assertEqual("near_synonym", collisions[0]["reason"])
        with self.assertRaisesRegex(ValueError, "held-out collision"):
            validate_used_case_registry(registry(cases))

        cases[1]["case_family_id"] = "family-a"
        cases[1]["query"] = "unrelated wording"
        self.assertEqual("case_family_id", find_held_out_collisions(cases)[0]["reason"])

    def test_independent_track_versions_and_noncolliding_cases_are_valid(self):
        value = registry([
            entry("R1", "family-a", "rag", "rag-eval-r1-v3-human", "What does an A1C test measure?"),
            entry("E1", "family-b", "agent_e2e", "agent-e2e-r1-v1", "Create a visit brief from a synthetic record."),
            entry("L1", "family-c", "long_fault", "long-fault-r1-v1", "Resume after a synthetic storage fault."),
            entry("S1", "family-d", "skill_evolution", "skill-evolution-r1-v1", "Merge a durable formatting rule."),
        ])
        parsed = validate_used_case_registry(value)
        self.assertEqual(4, len(parsed["entries"]))

    def test_registry_rejects_a_stale_normalized_query_hash(self):
        value = registry([
            entry("R1", "family-a", "rag", "rag-eval-r1-v3-human", "What does A1C measure?")
        ])
        value["entries"][0]["query_sha256"] = "0" * 64
        with self.assertRaisesRegex(ValueError, "bind the normalized query"):
            validate_used_case_registry(value)


if __name__ == "__main__":
    unittest.main()
