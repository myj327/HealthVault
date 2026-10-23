import hashlib
import json
import shutil
import tempfile
import unittest
from pathlib import Path

from healthvault.rag.corpus import (
    DEFAULT_CORPUS_ROOT,
    load_chunk_artifact,
    load_source_snapshots,
    parse_chunk_id,
    render_corpus_artifacts,
)
from healthvault.rag.chunking import estimate_tokens


def canonical_bytes(value):
    return (
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")


class KnowledgeCorpusTests(unittest.TestCase):
    def test_default_corpus_is_official_versioned_and_reproducible(self):
        sources = load_source_snapshots(DEFAULT_CORPUS_ROOT)

        self.assertGreaterEqual(len(sources), 10)
        self.assertLessEqual(len(sources), 15)
        self.assertEqual(len(sources), len({source.source_id for source in sources}))
        for source in sources:
            self.assertIn(
                source.publisher,
                {"CDC", "NIDDK", "NHLBI", "MedlinePlus", "ODPHP"},
            )
            self.assertTrue(source.url.startswith("https://"))
            self.assertEqual("2026-08-24", source.retrieved_on)
            self.assertIn("public domain", source.reuse_basis.casefold())
            self.assertEqual(64, len(source.source_sha256))
            self.assertEqual(
                source.title_path_whitelist,
                tuple(section.title_path for section in source.sections),
            )
            self.assertFalse(
                any(
                    block.kind == "heading"
                    for section in source.sections
                    for block in section.blocks
                )
            )

        evidence_text = "\n".join(
            block.text
            for source in sources
            for section in source.sections
            for block in section.blocks
        ).casefold()
        for forbidden in (
            "book traversal links",
            "previous next",
            "pdf version",
            "complete blood count",
            "blood clotting tests",
            "autoantibody",
            "warfarin",
            "heparin",
            "statin",
        ):
            with self.subTest(forbidden=forbidden):
                self.assertNotIn(forbidden, evidence_text)

        rendered = render_corpus_artifacts(DEFAULT_CORPUS_ROOT)
        self.assertEqual(
            {
                "chunks/natural_section.json",
                "chunks/hybrid_v1.json",
                "manifest.json",
            },
            set(rendered),
        )
        for relative_path, content in rendered.items():
            self.assertEqual(
                (DEFAULT_CORPUS_ROOT / relative_path).read_bytes(),
                content,
            )
        self.assertEqual(rendered, render_corpus_artifacts(DEFAULT_CORPUS_ROOT))

    def test_hybrid_chunks_and_manifest_have_complete_provenance(self):
        sources = {
            source.source_id: source
            for source in load_source_snapshots(DEFAULT_CORPUS_ROOT)
        }
        artifact = load_chunk_artifact(
            DEFAULT_CORPUS_ROOT / "chunks" / "hybrid_v1.json"
        )
        natural = load_chunk_artifact(
            DEFAULT_CORPUS_ROOT / "chunks" / "natural_section.json"
        )
        manifest = json.loads(
            (DEFAULT_CORPUS_ROOT / "manifest.json").read_text(encoding="utf-8")
        )

        self.assertEqual("r1-2026-08-24-v1", artifact.corpus_version)
        self.assertEqual("hybrid_v1", artifact.strategy)
        self.assertGreaterEqual(len(artifact.chunks), 80)
        self.assertLessEqual(len(artifact.chunks), 150)
        self.assertEqual(len(artifact.chunks), len({c.chunk_id for c in artifact.chunks}))
        natural_by_section = {
            (chunk.source_id, chunk.title_path): chunk for chunk in natural.chunks
        }
        self.assertEqual(
            sum(len(source.sections) for source in sources.values()),
            len(natural.chunks),
        )
        self.assertEqual(
            {
                (source.source_id, section.title_path)
                for source in sources.values()
                for section in source.sections
            },
            set(natural_by_section),
        )
        short_intact = []
        for chunk in artifact.chunks:
            parsed = parse_chunk_id(chunk.chunk_id)
            self.assertEqual(artifact.corpus_version, parsed.corpus_version)
            self.assertEqual(chunk.source_id, parsed.source_id)
            self.assertEqual(chunk.chunk_id, chunk.citation_id)
            self.assertEqual(chunk.ordinal, parsed.ordinal)
            self.assertTrue(chunk.title_path)
            self.assertTrue(chunk.body.strip())
            self.assertLessEqual(chunk.estimated_tokens, 600)
            if chunk.estimated_tokens < 80:
                short_intact.append(chunk)
                natural_chunk = natural_by_section[
                    (chunk.source_id, chunk.title_path)
                ]
                self.assertEqual(natural_chunk.body, chunk.body)
                self.assertEqual(0, chunk.overlap_estimated_tokens)
            self.assertEqual(
                hashlib.sha256(chunk.body.encode("utf-8")).hexdigest(),
                chunk.body_sha256,
            )
            self.assertTrue(chunk.url.startswith("https://"))
            self.assertEqual("2026-08-24", chunk.retrieved_on)
            source = sources[chunk.source_id]
            self.assertEqual(source.version, chunk.source_version)
            self.assertEqual(source.source_sha256, chunk.source_sha256)
            self.assertEqual(source.url, chunk.url)
            self.assertEqual(source.retrieved_on, chunk.retrieved_on)
            self.assertIn(chunk.title_path, {section.title_path for section in source.sections})
            self.assertTrue(all(" | " not in part for part in chunk.title_path))

        stats = manifest["strategies"]["hybrid_v1"]
        self.assertEqual(len(artifact.chunks), stats["chunk_count"])
        self.assertEqual(len(short_intact), stats["short_intact_count"])
        self.assertGreater(stats["short_intact_count"], 0)
        self.assertEqual(80, manifest["chunking"]["minimum_estimated_tokens"])
        self.assertEqual(600, manifest["chunking"]["maximum_estimated_tokens"])
        self.assertEqual([40, 80], manifest["chunking"]["overlap_estimated_tokens"])
        self.assertEqual(
            64,
            len(manifest["corpus_sha256"]),
        )

    def test_one_source_fixture_change_is_isolated_and_updates_corpus_hashes(self):
        baseline = render_corpus_artifacts(DEFAULT_CORPUS_ROOT)
        with tempfile.TemporaryDirectory() as temp_dir:
            copied_root = Path(temp_dir) / DEFAULT_CORPUS_ROOT.name
            shutil.copytree(DEFAULT_CORPUS_ROOT, copied_root)
            source_path = sorted((copied_root / "sources").glob("*.json"))[0]
            source = json.loads(source_path.read_text(encoding="utf-8"))
            changed_source_id = source["source_id"]
            source["sections"][0]["blocks"][0]["text"] += " Fixture mutation."
            source["source_sha256"] = hashlib.sha256(
                json.dumps(
                    source["sections"],
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                ).encode("utf-8")
            ).hexdigest()
            source_path.write_text(
                json.dumps(source, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )

            changed = render_corpus_artifacts(copied_root)

        self.assertNotEqual(baseline["manifest.json"], changed["manifest.json"])
        for strategy in ("natural_section", "hybrid_v1"):
            relative_path = f"chunks/{strategy}.json"
            before = json.loads(baseline[relative_path])
            after = json.loads(changed[relative_path])
            self.assertNotEqual(before, after)
            self.assertEqual(
                [
                    chunk
                    for chunk in before["chunks"]
                    if chunk["source_id"] != changed_source_id
                ],
                [
                    chunk
                    for chunk in after["chunks"]
                    if chunk["source_id"] != changed_source_id
                ],
            )

        before_manifest = json.loads(baseline["manifest.json"])
        after_manifest = json.loads(changed["manifest.json"])
        self.assertNotEqual(
            before_manifest["corpus_sha256"], after_manifest["corpus_sha256"]
        )
        before_sources = {
            source["source_id"]: source for source in before_manifest["sources"]
        }
        after_sources = {
            source["source_id"]: source for source in after_manifest["sources"]
        }
        self.assertEqual(
            {changed_source_id},
            {
                source_id
                for source_id in before_sources
                if before_sources[source_id] != after_sources[source_id]
            },
        )

    def copy_corpus(self, directory):
        copied_root = Path(directory) / DEFAULT_CORPUS_ROOT.name
        shutil.copytree(DEFAULT_CORPUS_ROOT, copied_root)
        return copied_root

    def write_artifact(self, root, strategy, artifact, update_manifest=True):
        artifact_path = root / "chunks" / f"{strategy}.json"
        artifact_bytes = canonical_bytes(artifact)
        artifact_path.write_bytes(artifact_bytes)
        if update_manifest:
            manifest_path = root / "manifest.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest["strategies"][strategy]["artifact_sha256"] = hashlib.sha256(
                artifact_bytes
            ).hexdigest()
            manifest_path.write_bytes(canonical_bytes(manifest))
        return artifact_path

    def test_artifact_rejects_forged_body_old_hash_and_manifest_mismatch(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = self.copy_corpus(temp_dir)
            path = root / "chunks" / "hybrid_v1.json"
            artifact = json.loads(path.read_text(encoding="utf-8"))
            forged_body = " ".join(["FORGED EVIDENCE."] * 30)
            artifact["chunks"][0]["body"] = forged_body
            artifact["chunks"][0]["estimated_tokens"] = estimate_tokens(forged_body)
            path = self.write_artifact(root, "hybrid_v1", artifact)
            with self.assertRaisesRegex(ValueError, "body_sha256"):
                load_chunk_artifact(path)

        with tempfile.TemporaryDirectory() as temp_dir:
            root = self.copy_corpus(temp_dir)
            path = root / "chunks" / "hybrid_v1.json"
            artifact = json.loads(path.read_text(encoding="utf-8"))
            artifact["chunks"][0]["title"] = "unexpected field"
            path = self.write_artifact(
                root, "hybrid_v1", artifact, update_manifest=False
            )
            with self.assertRaisesRegex(ValueError, "manifest artifact hash"):
                load_chunk_artifact(path)

    def test_artifact_rejects_forged_body_with_all_derived_hashes_updated(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = self.copy_corpus(temp_dir)
            path = root / "chunks" / "hybrid_v1.json"
            artifact = json.loads(path.read_text(encoding="utf-8"))
            forged_body = " ".join(["FORGED EVIDENCE."] * 30)
            artifact["chunks"][0]["body"] = forged_body
            artifact["chunks"][0]["estimated_tokens"] = estimate_tokens(forged_body)
            artifact["chunks"][0]["body_sha256"] = hashlib.sha256(
                forged_body.encode("utf-8")
            ).hexdigest()
            artifact_bytes = canonical_bytes(artifact)
            path.write_bytes(artifact_bytes)

            manifest_path = root / "manifest.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            sizes = [chunk["estimated_tokens"] for chunk in artifact["chunks"]]
            strategy = manifest["strategies"]["hybrid_v1"]
            strategy.update(
                {
                    "chunk_count": len(sizes),
                    "minimum_estimated_tokens": min(sizes),
                    "maximum_estimated_tokens": max(sizes),
                    "mean_estimated_tokens": round(sum(sizes) / len(sizes), 3),
                    "target_range_count": sum(250 <= size <= 450 for size in sizes),
                    "artifact_sha256": hashlib.sha256(artifact_bytes).hexdigest(),
                }
            )
            digest_input = "\n".join(
                [
                    *(source["source_sha256"] for source in manifest["sources"]),
                    manifest["aliases_sha256"],
                    manifest["strategies"]["natural_section"]["artifact_sha256"],
                    strategy["artifact_sha256"],
                ]
            ).encode("ascii")
            manifest["corpus_sha256"] = hashlib.sha256(digest_input).hexdigest()
            manifest_path.write_bytes(canonical_bytes(manifest))

            with self.assertRaisesRegex(ValueError, "derived from source snapshots"):
                load_chunk_artifact(path)

    def test_artifact_rejects_schema_id_strategy_and_source_forgery(self):
        mutations = {
            "extra field": lambda artifact: artifact.update({"extra": True}),
            "missing top field": lambda artifact: artifact.pop("schema_version"),
            "boolean schema version": lambda artifact: artifact.update(
                {"schema_version": True}
            ),
            "invalid chunk schema": lambda artifact: artifact["chunks"][0].update(
                {"extra": True}
            ),
            "missing chunk field": lambda artifact: artifact["chunks"][0].pop(
                "language"
            ),
            "invalid chunk ID": lambda artifact: artifact["chunks"][0].update(
                {
                    "chunk_id": artifact["chunks"][0]["chunk_id"][:-12]
                    + "XXXXXXXXXXXX",
                    "citation_id": artifact["chunks"][0]["citation_id"][:-12]
                    + "XXXXXXXXXXXX",
                }
            ),
            "duplicate chunk ID": lambda artifact: artifact["chunks"][1].update(
                {
                    "chunk_id": artifact["chunks"][0]["chunk_id"],
                    "citation_id": artifact["chunks"][0]["citation_id"],
                    "source_id": artifact["chunks"][0]["source_id"],
                    "source_version": artifact["chunks"][0]["source_version"],
                    "title_path": artifact["chunks"][0]["title_path"],
                    "ordinal": artifact["chunks"][0]["ordinal"],
                }
            ),
            "invalid strategy": lambda artifact: artifact.update(
                {"strategy": "unexpected_strategy"}
            ),
            "source metadata mismatch": lambda artifact: artifact["chunks"][0].update(
                {"publisher": "Forged Publisher"}
            ),
            "invalid source hash": lambda artifact: artifact["chunks"][0].update(
                {"source_sha256": "X" * 64}
            ),
            "invalid source URL": lambda artifact: artifact["chunks"][0].update(
                {"url": "http://example.test/forged"}
            ),
            "invalid ordinal": lambda artifact: artifact["chunks"][0].update(
                {"ordinal": -1}
            ),
            "invalid token estimate": lambda artifact: artifact["chunks"][0].update(
                {
                    "estimated_tokens": artifact["chunks"][0]["estimated_tokens"]
                    + 1
                }
            ),
        }
        for label, mutate in mutations.items():
            with self.subTest(label=label), tempfile.TemporaryDirectory() as temp_dir:
                root = self.copy_corpus(temp_dir)
                path = root / "chunks" / "hybrid_v1.json"
                artifact = json.loads(path.read_text(encoding="utf-8"))
                mutate(artifact)
                path = self.write_artifact(root, "hybrid_v1", artifact)
                with self.assertRaises(ValueError):
                    load_chunk_artifact(path)


if __name__ == "__main__":
    unittest.main()
