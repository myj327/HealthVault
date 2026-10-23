import unittest
from dataclasses import replace

from healthvault.rag.chunking import (
    _section_units,
    SourceBlock,
    SourceSection,
    SourceSnapshot,
    build_chunks,
    estimate_tokens,
)
from healthvault.rag.corpus import DEFAULT_CORPUS_ROOT, load_source_snapshots


def words(prefix, count):
    return " ".join(f"{prefix}{index}" for index in range(count))


def sentences(prefix, count, words_per_sentence=12):
    return " ".join(
        f"{words(prefix + str(index) + '-', words_per_sentence)}."
        for index in range(count)
    )


class KnowledgeChunkingTests(unittest.TestCase):
    def make_source(self):
        return SourceSnapshot(
            source_id="synthetic-official-source",
            publisher="Synthetic Federal Publisher",
            title="Synthetic public-domain fixture",
            version="fixture-v1",
            url="https://example.test/official",
            language="en",
            retrieved_on="2026-08-24",
            reuse_basis="Synthetic public domain fixture.",
            source_sha256="a" * 64,
            sections=(
                SourceSection(
                    title_path=("Document", "Short section"),
                    topic_tags=("fasting_glucose",),
                    blocks=(
                        SourceBlock("paragraph", words("shorta", 42)),
                        SourceBlock("paragraph", words("shortb", 44)),
                    ),
                ),
                SourceSection(
                    title_path=("Document", "Long section"),
                    topic_tags=("hba1c",),
                    blocks=tuple(
                        SourceBlock("paragraph", sentences(f"long{index}-", 5))
                        for index in range(5)
                    ),
                ),
                SourceSection(
                    title_path=("Document", "Structured list"),
                    topic_tags=("visit_preparation",),
                    blocks=(SourceBlock("list", words("item", 100)),),
                ),
                SourceSection(
                    title_path=("Document", "Structured table"),
                    topic_tags=("ldl_cholesterol",),
                    blocks=(
                        SourceBlock(
                            "table",
                            "Column A | Column B\n" + "\n".join(
                                f"row{index} {words('cell', 28)} | value{index}"
                                for index in range(8)
                            ),
                        ),
                    ),
                ),
            ),
        )

    def test_natural_and_hybrid_strategies_are_deterministic_and_distinct(self):
        source = self.make_source()
        natural = build_chunks((source,), "natural_section")
        first = build_chunks((source,), "hybrid_v1")
        second = build_chunks((source,), "hybrid_v1")

        self.assertEqual(4, len(natural))
        self.assertEqual(first, second)
        self.assertNotEqual(natural, first)
        self.assertGreater(len(first), len(natural))
        self.assertTrue(all(80 <= chunk.estimated_tokens <= 600 for chunk in first))

        short_chunks = [chunk for chunk in first if "Short section" in chunk.title_path]
        self.assertEqual(1, len(short_chunks))
        self.assertIn("shorta0", short_chunks[0].body)
        self.assertIn("shortb0", short_chunks[0].body)

        long_chunks = [chunk for chunk in first if "Long section" in chunk.title_path]
        self.assertGreaterEqual(len(long_chunks), 2)
        self.assertTrue(any(chunk.overlap_estimated_tokens >= 40 for chunk in long_chunks[1:]))
        self.assertTrue(all(chunk.overlap_estimated_tokens <= 80 for chunk in long_chunks))
        self.assertTrue(all(chunk.overlap_estimated_tokens == 0 for chunk in first if chunk not in long_chunks))

        table_chunks = [chunk for chunk in first if "Structured table" in chunk.title_path]
        self.assertTrue(table_chunks)
        self.assertTrue(all(chunk.body.startswith("Column A | Column B") for chunk in table_chunks))
        self.assertTrue(any("- " in chunk.body for chunk in first if "Structured list" in chunk.title_path))
        self.assertTrue(all(" | " not in part for chunk in first for part in chunk.title_path))
        self.assertTrue(all(chunk.body.rstrip().endswith(".") for chunk in long_chunks))

        changed_version = build_chunks(
            (replace(source, version="fixture-v2"),), "hybrid_v1"
        )
        self.assertEqual(
            [chunk.body_sha256 for chunk in first],
            [chunk.body_sha256 for chunk in changed_version],
        )
        self.assertNotEqual(
            [chunk.chunk_id for chunk in first],
            [chunk.chunk_id for chunk in changed_version],
        )

    def test_short_sections_never_merge_across_title_paths(self):
        source = replace(
            self.make_source(),
            sections=(
                SourceSection(
                    title_path=("Document", "First real heading"),
                    topic_tags=("hba1c",),
                    blocks=(SourceBlock("paragraph", words("first", 18)),),
                ),
                SourceSection(
                    title_path=("Document", "Second real heading"),
                    topic_tags=("hba1c",),
                    blocks=(SourceBlock("paragraph", words("second", 18)),),
                ),
            ),
        )

        natural = build_chunks((source,), "natural_section")
        self.assertEqual(
            [
                ("Document", "First real heading"),
                ("Document", "Second real heading"),
            ],
            [chunk.title_path for chunk in natural],
        )
        hybrid = build_chunks((source,), "hybrid_v1")
        self.assertEqual(2, len(hybrid))
        self.assertTrue(all(chunk.estimated_tokens < 80 for chunk in hybrid))
        self.assertTrue(all(chunk.overlap_estimated_tokens == 0 for chunk in hybrid))
        self.assertEqual(
            [chunk.title_path for chunk in natural],
            [chunk.title_path for chunk in hybrid],
        )
        self.assertEqual(
            [chunk.body for chunk in natural],
            [chunk.body for chunk in hybrid],
        )

    def test_default_hybrid_covers_every_source_semantic_unit(self):
        sources = load_source_snapshots(DEFAULT_CORPUS_ROOT)
        chunks = build_chunks(sources, "hybrid_v1")
        chunks_by_section = {}
        for chunk in chunks:
            chunks_by_section.setdefault(
                (chunk.source_id, chunk.title_path), []
            ).append(chunk.body)

        for source in sources:
            for section in source.sections:
                bodies = chunks_by_section.get(
                    (source.source_id, section.title_path), []
                )
                for unit in _section_units(section):
                    with self.subTest(
                        source_id=source.source_id,
                        title_path=section.title_path,
                        semantic_unit=unit.text[:80],
                    ):
                        self.assertTrue(
                            any(unit.text in body for body in bodies),
                            "semantic unit is absent from hybrid chunks",
                        )

    def test_long_paragraph_splits_only_at_complete_sentence_boundaries(self):
        text = sentences("boundary", 24, words_per_sentence=15)
        source = replace(
            self.make_source(),
            sections=(
                SourceSection(
                    title_path=("Document", "Sentence boundary section"),
                    topic_tags=("hba1c",),
                    blocks=(SourceBlock("paragraph", text),),
                ),
            ),
        )

        chunks = build_chunks((source,), "hybrid_v1")

        self.assertGreater(len(chunks), 1)
        self.assertTrue(all(chunk.body.rstrip().endswith(".") for chunk in chunks))
        source_sentences = {sentence.strip() + "." for sentence in text.split(".") if sentence.strip()}
        for chunk in chunks:
            chunk_sentences = {
                sentence.strip() + "."
                for sentence in chunk.body.split(".")
                if sentence.strip()
            }
            self.assertTrue(chunk_sentences <= source_sentences)

    def test_estimator_is_dependency_free_and_explicit(self):
        self.assertEqual(5, estimate_tokens("one two three four"))
        self.assertGreaterEqual(estimate_tokens("空腹血糖"), 4)


if __name__ == "__main__":
    unittest.main()
