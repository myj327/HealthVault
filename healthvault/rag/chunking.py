"""Deterministic natural-section and structure-aware hybrid chunking."""

from __future__ import annotations

import hashlib
import math
import re
from dataclasses import dataclass


CORPUS_VERSION = "r1-2026-08-24-v1"
TOKEN_ESTIMATOR_VERSION = "char4-cjk-v1"
TARGET_MIN = 250
TARGET_MAX = 450
MINIMUM = 80
MAXIMUM = 600
OVERLAP = 60

_SENTENCE_BOUNDARY_RE = re.compile(r"(?<=[.!?])\s+")


@dataclass(frozen=True)
class SourceBlock:
    kind: str
    text: str
    table_header: str | None = None


@dataclass(frozen=True)
class SourceSection:
    title_path: tuple[str, ...]
    topic_tags: tuple[str, ...]
    blocks: tuple[SourceBlock, ...]


@dataclass(frozen=True)
class SourceSnapshot:
    source_id: str
    publisher: str
    title: str
    version: str
    url: str
    language: str
    retrieved_on: str
    reuse_basis: str
    source_sha256: str
    sections: tuple[SourceSection, ...]
    title_path_whitelist: tuple[tuple[str, ...], ...] = ()


@dataclass(frozen=True)
class KnowledgeChunk:
    chunk_id: str
    citation_id: str
    corpus_version: str
    strategy: str
    source_id: str
    source_version: str
    publisher: str
    topic_tags: tuple[str, ...]
    title_path: tuple[str, ...]
    ordinal: int
    body: str
    estimated_tokens: int
    overlap_estimated_tokens: int
    body_sha256: str
    source_sha256: str
    url: str
    language: str
    retrieved_on: str


@dataclass(frozen=True)
class _BoundaryUnit:
    text: str
    separator_before: str


def estimate_tokens(text: str) -> int:
    """Use a dependency-free chars/4 estimate, counting CJK characters directly."""
    cjk_characters = re.findall(r"[\u3400-\u9fff]", text)
    non_cjk_text = re.sub(r"[\u3400-\u9fff]", "", text)
    return max(1, math.ceil(len(non_cjk_text) / 4) + len(cjk_characters))


def make_chunk_id(
    source_id: str,
    source_version: str,
    title_path: tuple[str, ...],
    ordinal: int,
    *,
    corpus_version: str = CORPUS_VERSION,
) -> str:
    """Build the stable ID from every identity field required by the corpus."""
    identity = "\0".join(
        (source_id, source_version, " / ".join(title_path), str(ordinal))
    )
    digest = hashlib.sha256(identity.encode("utf-8")).hexdigest()[:12]
    return f"HVK:{corpus_version}:{source_id}:{ordinal:03d}:{digest}"


def _render_block(block: SourceBlock) -> str:
    if block.kind == "list":
        return f"- {block.text}"
    if block.kind == "table" and block.table_header:
        return f"{block.table_header}\n{block.text}"
    return block.text


def _sentences(text: str) -> tuple[str, ...]:
    parts = tuple(part.strip() for part in _SENTENCE_BOUNDARY_RE.split(text) if part.strip())
    if not parts:
        return ()
    if any(estimate_tokens(part) > MAXIMUM for part in parts):
        raise ValueError("A sentence exceeds the hybrid maximum boundary")
    return parts


def _section_units(section: SourceSection) -> list[_BoundaryUnit]:
    table = _table_rows(section)
    if table is not None:
        header, rows = table
        return [
            _BoundaryUnit(header, ""),
            *(_BoundaryUnit(row, "\n") for row in rows),
        ]
    units: list[_BoundaryUnit] = []
    for block in section.blocks:
        block_separator = "" if not units else "\n\n"
        if block.kind == "paragraph":
            for index, sentence in enumerate(_sentences(block.text)):
                units.append(
                    _BoundaryUnit(
                        sentence,
                        block_separator if index == 0 else " ",
                    )
                )
        elif block.kind == "list":
            sentences = _sentences(block.text)
            if estimate_tokens(_render_block(block)) <= MAXIMUM:
                units.append(_BoundaryUnit(_render_block(block), block_separator))
            else:
                for index, sentence in enumerate(sentences):
                    units.append(
                        _BoundaryUnit(
                            f"- {sentence}",
                            block_separator if index == 0 else "\n",
                        )
                    )
        else:
            rendered = _render_block(block)
            if estimate_tokens(rendered) > MAXIMUM:
                raise ValueError("A structural block exceeds the hybrid maximum boundary")
            units.append(_BoundaryUnit(rendered, block_separator))
    return units


def _render_units(units: list[_BoundaryUnit]) -> str:
    body = ""
    for unit in units:
        body += (unit.separator_before if body else "") + unit.text
    return body.strip()


def _merge_short_pack(packs: list[list[_BoundaryUnit]]) -> list[list[_BoundaryUnit]]:
    index = 0
    while index < len(packs):
        if estimate_tokens(_render_units(packs[index])) >= MINIMUM:
            index += 1
            continue
        if index + 1 < len(packs):
            merged = [*packs[index], *packs[index + 1]]
            if estimate_tokens(_render_units(merged)) <= MAXIMUM:
                packs[index:index + 2] = [merged]
                continue
        if index > 0:
            merged = [*packs[index - 1], *packs[index]]
            if estimate_tokens(_render_units(merged)) <= MAXIMUM:
                packs[index - 1:index + 1] = [merged]
                index -= 1
                continue
        raise ValueError("A title_path cannot meet the hybrid minimum boundary")
    return packs


def _pack_units(units: list[_BoundaryUnit]) -> list[list[_BoundaryUnit]]:
    if not units:
        return []
    if estimate_tokens(_render_units(units)) < MINIMUM:
        raise ValueError("A title_path cannot meet the hybrid minimum boundary")
    packs: list[list[_BoundaryUnit]] = []
    current: list[_BoundaryUnit] = []
    for unit in units:
        if estimate_tokens(unit.text) > MAXIMUM:
            raise ValueError("A semantic unit exceeds the hybrid maximum boundary")
        candidate = [*current, unit]
        candidate_tokens = estimate_tokens(_render_units(candidate))
        current_tokens = estimate_tokens(_render_units(current)) if current else 0
        if current and candidate_tokens > TARGET_MAX and current_tokens >= MINIMUM:
            packs.append(current)
            current = [unit]
        elif candidate_tokens <= MAXIMUM:
            current = candidate
        else:
            if not current:
                raise ValueError("A semantic unit exceeds the hybrid maximum boundary")
            packs.append(current)
            current = [unit]
    if current:
        packs.append(current)
    return _merge_short_pack(packs)


def _align_overlap_boundaries(
    packs: list[list[_BoundaryUnit]],
) -> list[list[_BoundaryUnit]]:
    for index in range(len(packs) - 1):
        while True:
            try:
                _overlap_suffix(packs[index])
                break
            except ValueError:
                if len(packs[index]) <= 1:
                    raise
                moved = packs[index].pop()
                packs[index + 1].insert(0, moved)
                if (
                    estimate_tokens(_render_units(packs[index])) < MINIMUM
                    or estimate_tokens(_render_units(packs[index + 1])) > MAXIMUM
                ):
                    raise ValueError(
                        "No semantic pack boundary can provide valid overlap"
                    )
    return packs


def _overlap_suffix(units: list[_BoundaryUnit]) -> tuple[str, int]:
    for start in range(len(units) - 1, -1, -1):
        text = _render_units(units[start:])
        tokens = estimate_tokens(text)
        if 40 <= tokens <= 80:
            return text, tokens
        if tokens > 80:
            break
    raise ValueError("No complete semantic boundary can provide 40-80 overlap tokens")


def _table_rows(section: SourceSection) -> tuple[str, list[str]] | None:
    if not section.blocks or any(block.kind != "table" for block in section.blocks):
        return None
    header = next(
        (block.table_header for block in section.blocks if block.table_header),
        None,
    )
    rows: list[str] = []
    for block in section.blocks:
        if block.table_header:
            header = block.table_header
            rows.append(block.text)
            continue
        lines = [line.strip() for line in block.text.splitlines() if line.strip()]
        if header is None and lines:
            header = lines.pop(0)
        rows.extend(lines)
    if header is None or not rows:
        return None
    return header, rows


def _table_bodies(section: SourceSection) -> list[tuple[str, int]] | None:
    parsed = _table_rows(section)
    if parsed is None:
        return None
    header, rows = parsed
    groups: list[list[str]] = []
    current: list[str] = []
    for row in rows:
        candidate = "\n".join((header, *current, row))
        if current and estimate_tokens(candidate) > TARGET_MAX:
            groups.append(current)
            current = [row]
        else:
            current.append(row)
        if estimate_tokens("\n".join((header, *current))) > MAXIMUM:
            raise ValueError("A table row exceeds the hybrid maximum boundary")
    if current:
        groups.append(current)
    bodies = ["\n".join((header, *group)) for group in groups]
    if len(bodies) > 1 and estimate_tokens(bodies[-1]) < MINIMUM:
        merged = "\n".join((bodies[-2], *bodies[-1].splitlines()[1:]))
        if estimate_tokens(merged) <= MAXIMUM:
            bodies[-2:] = [merged]
    if len(bodies) == 1 and estimate_tokens(bodies[0]) < MINIMUM:
        raise ValueError("A table title_path cannot meet hybrid size boundaries")
    if any(not MINIMUM <= estimate_tokens(body) <= MAXIMUM for body in bodies):
        raise ValueError("A table title_path cannot meet hybrid size boundaries")
    return [(body, 0) for body in bodies]


def _hybrid_bodies(section: SourceSection) -> list[tuple[str, int]]:
    natural_body = "\n\n".join(
        _render_block(block) for block in section.blocks
    ).strip()
    if not natural_body:
        return []
    if estimate_tokens(natural_body) < MINIMUM:
        return [(natural_body, 0)]
    table_bodies = _table_bodies(section)
    if table_bodies is not None:
        return table_bodies
    packs = _pack_units(_section_units(section))
    if not packs:
        return []
    if len(packs) == 1:
        body = _render_units(packs[0])
        return [(body, 0)]
    packs = _align_overlap_boundaries(packs)
    bodies: list[tuple[str, int]] = []
    for index, pack in enumerate(packs):
        own_body = _render_units(pack)
        if index == 0:
            bodies.append((own_body, 0))
            continue
        overlap_text, overlap_tokens = _overlap_suffix(packs[index - 1])
        body = f"{overlap_text}\n\n{own_body}"
        if estimate_tokens(body) > MAXIMUM:
            raise ValueError("Semantic overlap exceeds the hybrid maximum boundary")
        bodies.append((body, overlap_tokens))
    return bodies


def build_chunks(
    sources: tuple[SourceSnapshot, ...],
    strategy: str,
    *,
    corpus_version: str = CORPUS_VERSION,
    token_budget=None,
) -> tuple[KnowledgeChunk, ...]:
    if strategy not in {"natural_section", "hybrid_v1"}:
        raise ValueError("Unsupported chunk strategy")
    chunks = []
    for source in sorted(sources, key=lambda item: item.source_id):
        source_ordinal = 0
        for section in source.sections:
            if strategy == "natural_section":
                bodies = [
                    (
                        "\n\n".join(_render_block(block) for block in section.blocks),
                        0,
                    )
                ]
            else:
                if token_budget is None:
                    bodies = _hybrid_bodies(section)
                else:
                    prefix = "\n".join((" / ".join(section.title_path), " ".join(section.topic_tags), ""))
                    natural = "\n\n".join(_render_block(block) for block in section.blocks).strip()
                    parsed = _table_rows(section)
                    bodies = token_budget.split(natural, prefix=prefix, header=parsed[0] if parsed else "")
            for body, overlap in bodies:
                body = body.strip()
                if not body:
                    continue
                estimated_tokens = estimate_tokens(body)
                if strategy == "hybrid_v1":
                    natural_body = "\n\n".join(
                        _render_block(block) for block in section.blocks
                    ).strip()
                    short_intact = (
                        estimated_tokens < MINIMUM
                        and body == natural_body
                        and overlap == 0
                    )
                    if estimated_tokens > MAXIMUM or (
                        estimated_tokens < MINIMUM and not short_intact and token_budget is None
                    ):
                        raise ValueError("Hybrid chunk violates size boundaries")
                chunk_id = make_chunk_id(
                    source.source_id,
                    source.version,
                    section.title_path,
                    source_ordinal,
                    corpus_version=corpus_version,
                )
                chunks.append(
                    KnowledgeChunk(
                        chunk_id=chunk_id,
                        citation_id=chunk_id,
                        corpus_version=corpus_version,
                        strategy=strategy,
                        source_id=source.source_id,
                        source_version=source.version,
                        publisher=source.publisher,
                        topic_tags=section.topic_tags,
                        title_path=section.title_path,
                        ordinal=source_ordinal,
                        body=body,
                        estimated_tokens=estimated_tokens,
                        overlap_estimated_tokens=overlap,
                        body_sha256=hashlib.sha256(body.encode("utf-8")).hexdigest(),
                        source_sha256=source.source_sha256,
                        url=source.url,
                        language=source.language,
                        retrieved_on=source.retrieved_on,
                    )
                )
                source_ordinal += 1
    if not chunks:
        raise ValueError("Chunk strategy produced no chunks")
    return tuple(chunks)
