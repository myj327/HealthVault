"""Rebuild committed knowledge artifacts from normalized offline snapshots."""

from __future__ import annotations

import argparse
from pathlib import Path

from healthvault.rag.corpus import DEFAULT_CORPUS_ROOT, render_corpus_artifacts


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus-root", type=Path, default=DEFAULT_CORPUS_ROOT)
    parser.add_argument("--corpus-version", default=None)
    args = parser.parse_args(argv)
    corpus_version = args.corpus_version
    if corpus_version is None:
        corpus_version = DEFAULT_CORPUS_ROOT.name if args.corpus_root != DEFAULT_CORPUS_ROOT else "r1-2026-08-24-v1"
    for relative_path, content in render_corpus_artifacts(
        args.corpus_root, corpus_version=corpus_version
    ).items():
        path = args.corpus_root / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
