"""Versioned, lossless boundary fallback with local tokenizer input budgets."""
from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
from pathlib import Path
from typing import Any


class TokenBudget:
    limit = 600

    def __init__(self, tokenizer_path: Path | None = None):
        self.tokenizer = None
        self.identity = "utf8-byte-bound-v1"
        self.spans: list[dict[str, Any]] = []
        if tokenizer_path is not None:
            from tokenizers import Tokenizer
            self.identity = hashlib.sha256(tokenizer_path.read_bytes()).hexdigest()
            self.tokenizer = Tokenizer.from_file(str(tokenizer_path))
            self.tokenizer.no_truncation()
            self.tokenizer.no_padding()

    def count(self, text: str) -> int:
        return len(self.tokenizer.encode(text).ids) if self.tokenizer else len(text.encode("utf-8")) + 8

    def fits(self, text: str, prefix: str = "") -> bool:
        from healthvault.rag.chunking import MAXIMUM, estimate_tokens
        return self.count(prefix + text) <= self.limit and estimate_tokens(text) <= MAXIMUM

    @classmethod
    def install(cls, root: Path, tokenizer_path: Path | None = None) -> None:
        configured = os.environ.get("HEALTHVAULT_RAG_TOKENIZER_PATH")
        model = os.environ.get("HEALTHVAULT_HYBRID_MODEL_PATH")
        tokenizer_path = tokenizer_path or (Path(configured) if configured else Path(model) / "tokenizer.json" if model else None)
        cached = root.parents[2] / "tokenizers" / "default.json"
        if tokenizer_path is None and cached.is_file():
            tokenizer_path = cached
        root.mkdir(parents=True, exist_ok=True)
        if tokenizer_path is not None:
            shutil.copyfile(tokenizer_path, root / "tokenizer.json")
        budget = cls(root / "tokenizer.json" if tokenizer_path else None)
        (root / "chunk-policy.json").write_text(json.dumps({"version": "bounded-v2", "tokenizer": budget.identity,
                                                           "maximum_input_tokens": budget.limit}, sort_keys=True), encoding="utf-8")

    @classmethod
    def load(cls, root: Path) -> TokenBudget | None:
        path = root / "chunk-policy.json"
        if not path.is_file():
            return None
        value = json.loads(path.read_text(encoding="utf-8"))
        if set(value) != {"version", "tokenizer", "maximum_input_tokens"} or value["version"] != "bounded-v2" or value["maximum_input_tokens"] != cls.limit:
            raise ValueError("Unsupported chunk budget policy.")
        budget = cls(root / "tokenizer.json" if value["tokenizer"] != "utf8-byte-bound-v1" else None)
        if value["tokenizer"] != budget.identity:
            raise ValueError("Chunk tokenizer hash mismatch.")
        return budget

    def split(self, text: str, *, prefix: str, header: str = "") -> list[tuple[str, int]]:
        # Character spans preserve Unicode exactly; the tokenizer only measures
        # candidate windows. No token decode/truncation can discard source text.
        if not self.fits("a", prefix):
            raise ValueError("Knowledge title metadata exceeds the input budget.")
        if header and not self.fits(header + "\n" + "a" * 100, prefix):
            header = ""  # The complete header remains in the original text spans.
        start = 0
        bodies = []
        while start < len(text):
            context = header + "\n" if header and start > 0 else ""
            low, high = start + 1, len(text)
            end = start
            while low <= high:
                middle = (low + high) // 2
                if self.fits(context + text[start:middle], prefix):
                    end = middle; low = middle + 1
                else:
                    high = middle - 1
            if end == start:
                raise ValueError("No source character fits the configured input budget.")
            reason = "token_window"
            if end < len(text):
                candidates = list(re.finditer(r"[。！？.!?；;\n]|\s+|\|", text[start:end]))
                boundary = next((match for match in reversed(candidates) if match.end() >= (end - start) // 2), None)
                if boundary:
                    end = start + boundary.end(); reason = "structural_boundary"
            else:
                reason = "section_end"
            body = context + text[start:end]
            assert self.fits(body, prefix)
            self.spans.append({"start": start, "end": end, "fallback": reason, "context_header": bool(context),
                               "input_tokens": self.count(prefix + body), "overlap_tokens": 0})
            bodies.append((body, 0))
            start = end
        return bodies
