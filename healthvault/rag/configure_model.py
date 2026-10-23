"""Bind the server to an existing local tokenizer/model without downloading files."""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path

from healthvault.rag.releases import KnowledgeReleaseStore, _write_json
from healthvault.rag.budget import TokenBudget
from healthvault.rag.vector_cache import model_fingerprint


def configure(model: Path, root: Path) -> dict:
    model = model.resolve()
    tokenizer = model / 'tokenizer.json'
    if not tokenizer.is_file(): raise ValueError('Local tokenizer.json is required.')
    # Verify it loads, with truncation/padding disabled by the budget adapter.
    budget = TokenBudget(tokenizer_path=tokenizer)
    fingerprint = model_fingerprint(model)
    destination = root / 'tokenizers' / 'default.json'
    destination.parent.mkdir(parents=True,exist_ok=True)
    shutil.copyfile(tokenizer,destination)
    value = {'model_path':str(model),'model_sha256':fingerprint,'tokenizer_sha256':budget.identity}
    _write_json(root / 'local-model.json',value)
    return value


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model-path',required=True,type=Path)
    parser.add_argument('--release-root',type=Path)
    args = parser.parse_args()
    value = configure(args.model_path,args.release_root or KnowledgeReleaseStore().root)
    print('Local model configured; fingerprint '+value['model_sha256'])


if __name__ == '__main__': main()
