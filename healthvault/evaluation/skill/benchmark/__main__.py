"""Run the TASK-021 zero-API fake smoke and refresh bound outputs."""

from __future__ import annotations

import json

from .core import write_default_outputs


def main() -> int:
    print(json.dumps(write_default_outputs(), ensure_ascii=False, sort_keys=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
