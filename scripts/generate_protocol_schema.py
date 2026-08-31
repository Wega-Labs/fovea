#!/usr/bin/env python3
"""Regenerate the committed Fovea protocol v1 JSON Schema."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from fovea.protocol import protocol_schema_text  # noqa: E402


def main() -> int:
    destination = ROOT / "schema" / "fovea-protocol-v1.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(protocol_schema_text(), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
