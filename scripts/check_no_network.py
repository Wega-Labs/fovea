#!/usr/bin/env python3
"""Fail when runtime source introduces networking outside the pinned model fetcher."""

from __future__ import annotations

import ast
import sys
from dataclasses import dataclass
from pathlib import Path

NETWORK_MODULES = frozenset(
    {
        "aiohttp",
        "http",
        "httpx",
        "requests",
        "socket",
        "urllib",
        "websocket",
        "websockets",
    }
)
DEFAULT_ALLOWED = frozenset({Path("fovea/webcam/model.py")})


@dataclass(frozen=True, slots=True)
class Violation:
    path: Path
    line: int
    detail: str


def find_network_usage(
    source_root: Path,
    *,
    allowed: frozenset[Path] = DEFAULT_ALLOWED,
) -> tuple[Violation, ...]:
    violations: list[Violation] = []
    for path in sorted(source_root.rglob("*.py")):
        relative = path.relative_to(source_root)
        if relative in allowed:
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    root = alias.name.partition(".")[0]
                    if root in NETWORK_MODULES:
                        violations.append(Violation(relative, node.lineno, alias.name))
            elif isinstance(node, ast.ImportFrom) and node.module is not None:
                root = node.module.partition(".")[0]
                if root in NETWORK_MODULES:
                    violations.append(Violation(relative, node.lineno, node.module))
    return tuple(violations)


def main(argv: list[str] | None = None) -> int:
    args = sys.argv[1:] if argv is None else argv
    source_root = Path(args[0]) if args else Path("src")
    violations = find_network_usage(source_root)
    for violation in violations:
        print(
            f"{source_root / violation.path}:{violation.line}: "
            f"network module outside pinned model downloader: {violation.detail}",
            file=sys.stderr,
        )
    return 1 if violations else 0


if __name__ == "__main__":
    raise SystemExit(main())
