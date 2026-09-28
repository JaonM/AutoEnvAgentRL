#!/usr/bin/env python3
"""Validate the canonical sandbox Docker build context before image creation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from env_factory.evidence.material_artifacts import (
    docker_build_context_digest,
    docker_context_errors,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    args = parser.parse_args()
    errors = docker_context_errors(args.root.resolve())
    report = {
        "version": "1.0",
        "valid": not errors,
        "errors": errors,
        "context_sha256": (
            docker_build_context_digest(args.root.resolve()) if not errors else None
        ),
    }
    print(json.dumps(report, ensure_ascii=False))
    return 0 if not errors else 1


if __name__ == "__main__":
    raise SystemExit(main())
