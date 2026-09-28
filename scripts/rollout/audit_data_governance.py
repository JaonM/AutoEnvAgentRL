#!/usr/bin/env python3
"""Audit a sandbox's data origin and external-model payloads."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from dotenv import load_dotenv

from env_factory.evidence.data_governance import audit
from env_factory.evidence.model_roles import resolve_model_roles


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    project = Path(__file__).resolve().parents[2]
    load_dotenv(project / ".env")
    roles = resolve_model_roles(os.environ)
    report = audit(
        args.root.resolve(),
        agent_base_url=roles["agent"]["base_url"],
        agent_model=roles["agent"]["model"],
        runtime_base_url=roles["runtime"]["base_url"],
        runtime_model=roles["runtime"]["model"],
    )
    output = args.output or args.root / "data_governance.json"
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["eligible_for_external_model_processing"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
