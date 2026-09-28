#!/usr/bin/env python3
"""Audit the policy-visible portion of a collected rollout before export."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from env_factory.evidence.material_privacy import audit_rollout_privacy


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rollout", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rollout = json.loads(args.rollout.read_text(encoding="utf-8"))
    report = audit_rollout_privacy(rollout)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["eligible_for_policy_training_export"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
